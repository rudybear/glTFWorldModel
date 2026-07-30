"""Batched, differentiable quaternion/rotation math for the dynamics model.

All quaternions use the same convention as ``gltfworld.scene.contract``:
**xyzw** (scalar last), unit-norm, hemisphere-normalized so ``w >= 0``
(a rotation and its negation represent the same element of SO(3); pinning
the sign makes numeric comparison well defined). Every function here is pure
PyTorch (no autograd-breaking numpy round trips) and batched over an
arbitrary number of leading dims (``(..., 4)`` for quats, ``(..., 3)`` for
vectors, ``(..., 6)`` for the 6D rotation representation).

Cross-checked against ``scipy.spatial.transform.Rotation`` in
``tests/test_rotations.py`` (scipy's ``Rotation.from_quat``/``as_quat``/
``from_rotvec`` use the same xyzw / axis-angle conventions, so comparisons
are direct, no convention translation needed).

**6D rotation representation** (Zhou et al. 2019, "On the Continuity of
Rotation Representations in Neural Networks"): the first two columns of the
3x3 rotation matrix, concatenated into a 6-vector. Recovered back to a valid
orthonormal matrix via Gram-Schmidt. Used as the dynamics model's rotation
*input* feature (continuous, no double-cover/wrap-around discontinuity a raw
quaternion or Euler angle would have) -- the model still predicts rotation
*updates* as axis-angle (via :func:`axis_angle_to_quat`, the exponential map)
and composes them onto the previous-step quaternion, per DESIGN.md.

**Geodesic angle** (:func:`quat_geodesic_angle`): computed via
``2 * atan2(||q1 - q2||, ||q1 + q2||)`` after resolving the double-cover sign
ambiguity (flipping ``q2`` to whichever hemisphere is closer to ``q1``, not
just the canonical ``w >= 0`` hemisphere), rather than the textbook
``2 * arccos(|dot(q1, q2)|)``. Both give the same value in exact arithmetic,
but ``arccos``'s derivative diverges as its argument approaches +/-1 (i.e.
exactly where most training pairs land -- two nearby physics states one
30Hz frame apart are usually only a few degrees apart), which would blow up
the loss gradient right where most of the training signal actually lives.
The ``atan2``-of-norms form has a well-behaved gradient everywhere on
``[0, pi]``, including at 0.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


# --- basic quaternion ops -----------------------------------------------------


def quat_normalize(q: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Unit-normalize ``q`` (..., 4) xyzw, safe for a (near-)zero input."""
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=eps)
    return q / norm


def quat_hemisphere(q: torch.Tensor) -> torch.Tensor:
    """Flip ``q`` (..., 4) xyzw so its ``w`` (last) component is >= 0.

    Same convention as ``gltfworld.scene.contract._hemisphere_normalize``,
    just batched torch instead of numpy.
    """
    sign = torch.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 * q2`` for xyzw quaternions (..., 4), batched.

    Composition convention matches ``scipy``: ``quat_multiply(q1, q2)``
    represents "apply ``q2`` first, then ``q1``" (``(R1 * R2).apply(v) ==
    R1.apply(R2.apply(v))``).
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse, for a unit quaternion) of ``q`` (..., 4) xyzw."""
    return torch.cat([-q[..., 0:3], q[..., 3:4]], dim=-1)


# --- axis-angle exponential map -----------------------------------------------


def axis_angle_to_quat(r: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Exponential map: rotation vector ``r`` (..., 3) (axis * angle, radians)
    -> unit quaternion (..., 4) xyzw.

    ``q = (sin(theta/2) * axis, cos(theta/2))``, ``theta = ||r||``,
    ``axis = r / theta``. Stable at ``theta -> 0`` via a Taylor expansion of
    ``sin(theta/2) / theta`` (which has a well-defined ``-> 0.5`` limit, not
    a ``0/0`` singularity) instead of dividing by a clamped-away-from-zero
    ``theta`` directly.
    """
    theta = torch.linalg.norm(r, dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta < eps
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    # sin(half) / theta = 0.5 - theta^2/48 + O(theta^4) as theta -> 0.
    small_coeff = 0.5 - (theta * theta) / 48.0
    coeff = torch.where(small, small_coeff, torch.sin(half) / safe_theta)
    xyz = r * coeff
    w = torch.cos(half)
    return torch.cat([xyz, w], dim=-1)


def quat_to_axis_angle(q: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Logarithmic map, the inverse of :func:`axis_angle_to_quat`: unit
    quaternion ``q`` (..., 4) xyzw -> rotation vector ``r`` (..., 3)
    (axis * angle, radians).

    ``q`` is hemisphere-normalized first (``w >= 0``), which pins ``theta``
    (``= ||r||``) to the principal range ``[0, pi]`` -- the minimal rotation
    angle representing ``q``, not the "other way around" ``2*pi - theta``
    cover. ``theta = 2 * atan2(||xyz||, w)`` (well-behaved for any ``w``,
    unlike ``2 * arccos(w)`` whose derivative blows up as ``w -> +-1``, the
    same reason ``quat_geodesic_angle`` avoids ``arccos``); ``axis = xyz /
    ||xyz||``, recovered stably at ``theta -> 0`` via a Taylor expansion of
    ``theta / sin(theta/2)`` (which has a well-defined ``-> 2`` limit, not a
    ``0/0`` singularity) rather than dividing by a clamped-away-from-zero
    ``||xyz||`` directly -- the same style :func:`axis_angle_to_quat` itself
    uses for its own small-angle branch. Cross-checked against
    ``scipy.spatial.transform.Rotation.as_rotvec()`` and as an exact
    round-trip partner of :func:`axis_angle_to_quat` in
    ``tests/test_rotations.py``.
    """
    q = quat_hemisphere(quat_normalize(q, eps=eps))
    xyz = q[..., 0:3]
    w = q[..., 3:4].clamp(min=-1.0, max=1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(sin_half, w)
    small = theta < eps
    safe_sin_half = torch.where(small, torch.ones_like(sin_half), sin_half)
    # theta / sin(theta/2) = 2 + theta^2/12 + O(theta^4) as theta -> 0.
    small_coeff = 2.0 + (theta * theta) / 12.0
    coeff = torch.where(small, small_coeff, theta / safe_sin_half)
    return xyz * coeff


# --- quaternion <-> rotation matrix -------------------------------------------


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion ``q`` (..., 4) xyzw -> rotation matrix (..., 3, 3)."""
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _sqrt_positive_part(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """``sqrt(max(x, 0))``, gradient-safe (never sqrt's a negative number)."""
    return torch.sqrt(torch.clamp(x, min=eps))


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``m`` (..., 3, 3) -> unit quaternion (..., 4) xyzw,
    hemisphere-normalized (``w >= 0``).

    Numerically stable, branchless (via ``torch.where``/``argmax`` selection
    rather than Python-level ``if``, so it batches and backprops cleanly):
    the standard "largest of the four Shepperd candidates" method, adapted
    from the batched formulation in pytorch3d's
    ``transforms.matrix_to_quaternion`` (wxyz there; xyzw here to match this
    project's convention).
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )  # (..., 4): [qw_abs, qx_abs, qy_abs, qz_abs]

    quat_by_case = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # (..., 4 cases, 4 wxyz components)

    denom = (2.0 * q_abs).clamp(min=0.1).unsqueeze(-1)
    quat_candidates_wxyz = quat_by_case / denom  # (..., 4 cases, 4 wxyz components)

    best_case = torch.argmax(q_abs, dim=-1)  # (...,)
    wxyz = torch.gather(
        quat_candidates_wxyz,
        dim=-2,
        index=best_case[..., None, None].expand(*best_case.shape, 1, 4),
    ).squeeze(-2)  # (..., 4) wxyz

    xyzw = torch.cat([wxyz[..., 1:4], wxyz[..., 0:1]], dim=-1)
    return quat_hemisphere(quat_normalize(xyzw))


# --- quaternion <-> 6D rotation representation --------------------------------


def matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> 6D representation (..., 6): the first
    two columns, flattened (col0 then col1, each 3 components)."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D representation (..., 6) -> orthonormal rotation matrix (..., 3, 3)
    via Gram-Schmidt (Zhou et al. 2019, section 3.3)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=_EPS)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - a2_proj, dim=-1, eps=_EPS)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) xyzw -> 6D rotation representation (..., 6)."""
    return matrix_to_rotation_6d(quat_to_matrix(q))


def sixd_to_quat(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation representation (..., 6) -> unit quaternion (..., 4) xyzw."""
    return matrix_to_quat(rotation_6d_to_matrix(d6))


# --- geodesic distance ---------------------------------------------------------


def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Geodesic angle (radians, in ``[0, pi]``) between unit quaternions
    ``q1``, ``q2`` (..., 4) xyzw -- the rotation angle of ``q1^-1 * q2``.

    See the module docstring for why this uses the ``atan2``-of-norms form
    rather than ``2 * arccos(|dot|)``. Derivation of the ``4 *`` factor: let
    ``a`` be the angle between the (hemisphere-aligned) quaternions as plain
    R^4 unit vectors, so ``cos(a) = dot(q1, q2_aligned)``; the half-angle
    identities ``||q1 - q2|| = 2*sin(a/2)``, ``||q1 + q2|| = 2*cos(a/2)``
    give ``atan2(diff_norm, sum_norm) = a/2``. Since
    ``cos(a) = |dot(q1, q2)| = cos(theta/2)`` (``theta`` the rotation angle),
    ``a = theta/2``, so ``theta = 2*a = 4 * atan2(diff_norm, sum_norm)``.
    """
    q1 = quat_normalize(q1, eps=eps)
    q2 = quat_normalize(q2, eps=eps)
    dot = (q1 * q2).sum(dim=-1, keepdim=False)
    q2_aligned = torch.where(dot[..., None] < 0, -q2, q2)
    diff_norm = torch.linalg.norm(q1 - q2_aligned, dim=-1)
    sum_norm = torch.linalg.norm(q1 + q2_aligned, dim=-1)
    return 4.0 * torch.atan2(diff_norm, sum_norm)
