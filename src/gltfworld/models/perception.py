"""``PerceptionDETR``: single RGB frame -> a set of objects (pose, size, shape,
semantic class, existence). A DETR-lite: a ViT-style patch encoder feeds a
small transformer decoder over a fixed number of learned "object query"
tokens (Carion et al. 2020's set-prediction recipe, scaled down).

Architecture
------------

- **Patch embed**: ``256x256x3`` RGB, normalized from the dataset's ``[0, 1]``
  range to ``[-1, 1]`` *inside* :meth:`PerceptionDETR.forward` (so callers can
  keep feeding whatever :class:`~gltfworld.data.dataset.PerceptionDataset`
  already returns), then a single ``Conv2d(3, d_model, kernel=16, stride=16)``
  (a linear projection of each non-overlapping 16x16 patch) -> ``256`` tokens
  (``16x16`` patch grid), plus a learned per-token 2D positional embedding.
- **Encoder**: 6 pre-norm ``nn.TransformerEncoderLayer``s, ``d_model=256``, 8
  heads, MLP ratio 4 -- architecturally identical in spirit to
  ``gltfworld.models.dynamics.InteractionTransformer``'s encoder stack, just
  over image-patch tokens instead of object tokens.
- **Decoder**: ``N_MAX=5`` learned object-query tokens (one fixed embedding
  per query slot, not derived from the image -- standard DETR), 3 pre-norm
  ``nn.TransformerDecoderLayer``s (self-attention across queries + cross-
  attention to the encoder's image tokens), same ``d_model``/heads/MLP ratio.
- **Heads** (one shared 2-layer MLP trunk, split into per-field slices of its
  final linear layer -- cheap, and lets every field see the full decoded
  query representation): existence logit (1); position (3, see workspace
  normalization below); rotation as a 6D representation -> quaternion via
  :func:`gltfworld.models.rotations.sixd_to_quat` (continuous, no double-cover
  discontinuity as a *raw* regression target, same rationale
  ``InteractionTransformer`` uses for its rotation *input* feature); size (3);
  shape logits (``len(SHAPE_ORDER) == 3``); class logits
  (``len(CATEGORY_TO_CLASS_ID) == 3``).

**Position workspace normalization**: the raw position head output is passed
through a sigmoid and affinely mapped into a fixed ``[POS_MIN, POS_MAX]`` box
(then that *is* the denormalized world-unit position -- no further scaling),
rather than regressing raw meters directly. This keeps the position head's
pre-activation on a well-conditioned scale (sigmoid saturates gracefully at
the box edges instead of an unbounded linear head that can blow up early in
training) while still covering ``wm-scenes-v1``'s actual sampled range with
margin: objects start with ``x, z in [-0.75, 0.75]`` and a drop height up to
``ground_top + max_bounding_radius(~0.43) + 1.2 ~= 1.65`` m (DESIGN.md's V3
section), and can drift somewhat further over an episode under up to 1.5 m/s
of initial velocity plus gravity/contacts -- ``POS_MIN``/``POS_MAX`` below
give at least 1m of slack on every side of that range. Size is normalized the
same way into ``[SIZE_MIN, SIZE_MAX]`` (``wm-scenes-v1`` samples characteristic
size uniformly in ``[0.05, 0.25]``, DESIGN.md's V3 section).

**V6.3: CNN encoder option (small-data regime)**. ``encoder="vit"`` (default)
is the architecture above, unchanged. ``encoder="cnn"`` swaps the patch-embed
+ positional-embedding + 6-layer from-scratch transformer encoder stack for a
small from-scratch convnet: a stride-1 stem (``3 -> 32`` channels) followed
by 4 stride-2 stages (output channels 32/64/128/256, ``GroupNorm`` + ``SiLU``,
each stage a stack of plain conv blocks -- see :data:`CNN_BLOCKS_PER_STAGE`
for the exact per-stage depth, tuned only to land the whole model's parameter
count in the 6-12M band, not architecturally load-bearing), taking
``256x256`` down to a ``16x16x256`` feature map (4 stride-2 halvings:
``256 / 2**4 == 16``, the same token count -- 256 -- the ``vit`` patch grid
produces), a ``1x1`` conv projecting to ``d_model`` tokens, and a learned
per-token 2D positional embedding (analogous role to the ``vit`` path's
``pos_embed``, separate parameter). The resulting ``(B, 256, d_model)``
token sequence is fed to the *same* decoder/heads below unchanged -- no
transformer self-attention layers over image tokens in this path at all, the
convnet's own stacked local receptive fields do that mixing instead.
**Why**: see DESIGN.md's V6.3 section -- the ``vit`` encoder's from-scratch
transformer has no built-in spatial inductive bias (locality, translation
equivariance) and needs far more data to learn one than this project's
dataset scale provides (V6.1/V6.2 postmortems: 0.461m median val position
error against a 0.05m bar even after fixing dataset scale and an
out-of-box-GT defect, with a persistent train/val gap, while the decoder/
heads/loss pathway itself was independently confirmed healthy via an 8-frame
overfit test). A CNN's built-in bias is the standard fix for exactly this
symptom in a small-data regime.

Output of :meth:`PerceptionDETR.forward`: a dict of per-query tensors, all in
*denormalized* (real-world) units --

- ``existence_logit``: ``(B, N_MAX)`` -- sigmoid -> P(this query is a real object)
- ``position``: ``(B, N_MAX, 3)`` meters
- ``quat``: ``(B, N_MAX, 4)`` xyzw, unit, hemisphere-normalized (via
  :func:`gltfworld.models.rotations.matrix_to_quat`, which already hemisphere-
  normalizes internally)
- ``size``: ``(B, N_MAX, 3)`` meters, same per-shape convention as
  ``gltfworld.scene.scene.ObjectSpec.size``
- ``shape_logits``: ``(B, N_MAX, 3)``, order matches
  ``gltfworld.scene.contract.SHAPE_ORDER``
- ``class_logits``: ``(B, N_MAX, len(CATEGORY_TO_CLASS_ID))``, order matches
  ``gltfworld.scene.contract.CATEGORY_TO_CLASS_ID``

**Parameter count, honestly reported (a documented deviation from the "~12-16M"
approximation, same precedent as DESIGN.md V5's ``NoInteractionMLP``)**: the
architecture description above (patch16/256 tokens, 6 encoder + 3 decoder
layers, ``d_model=256``, 8 heads, MLP ratio 4, 5 queries) was implemented
literally; the actual measured count (printed by
``python -m gltfworld.models.perception``, and recorded in DESIGN.md's V6
section) lands a bit under the milestone spec text's own "~12-16M" estimate.
Per this project's own stated policy (favor the literal, testable
architecture description over an approximate headline number when the two
aren't simultaneously satisfiable), the architecture was kept as specified
rather than padding hidden dims solely to hit a parameter target the spec's
own component list doesn't otherwise ask for.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gltfworld.models.rotations import sixd_to_quat
from gltfworld.scene.contract import CATEGORY_TO_CLASS_ID, SHAPE_ORDER

IMG_SIZE = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 256

D_MODEL = 256
ENC_LAYERS = 6
DEC_LAYERS = 3
N_HEADS = 8
MLP_RATIO = 4
N_MAX = 5

NUM_SHAPES = len(SHAPE_ORDER)  # 3
NUM_CLASSES = len(CATEGORY_TO_CLASS_ID)  # 3

# Workspace bounds the position head's sigmoid output is affinely mapped
# into -- see module docstring for the margin rationale against
# wm-scenes-v1's actual sampled range.
POS_MIN = (-2.0, -0.5, -2.0)
POS_MAX = (2.0, 3.0, 2.0)

SIZE_MIN = 0.02
SIZE_MAX = 0.35

# head output layout: existence(1) + position_raw(3) + rot6d(6) + size_raw(3)
# + shape_logits(NUM_SHAPES) + class_logits(NUM_CLASSES)
_HEAD_SPLIT = (1, 3, 6, 3, NUM_SHAPES, NUM_CLASSES)
HEAD_OUT_DIM = sum(_HEAD_SPLIT)

ENCODER_TYPES = ("vit", "cnn")

# CNN encoder (V6.3): stem + 4 stride-2 stages, output channels per the
# module docstring's "channels ~32/64/128/256" spec. Depth per stage
# (CNN_BLOCKS_PER_STAGE) is not architecturally load-bearing -- it is tuned
# only to land the *whole model's* (CNN encoder + existing decoder/heads)
# parameter count in the 6-12M target band; see
# `python -m gltfworld.models.perception`'s printed count.
CNN_STAGE_CHANNELS = (32, 64, 128, 256)
CNN_BLOCKS_PER_STAGE = (2, 2, 3, 5)
CNN_GROUPNORM_GROUPS = 8


class _PatchEmbed(nn.Module):
    """Non-overlapping 16x16 patch linear projection (a strided conv is
    exactly a per-patch linear layer applied at every grid location)."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, num_patches, d_model)
        x = self.proj(x)
        b, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)


class _ConvGNSiLU(nn.Module):
    """One ``Conv2d`` + ``GroupNorm`` + ``SiLU`` block -- the CNN encoder's
    only building block (see module docstring's V6.3 section)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(CNN_GROUPNORM_GROUPS, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _CNNStage(nn.Module):
    """One stride-2 stage: a first block does the stride-2 downsample +
    channel change, remaining blocks are stride-1 same-channel refinement."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int) -> None:
        super().__init__()
        blocks = [_ConvGNSiLU(in_ch, out_ch, stride=2)]
        blocks += [_ConvGNSiLU(out_ch, out_ch, stride=1) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _CNNEncoder(nn.Module):
    """Small from-scratch convnet: stride-1 stem + 4 stride-2 stages ->
    ``16x16`` feature map -> ``1x1`` conv projection to ``d_model`` tokens +
    learned 2D positional embedding. Replaces the ``vit`` path's patch-embed
    + positional-embedding + transformer-encoder stack wholesale -- output is
    a ``(B, NUM_PATCHES, d_model)`` token sequence fed straight to the
    existing decoder, with no self-attention mixing over image tokens at all
    (the convnet's stacked local receptive fields already do that). See
    module docstring's V6.3 section for the data-regime rationale."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        stem_ch = CNN_STAGE_CHANNELS[0]
        self.stem = _ConvGNSiLU(3, stem_ch, stride=1)

        stages = []
        in_ch = stem_ch
        for out_ch, n_blocks in zip(CNN_STAGE_CHANNELS, CNN_BLOCKS_PER_STAGE):
            stages.append(_CNNStage(in_ch, out_ch, n_blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, num_patches, d_model)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.proj(x)
        b, d, h, w = x.shape
        if h * w != NUM_PATCHES:
            raise ValueError(f"CNN encoder produced a {h}x{w} feature map, expected {IMG_SIZE // PATCH_SIZE} squared")
        tokens = x.flatten(2).transpose(1, 2)
        return tokens + self.pos_embed


class PerceptionDETR(nn.Module):
    """Single RGB frame -> ``N_MAX`` object queries (pose, size, shape, class,
    existence). See module docstring for the full architecture writeup."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        enc_layers: int = ENC_LAYERS,
        dec_layers: int = DEC_LAYERS,
        n_heads: int = N_HEADS,
        mlp_ratio: int = MLP_RATIO,
        n_queries: int = N_MAX,
        encoder: str = "vit",
    ) -> None:
        super().__init__()
        if encoder not in ENCODER_TYPES:
            raise ValueError(f"encoder must be one of {ENCODER_TYPES}, got {encoder!r}")
        self.d_model = d_model
        self.n_queries = n_queries
        self.encoder_type = encoder

        if encoder == "vit":
            self.patch_embed = _PatchEmbed(d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * mlp_ratio,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=enc_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
            )
        else:  # "cnn" -- see module docstring's V6.3 section
            self.cnn_encoder = _CNNEncoder(d_model)

        self.query_embed = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.normal_(self.query_embed, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers, norm=nn.LayerNorm(d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, HEAD_OUT_DIM),
        )

        self.register_buffer("pos_min", torch.tensor(POS_MIN, dtype=torch.float32))
        self.register_buffer("pos_max", torch.tensor(POS_MAX, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """``rgb``: ``(B, H, W, 3)`` float32 in ``[0, 1]`` (the layout
        ``gltfworld.data.dataset.PerceptionDataset`` returns) -> a dict of
        denormalized per-query tensors, see module docstring."""
        if rgb.shape[-3:] != (IMG_SIZE, IMG_SIZE, 3):
            raise ValueError(f"expected rgb (..., {IMG_SIZE}, {IMG_SIZE}, 3), got {tuple(rgb.shape)}")

        x = rgb.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        if self.encoder_type == "vit":
            tokens = self.patch_embed(x) + self.pos_embed
            memory = self.encoder(tokens)
        else:  # "cnn"
            memory = self.cnn_encoder(x)

        b = rgb.shape[0]
        queries = self.query_embed.expand(b, -1, -1)
        decoded = self.decoder(queries, memory)  # (B, n_queries, d_model)

        out = self.head(decoded)
        existence_logit, pos_raw, rot6d, size_raw, shape_logits, class_logits = out.split(_HEAD_SPLIT, dim=-1)

        position = self.pos_min + torch.sigmoid(pos_raw) * (self.pos_max - self.pos_min)
        quat = sixd_to_quat(rot6d)
        size = SIZE_MIN + torch.sigmoid(size_raw) * (SIZE_MAX - SIZE_MIN)

        return {
            "existence_logit": existence_logit.squeeze(-1),  # (B, n_queries)
            "position": position,
            "quat": quat,
            "size": size,
            "shape_logits": shape_logits,
            "class_logits": class_logits,
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PerceptionDETR(encoder="vit")
    n_params = count_params(model)
    print(f"PerceptionDETR(encoder='vit') parameters: {n_params:,}")
    if not (12_000_000 <= n_params <= 16_000_000):
        print(
            f"NOTE: {n_params:,} is outside the milestone spec's approximate '~12-16M' "
            "band -- a documented, deliberate deviation (literal architecture spec kept "
            "as ground truth over the approximate count; see module docstring)."
        )

    cnn_model = PerceptionDETR(encoder="cnn")
    n_cnn_params = count_params(cnn_model)
    print(f"PerceptionDETR(encoder='cnn') parameters: {n_cnn_params:,}")
    if not (6_000_000 <= n_cnn_params <= 12_000_000):
        raise SystemExit(
            f"PerceptionDETR(encoder='cnn') parameter count {n_cnn_params:,} is outside "
            "the V6.3 target 6-12M band -- tune CNN_STAGE_CHANNELS/CNN_BLOCKS_PER_STAGE."
        )
