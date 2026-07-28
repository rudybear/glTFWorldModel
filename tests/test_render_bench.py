"""Rendering throughput benchmark for `EpisodeRenderer` (gpu-marked).

Persistent renderer + a 4-object scene, 500 frames, three variants:

- (a) rgb-only
- (b) rgb+depth
- (c) rgb+depth+seg

Hard floor: (c) must be >= 100 fps. Target is >= 300 fps; if (c) lands
under that, the number is still reported (not hidden), along with a quick
``time.perf_counter`` breakdown of where the time goes.
"""

from __future__ import annotations

import time

import pytest
from conftest import make_sample_episode

pytestmark = pytest.mark.gpu

N_FRAMES = 500
HARD_FLOOR_FPS = 100.0
TARGET_FPS = 300.0


def test_benchmark_rgb_depth_seg_fps(episode_renderer):
    # n_objects=3 -> 4 total objects (the ground box + 3 falling shapes),
    # i.e. the "4-object scene" the spec calls for.
    episode = make_sample_episode(n_objects=3, T=N_FRAMES)
    assert episode.series.num_objects == 4
    episode_renderer.load(episode)

    # Warm up (context/shader/buffer setup shouldn't count against fps).
    for t in range(10):
        episode_renderer.set_frame(t % episode.series.num_frames)
        episode_renderer.render()

    # (a) rgb-only
    t0 = time.perf_counter()
    for t in range(N_FRAMES):
        episode_renderer.set_frame(t)
        _rgb, _ = episode_renderer.render_rgbd()
    dt_a = time.perf_counter() - t0
    fps_a = N_FRAMES / dt_a

    # (b) rgb+depth
    t0 = time.perf_counter()
    for t in range(N_FRAMES):
        episode_renderer.set_frame(t)
        _rgb, _depth = episode_renderer.render_rgbd()
    dt_b = time.perf_counter() - t0
    fps_b = N_FRAMES / dt_b

    # (c) rgb+depth+seg (two GL forward passes per frame: one rgb+depth,
    # one SEG).
    t0 = time.perf_counter()
    for t in range(N_FRAMES):
        episode_renderer.set_frame(t)
        _frame = episode_renderer.render()
    dt_c = time.perf_counter() - t0
    fps_c = N_FRAMES / dt_c

    # Diagnostic breakdown: DEPTH_ONLY skips the lighting bind and the RGB
    # glReadPixels call that render_rgbd() does, so (rgbd - depth_only)
    # approximates "color shading + readback" cost, and (c - b) is the
    # incremental cost of the whole extra SEG pass.
    t0 = time.perf_counter()
    for t in range(N_FRAMES):
        episode_renderer.set_frame(t)
        episode_renderer._render_depth_only()
    dt_depth_only = time.perf_counter() - t0
    fps_depth_only = N_FRAMES / dt_depth_only

    ms = lambda dt: dt * 1000.0 / N_FRAMES  # noqa: E731

    print()
    print(f"=== render benchmark: {N_FRAMES} frames, 4 objects, 256x256 ===")
    print(f"  (a) rgb-only:      {fps_a:9.1f} fps  ({ms(dt_a):.3f} ms/frame)")
    print(f"  (b) rgb+depth:     {fps_b:9.1f} fps  ({ms(dt_b):.3f} ms/frame)")
    print(f"  (c) rgb+depth+seg: {fps_c:9.1f} fps  ({ms(dt_c):.3f} ms/frame)  <- hard floor 100, target 300")
    print("  --- where the time goes ---")
    print(f"  depth-only pass alone:        {fps_depth_only:9.1f} fps  ({ms(dt_depth_only):.3f} ms/frame)")
    print(f"  color shade+readback (b - depth-only): {ms(dt_b) - ms(dt_depth_only):+.3f} ms/frame")
    print(f"  extra SEG pass (c - b):                {ms(dt_c) - ms(dt_b):+.3f} ms/frame")
    print(
        "  note: (a) and (b) issue the *same* underlying pyrender call -- "
        "OffscreenRenderer.render() always reads back both color and depth "
        "together unless RenderFlags.DEPTH_ONLY is set, so there is no "
        "cheaper rgb-only path through the public API; reported separately "
        "anyway per spec, and their near-equality is itself the honest finding."
    )
    if fps_c < TARGET_FPS:
        print(f"  ** (c) is below the {TARGET_FPS:.0f} fps target ({fps_c:.1f} fps) -- see breakdown above **")
    print("=" * 60)

    assert fps_c >= HARD_FLOOR_FPS, f"hard floor violated: (c) rgb+depth+seg = {fps_c:.1f} fps < {HARD_FLOOR_FPS:.0f}"
