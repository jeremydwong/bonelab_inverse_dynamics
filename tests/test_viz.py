"""Headless tests for boneid.viz.

Everything here runs without a browser: meshcat spawns a local zmq/web server
subprocess, we push geometry at it, and pull the static HTML snapshot back.
One viewer is shared by the whole module (starting a server per test is slow)
and torn down by the fixture.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pytest

from boneid.core import GroundWrench, Skeleton, SegmentKinematics
from boneid import viz


N_FRAMES = 20
RATE = 100.0

URDF_PATH = (Path(__file__).resolve().parents[1]
             / "assets" / "urdf" / "humanSubject01_66dof.urdf")


def _html_commands(html: str) -> bytes:
    """The scene commands embedded in a meshcat static page, decoded.

    `static_html` replays the command history into the page as base64
    msgpack in `fetch("data:application/octet-binary;base64,...")` calls, so
    scene paths are NOT literal text in the HTML. Decoding is what lets a test
    assert that a specific object really made it into the snapshot.
    """
    blobs = re.findall(r'base64,([A-Za-z0-9+/=]+)"', html)
    return b"".join(base64.b64decode(b) for b in blobs)


def test_html_commands_helper_sees_a_known_object(vis, skeleton, kin):
    viz.clear_scene(vis)
    viz.draw_skeleton(vis, skeleton, kin, 0)
    assert b"boneid/skeleton/shank" in _html_commands(viz.render_static_html(vis))


# ---------------------------------------------------------------------------
# Fixtures: a tiny 2-segment chain (shank + torso) swinging about the origin
# ---------------------------------------------------------------------------

def _fake_skeleton() -> Skeleton:
    return Skeleton(
        segment_names=["shank", "torso"],
        joint_names=["knee"],
        mass=np.array([3.5, 40.0]),
        com_local=np.array([[0.0, 0.0, -0.2], [0.0, 0.0, -0.25]]),
        inertia_local=np.stack([np.diag([0.05, 0.05, 0.01]),
                                np.diag([1.2, 1.0, 0.4])]),
        length=np.array([0.45, 0.55]),
    )


def _fake_kinematics(skeleton: Skeleton) -> SegmentKinematics:
    """Two stacked segments; the lower one sways in the x-z plane."""
    t = np.arange(N_FRAMES) / RATE
    n_seg = len(skeleton.segment_names)
    r = np.zeros((N_FRAMES, n_seg, 3, 3))
    prox = np.zeros((N_FRAMES, n_seg, 3))
    dist = np.zeros((N_FRAMES, n_seg, 3))
    for i, ti in enumerate(t):
        angle = 0.4 * np.sin(2 * np.pi * 1.0 * ti)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        # segment 0: distal end on the floor, proximal end up the chain
        d0 = np.array([0.0, 0.0, 0.0])
        p0 = d0 + rot @ np.array([0.0, 0.0, skeleton.length[0]])
        r[i, 0] = rot
        prox[i, 0], dist[i, 0] = p0, d0
        # segment 1: sits on top of segment 0, upright
        r[i, 1] = np.eye(3)
        dist[i, 1] = p0
        prox[i, 1] = p0 + np.array([0.0, 0.0, skeleton.length[1]])
    return SegmentKinematics(t=t, rate=RATE, r_world=r, prox_pos=prox,
                             dist_pos=dist)


def _fake_ground() -> GroundWrench:
    t = np.arange(N_FRAMES) / RATE
    force = np.zeros((N_FRAMES, 3))
    force[:, 2] = 600.0 + 200.0 * np.sin(2 * np.pi * t)
    force[:, 0] = 50.0 * np.cos(2 * np.pi * t)
    moment = np.cross(np.tile([0.05, 0.0, 0.0], (N_FRAMES, 1)), force)
    return GroundWrench(t=t, rate=RATE, force=force, moment=moment,
                        point=np.zeros(3))


def _fake_markers() -> dict[str, np.ndarray]:
    t = np.arange(N_FRAMES) / RATE
    sway = 0.1 * np.sin(2 * np.pi * t)
    return {
        "LANK": np.stack([sway, np.full(N_FRAMES, -0.1),
                          np.full(N_FRAMES, 0.08)], axis=1),
        "LKNE": np.stack([sway, np.full(N_FRAMES, -0.1),
                          np.full(N_FRAMES, 0.45)], axis=1),
        "SACR": np.stack([sway * 0.5, np.zeros(N_FRAMES),
                          np.full(N_FRAMES, 0.95)], axis=1),
    }


@pytest.fixture(scope="module")
def skeleton():
    return _fake_skeleton()


@pytest.fixture(scope="module")
def kin(skeleton):
    return _fake_kinematics(skeleton)


@pytest.fixture(scope="module")
def vis():
    v = viz.start_viewer(open_browser=False)
    try:
        yield v
    finally:
        viz.stop_viewer(v)


# ---------------------------------------------------------------------------
# Pure geometry (no viewer needed)
# ---------------------------------------------------------------------------

def test_segment_transform_places_and_stretches():
    m = viz.segment_transform([0.0, 0.0, 0.0], [0.0, 0.0, 2.0])
    assert np.allclose(m[:3, 3], [0.0, 0.0, 1.0])
    # local +Y (the cylinder axis) must map to world +Z with length 2
    assert np.allclose(m[:3, 1], [0.0, 0.0, 2.0])
    # and the frame stays right-handed / orthogonal in the unscaled columns
    assert np.isclose(np.linalg.norm(m[:3, 0]), 1.0)


def test_segment_transform_degenerate_endpoints_is_finite():
    m = viz.segment_transform([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.4)
    assert np.all(np.isfinite(m))


def test_top_segment_is_bulkier(skeleton):
    assert viz.segment_girth(skeleton, 1) > viz.segment_girth(skeleton, 0)
    # girth stays in a plausible band relative to segment length
    for s in range(2):
        assert 0.05 <= viz.segment_girth(skeleton, s) / skeleton.length[s] <= 0.2


def test_arrow_length_follows_force_scale():
    _, _, length = viz.arrow_transforms([0, 0, 0], [0.0, 0.0, 1000.0])
    assert np.isclose(length, 1.0)          # 1 kN -> 1 m, documented scale
    _, _, half = viz.arrow_transforms([0, 0, 0], [0.0, 0.0, 500.0])
    assert np.isclose(half, 0.5)


def test_force_application_point_falls_back_when_unloaded():
    t = np.arange(3) / RATE
    ground = GroundWrench(t=t, rate=RATE, force=np.zeros((3, 3)),
                          moment=np.zeros((3, 3)),
                          point=np.array([0.3, 0.2, 0.0]))
    p = viz.force_application_point(ground, 0)
    assert np.allclose(p, [0.3, 0.2, 0.0])


# ---------------------------------------------------------------------------
# Viewer round-trips
# ---------------------------------------------------------------------------

def test_start_viewer_headless(vis):
    assert vis.url().startswith("http")


def test_draw_skeleton_every_frame(vis, skeleton, kin):
    viz.build_scene(vis, skeleton)
    viz.set_camera(vis)
    for i in range(N_FRAMES):
        viz.draw_skeleton(vis, skeleton, kin, i, build=False)


def test_draw_markers(vis):
    markers = _fake_markers()
    viz.draw_markers(vis, markers, 0)
    for i in range(0, N_FRAMES, 5):
        viz.draw_markers(vis, markers, i, build=False)
    # a static [3] position and a NaN gap must both be tolerated
    viz.draw_markers(vis, {"FIXED": np.array([0.0, 0.0, 1.0]),
                           "GAP": np.full((N_FRAMES, 3), np.nan)}, 3)


def test_draw_ground_force_static(vis, skeleton):
    ground = _fake_ground()
    viz.build_scene(vis, skeleton)
    for i in range(0, N_FRAMES, 4):
        viz.draw_ground_force(vis, ground, i)


def test_animate_builds_clips(vis, skeleton, kin):
    ground = _fake_ground()
    animation = viz.animate(vis, skeleton, kin, ground=ground,
                            realtime_scale=1.0, decimate=1)
    assert animation.clips, "no animation clips were created"
    # one clip per animated path: 2 segments + 2 joint balls + shaft + head
    assert len(animation.clips) == 6
    lowered = animation.lower()
    keys = lowered[0]["clip"]["tracks"][0]["keys"]
    assert len(keys) == N_FRAMES
    assert np.isclose(lowered[0]["clip"]["fps"], RATE)


def test_animate_decimate_and_realtime_scale(vis, skeleton, kin):
    animation = viz.animate(vis, skeleton, kin, realtime_scale=0.5, decimate=4)
    lowered = animation.lower()
    assert len(lowered[0]["clip"]["tracks"][0]["keys"]) == N_FRAMES // 4
    assert np.isclose(lowered[0]["clip"]["fps"], RATE / 4 * 0.5)


def test_render_static_html(vis, skeleton, kin):
    viz.animate(vis, skeleton, kin, ground=_fake_ground())
    html = viz.render_static_html(vis)
    assert isinstance(html, str)
    assert len(html) > 100_000
    assert "<html>" in html.lower()
    # the viewer payload itself must be in there, not just a stub page
    assert "MeshCat" in html or "meshcat" in html
    assert "set_object" in html or "set_transform" in html


def test_save_html(vis, skeleton, kin, tmp_path):
    viz.animate(vis, skeleton, kin, ground=_fake_ground())
    out = tmp_path / "nested" / "trial.html"
    written = viz.save_html(vis, out)
    assert written == str(out)
    assert out.stat().st_size > 100 * 1024


def test_clear_scene(vis, skeleton, kin):
    viz.draw_skeleton(vis, skeleton, kin, 0)
    viz.clear_scene(vis)


# ---------------------------------------------------------------------------
# A whole body: two legs + one pelvis band + one mass-sized torso
# ---------------------------------------------------------------------------
#
# The failure this guards against is visual and was seen in screenshots: legs
# ending in mid-air, a thin pelvis floating above a gap, and no upper body at
# all despite ~42 kg living above L5S1. `make_screenshot.py` renders the same
# scene through playwright.

HIP_HALF_SPAN = 0.09        # m, half the hip-to-hip distance
THIGH_LEN = 0.42
SHANK_LEN = 0.42
FOOT_LEN = 0.18
TORSO_MASS = 42.0           # kg above L5S1 — the mass the model does not carry
TORSO_LEN = 0.50            # m, L5S1 -> mid-acromion


def _leg_skeleton() -> Skeleton:
    """A 3-segment leg, distal->proximal: foot, shank, thigh."""
    return Skeleton(
        segment_names=["foot", "shank", "thigh"],
        joint_names=["ankle", "knee"],
        mass=np.array([1.0, 3.5, 8.0]),
        com_local=np.array([[0.0, 0.0, -0.09], [0.0, 0.0, -0.18],
                            [0.0, 0.0, -0.18]]),
        inertia_local=np.stack([np.diag([0.01, 0.01, 0.005]),
                                np.diag([0.05, 0.05, 0.01]),
                                np.diag([0.14, 0.14, 0.03])]),
        length=np.array([FOOT_LEN, SHANK_LEN, THIGH_LEN]),
    )


def _fake_body(n_frames: int = N_FRAMES) -> dict:
    """Two legs walking out of phase under one pelvis; plus the trunk points.

    Returns the pieces a caller of `animate` needs: per-side (skeleton, kin),
    the two hip trajectories, L5S1 and mid-acromion.
    """
    t = np.arange(n_frames) / RATE
    phase = 2 * np.pi * 1.0 * t
    hip_z = 0.92 + 0.02 * np.sin(2 * phase)
    body_x = 0.03 * np.sin(phase)

    l5s1 = np.stack([body_x, np.zeros(n_frames), hip_z + 0.10], axis=1)
    acromion = np.stack([body_x, np.zeros(n_frames), hip_z + 0.10 + TORSO_LEN],
                        axis=1)

    out = {"l5s1": l5s1, "acromion": acromion, "t": t, "legs": {}}
    for side, (sign, lag) in {"r": (-1.0, 0.0), "l": (+1.0, np.pi)}.items():
        hip = np.stack([body_x, np.full(n_frames, sign * HIP_HALF_SPAN),
                        hip_z], axis=1)
        swing = 0.5 * np.sin(phase + lag)          # thigh angle, rad
        knee = hip + THIGH_LEN * np.stack(
            [np.sin(swing), np.zeros(n_frames), -np.cos(swing)], axis=1)
        shank_ang = swing - 0.35 * (1.0 + np.sin(phase + lag))
        ankle = knee + SHANK_LEN * np.stack(
            [np.sin(shank_ang), np.zeros(n_frames), -np.cos(shank_ang)], axis=1)
        toe = ankle + FOOT_LEN * np.stack(
            [np.full(n_frames, 0.95), np.zeros(n_frames),
             np.full(n_frames, -0.3)], axis=1)

        prox = np.stack([ankle, knee, hip], axis=1)      # [T,3seg,3]
        dist = np.stack([toe, ankle, knee], axis=1)
        r = np.tile(np.eye(3), (n_frames, 3, 1, 1))
        out["legs"][side] = {
            "skeleton": _leg_skeleton(),
            "kin": SegmentKinematics(t=t, rate=RATE, r_world=r,
                                     prox_pos=prox, dist_pos=dist),
            "hip": hip,
        }
    return out


@pytest.fixture(scope="module")
def body():
    return _fake_body()


def test_pelvis_band_radius_is_wider_than_a_thigh(body):
    r = viz.pelvis_band_radius(body["legs"]["l"]["hip"], body["legs"]["r"]["hip"])
    # 55% of the hip-to-hip half distance
    assert np.isclose(r, viz.PELVIS_BAND_FRACTION * HIP_HALF_SPAN)
    # ...and fatter than the thigh cylinders that plug into it, or the thighs
    # would poke out the sides instead of joining one pelvis.
    # limb girth is GIRTH_FRACTION of segment length (the 4-segment report
    # chain gives the thigh exactly this; here the thigh is the chain top, so
    # compute the limb figure directly rather than reading the bulked-up one).
    assert r > viz.GIRTH_FRACTION * THIGH_LEN


def test_torso_radius_follows_the_mass_it_must_honour():
    # documented worked example: 42 kg over 0.5 m at 1000 kg/m^3 -> ~0.16 m
    r = viz.torso_radius(TORSO_MASS, TORSO_LEN)
    assert 0.15 < r < 0.18
    # and it really is the inverse of m = rho * pi * r^2 * h
    assert np.isclose(viz.BODY_DENSITY * np.pi * r ** 2 * TORSO_LEN, TORSO_MASS)
    # heavier -> fatter, taller -> thinner (same mass spread further)
    assert viz.torso_radius(2 * TORSO_MASS, TORSO_LEN) > r
    assert viz.torso_radius(TORSO_MASS, 2 * TORSO_LEN) < r
    # degenerate height must not divide by zero
    assert np.isfinite(viz.torso_radius(TORSO_MASS, 0.0))


def test_torso_radius_bridges_the_hips(body):
    """The trunk must be wide enough to actually meet the pelvis it sits on."""
    r = viz.torso_radius(TORSO_MASS, TORSO_LEN)
    assert r > HIP_HALF_SPAN


def test_draw_pelvis_band_and_torso_static(vis, body):
    viz.clear_scene(vis)
    r_band = viz.draw_pelvis_band(vis, body["legs"]["l"]["hip"],
                                  body["legs"]["r"]["hip"], body["l5s1"],
                                  frame_index=3)
    r_torso = viz.draw_torso(vis, body["l5s1"], body["acromion"], TORSO_MASS,
                             frame_index=3)
    assert r_band > 0 and r_torso > 0
    # re-posing without rebuilding is the animation-adjacent path
    for i in range(0, N_FRAMES, 5):
        viz.draw_pelvis_band(vis, body["legs"]["l"]["hip"],
                             body["legs"]["r"]["hip"], body["l5s1"],
                             frame_index=i, radius=r_band, build=False)
        viz.draw_torso(vis, body["l5s1"], body["acromion"], TORSO_MASS,
                       frame_index=i, radius=r_torso, build=False)
    commands = _html_commands(viz.render_static_html(vis))
    for path in (b"boneid/pelvis/band", b"boneid/pelvis/sacrum",
                 b"boneid/pelvis/cap_l", b"boneid/pelvis/cap_r",
                 b"boneid/torso/trunk"):
        assert path in commands, f"{path!r} never reached the scene"


def test_draw_pelvis_band_accepts_static_points(vis):
    r = viz.draw_pelvis_band(vis, np.array([0.0, 0.09, 0.9]),
                             np.array([0.0, -0.09, 0.9]))
    assert np.isclose(r, viz.PELVIS_BAND_FRACTION * HIP_HALF_SPAN)


def test_pelvis_span_tuple_forms():
    a, b, c = np.zeros(3), np.ones(3), 2 * np.ones(3)
    assert viz._pelvis_span_parts((a, b))[2] is None
    assert np.allclose(viz._pelvis_span_parts((a, b, c))[2], c)
    with pytest.raises(ValueError):
        viz._pelvis_span_parts((a,))
    with pytest.raises(ValueError):
        viz._torso_parts((a, b))


def test_draw_skeleton_with_pelvis_and_torso(vis, body):
    """The composed scene: one leg drawn with the body pieces attached."""
    viz.clear_scene(vis)
    leg = body["legs"]["r"]
    viz.draw_skeleton(
        vis, leg["skeleton"], leg["kin"], 0,
        pelvis_span=(body["legs"]["l"]["hip"], body["legs"]["r"]["hip"],
                     body["l5s1"]),
        torso=(body["l5s1"], body["acromion"], TORSO_MASS))
    commands = _html_commands(viz.render_static_html(vis))
    assert b"boneid/skeleton/thigh" in commands
    assert b"boneid/pelvis/band" in commands
    assert b"boneid/torso/trunk" in commands


def test_animate_pelvis_and_torso_have_tracks(vis, body):
    viz.clear_scene(vis)
    leg = body["legs"]["r"]
    animation = viz.animate(
        vis, leg["skeleton"], leg["kin"],
        pelvis_span=(body["legs"]["l"]["hip"], body["legs"]["r"]["hip"],
                     body["l5s1"]),
        torso=(body["l5s1"], body["acromion"], TORSO_MASS))
    paths = {p.lower() for p in animation.clips}
    # meshcat prefixes every scene path with "/meshcat"
    for expected in ("/meshcat/boneid/pelvis/band",
                     "/meshcat/boneid/pelvis/sacrum",
                     "/meshcat/boneid/pelvis/cap_l",
                     "/meshcat/boneid/pelvis/cap_r",
                     "/meshcat/boneid/torso/trunk"):
        assert expected in paths, f"{expected} was never animated"
    # every animated path must carry a full set of keyframes
    lowered = animation.lower()
    assert all(len(entry["clip"]["tracks"][0]["keys"]) == N_FRAMES
               for entry in lowered)
    html = viz.render_static_html(vis)
    assert len(html) > 100_000


def test_two_legs_share_one_pelvis_in_one_scene(vis, body):
    """Both legs, two roots, one appended animation — the report's layout."""
    viz.clear_scene(vis)
    span = (body["legs"]["l"]["hip"], body["legs"]["r"]["hip"], body["l5s1"])
    animation = viz.animate(vis, body["legs"]["r"]["skeleton"],
                            body["legs"]["r"]["kin"], root="boneid/r",
                            pelvis_span=span,
                            torso=(body["l5s1"], body["acromion"], TORSO_MASS))
    animation = viz.animate(vis, body["legs"]["l"]["skeleton"],
                            body["legs"]["l"]["kin"], root="boneid/l",
                            animation=animation, camera=False)
    paths = {p.lower() for p in animation.clips}
    assert any(p.startswith("/meshcat/boneid/r/") for p in paths)
    assert any(p.startswith("/meshcat/boneid/l/") for p in paths)
    assert "/meshcat/boneid/r/torso/trunk" in paths
    html = viz.render_static_html(vis)
    assert len(html) > 100_000


# ---------------------------------------------------------------------------
# Looping (CLAUDE.md rule: animations loop)
# ---------------------------------------------------------------------------

def test_animate_loops_by_default(vis, skeleton, kin, monkeypatch):
    seen = {}

    def spy(self, animation, play=True, repetitions=1):
        seen["repetitions"] = repetitions

    # Visualizer uses __slots__, so patch the class, not the instance
    monkeypatch.setattr(type(vis), "set_animation", spy)
    viz.animate(vis, skeleton, kin)
    assert seen["repetitions"] == viz.LOOP_REPETITIONS
    assert viz.LOOP_REPETITIONS >= 1000

    viz.animate(vis, skeleton, kin, repetitions=7)
    assert seen["repetitions"] == 7


# ---------------------------------------------------------------------------
# URDF skeleton (vendored human-gazebo humanSubject01, LGPL-2.1)
# ---------------------------------------------------------------------------

LEG_LINKS = {"RightFoot": 0, "RightLowerLeg": 1, "RightUpperLeg": 2}


def test_vendored_urdf_is_present_and_licensed():
    assert URDF_PATH.exists(), "the vendored URDF is missing"
    note = URDF_PATH.parent / "LICENSE-note.md"
    assert note.exists(), "vendored assets must state their source and license"
    text = note.read_text()
    assert "human-gazebo" in text and "LGPL-2.1" in text
    assert (URDF_PATH.parent / "UPSTREAM-LICENSE.txt").exists()


def test_parse_urdf_finds_primitive_links_only():
    links = viz.parse_urdf(URDF_PATH)
    # the anatomical links we would ever map onto our chain
    for name in ("Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot",
                 "LeftUpperLeg", "Head"):
        assert name in links
    # the massless *_f1/*_f2 helper links carry no <visual> and must be absent
    assert not [n for n in links if n.endswith(("_f1", "_f2"))]
    kinds = {v["kind"] for vs in links.values() for v in vs}
    assert kinds <= {"box", "cylinder", "sphere"}
    assert kinds == {"box", "cylinder", "sphere"}
    for visual in links["Pelvis"]:
        assert visual["origin"].shape == (4, 4)
        assert np.all(np.isfinite(visual["origin"]))
        assert visual["extent_z"] > 0


def test_parse_urdf_rejects_nothing_but_reports_empty(tmp_path):
    """A mesh-only URDF parses to {} rather than to invented geometry."""
    mesh_only = tmp_path / "meshy.urdf"
    mesh_only.write_text(
        '<robot name="m"><link name="A"><visual><geometry>'
        '<mesh filename="package://x/a.stl"/></geometry></visual></link></robot>')
    assert viz.parse_urdf(mesh_only) == {}
    with pytest.raises(ValueError):
        viz.draw_urdf_skeleton(None, mesh_only, {"A": 0}, None, 0)


def test_draw_urdf_skeleton_posed_at_frame_zero(vis, body):
    viz.clear_scene(vis)
    leg = body["legs"]["r"]
    scales = viz.draw_urdf_skeleton(vis, URDF_PATH, LEG_LINKS, leg["kin"], 0,
                                    skeleton=leg["skeleton"],
                                    color=viz.SEGMENT_COLOR)
    assert set(scales) == set(LEG_LINKS)
    # scaling from Skeleton.length puts a template model in our subject's size
    assert all(0.2 < s < 5.0 for s in scales.values())
    commands = _html_commands(viz.render_static_html(vis))
    for link in LEG_LINKS:
        assert f"boneid/urdf/{link}/v0".encode() in commands
    # ...and the geometry types the URDF actually declares came across
    assert b"BoxGeometry" in commands and b"CylinderGeometry" in commands


def test_urdf_auto_scale_matches_our_segment_length(vis, body):
    """For a +Z-grown link (a limb cylinder) the auto-scale is exact."""
    leg = body["legs"]["r"]
    links = viz.parse_urdf(URDF_PATH)
    urdf_shank = links["RightLowerLeg"][0]["params"]["length"]
    scales = viz.draw_urdf_skeleton(vis, URDF_PATH, {"RightLowerLeg": 1},
                                    leg["kin"], 0, skeleton=leg["skeleton"])
    assert np.isclose(scales["RightLowerLeg"] * urdf_shank, SHANK_LEN)


def test_draw_urdf_skeleton_explicit_scale_wins(vis, body):
    leg = body["legs"]["r"]
    scales = viz.draw_urdf_skeleton(vis, URDF_PATH, LEG_LINKS, leg["kin"], 0,
                                    scale_map={"RightFoot": 2.0},
                                    skeleton=leg["skeleton"])
    assert scales["RightFoot"] == 2.0
    # no skeleton and no scale_map -> the URDF's own size
    plain = viz.draw_urdf_skeleton(vis, URDF_PATH, LEG_LINKS, leg["kin"], 0)
    assert set(plain.values()) == {1.0}


def test_draw_urdf_skeleton_unknown_link_is_an_error(vis, body):
    leg = body["legs"]["r"]
    with pytest.raises(ValueError, match="RightFlipper"):
        viz.draw_urdf_skeleton(vis, URDF_PATH, {"RightFlipper": 0},
                               leg["kin"], 0)
