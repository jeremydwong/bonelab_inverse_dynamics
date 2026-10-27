"""Broader coverage for boneid.core utilities. The intelligible, physics-level
checks live in test_key.py; these pin down the individual tools."""

import numpy as np

from boneid.core import (angular_velocity, cop_from_wrench, detect_contact,
                         finite_difference, fit_rigid_transform,
                         inertia_world, lowpass, wrench_from_cop)
from boneid.simulate import minjerk, rot_y


def test_finite_difference_polynomial():
    rate = 100.0
    t = np.arange(200) / rate
    x = 3 * t**2 + 2 * t + 1
    d = finite_difference(x, rate)
    np.testing.assert_allclose(d[1:-1], 6 * t[1:-1] + 2, atol=1e-9)


def test_finite_difference_multicolumn():
    rate = 50.0
    t = np.arange(100) / rate
    x = np.stack([t, t**2], axis=1)
    d = finite_difference(x, rate)
    np.testing.assert_allclose(d[1:-1, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(d[1:-1, 1], 2 * t[1:-1], atol=1e-9)


def test_lowpass_preserves_dc_and_kills_noise():
    rate = 1000.0
    t = np.arange(2000) / rate
    clean = np.sin(2 * np.pi * 1.0 * t)
    noisy = clean + 0.3 * np.sin(2 * np.pi * 200.0 * t)
    filt = lowpass(noisy, rate, 10.0)
    assert np.abs(filt - clean)[100:-100].max() < 0.02
    # cutoff <= 0 is a no-op
    np.testing.assert_array_equal(lowpass(noisy, rate, 0.0), noisy)


def test_fit_rigid_transform_recovers_pose():
    rng = np.random.default_rng(0)
    neutral = rng.normal(size=(5, 3))
    angles = np.linspace(0, 1.0, 40)
    r_true = rot_y(angles)
    d_true = np.stack([angles, 2 * angles, np.ones_like(angles)], axis=1)
    moving = np.einsum("tij,mj->tmi", r_true, neutral) + d_true[:, None]
    r, d, res = fit_rigid_transform(neutral, moving)
    np.testing.assert_allclose(r, r_true, atol=1e-10)
    np.testing.assert_allclose(d, d_true, atol=1e-10)
    assert res.max() < 1e-10


def test_fit_rigid_transform_residual_reports_nonrigidity():
    rng = np.random.default_rng(1)
    neutral = rng.normal(size=(4, 3))
    moving = np.repeat(neutral[None], 3, axis=0)
    moving[1, 0] += 0.05  # perturb one marker in one frame
    _, _, res = fit_rigid_transform(neutral, moving)
    assert res[1] > 5 * max(res[0], 1e-12)


def test_angular_velocity_constant_spin():
    rate = 1000.0
    t = np.arange(500) / rate
    w_true = 4.0
    r = rot_y(w_true * t)
    w = angular_velocity(r, rate)
    np.testing.assert_allclose(w[2:-2, 1], w_true, atol=1e-4)
    np.testing.assert_allclose(w[2:-2, [0, 2]], 0.0, atol=1e-6)


def test_inertia_world_is_symmetric_similarity():
    i_local = np.diag([0.4, 0.2, 0.1])
    r = rot_y(np.array([0.3, 1.1]))
    i_w = inertia_world(r, i_local)
    # symmetric, trace-preserving, positive-definite — all destroyed by the
    # legacy one-sided product R @ I
    np.testing.assert_allclose(i_w, np.swapaxes(i_w, -1, -2), atol=1e-12)
    np.testing.assert_allclose(np.trace(i_w, axis1=-2, axis2=-1), 0.7,
                               atol=1e-12)
    assert (np.linalg.eigvalsh(i_w) > 0).all()
    one_sided = np.einsum("tij,jk->tik", r, i_local)
    assert not np.allclose(one_sided, i_w)


def test_cop_round_trip():
    rng = np.random.default_rng(2)
    t_n = 50
    force = rng.normal(size=(t_n, 3)) * 50
    force[:, 2] = 600 + 100 * rng.random(t_n)  # solid vertical load
    cop_true = rng.normal(size=(t_n, 3)) * 0.1
    cop_true[:, 2] = 0.0
    free_true = np.zeros((t_n, 3))
    free_true[:, 2] = rng.normal(size=t_n) * 5
    point = np.array([0.4, -0.3, 0.0])
    moment = wrench_from_cop(force, cop_true, free_true, point)
    cop, free = cop_from_wrench(force, moment, point, plane_height=0.0)
    np.testing.assert_allclose(cop, cop_true, atol=1e-10)
    np.testing.assert_allclose(free, free_true, atol=1e-8)


def test_cop_low_force_is_nan_not_garbage():
    force = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 800.0]])
    moment = np.array([[5.0, 5.0, 0.0], [10.0, -10.0, 0.0]])
    cop, _ = cop_from_wrench(force, moment, np.zeros(3), min_fz=10.0)
    assert np.isnan(cop[0, :2]).all()
    assert np.isfinite(cop[1]).all()


def test_detect_contact_events_and_debounce():
    fz = np.zeros(100)
    fz[10:40] = 700.0
    fz[25] = 0.0          # one-sample dropout inside stance
    fz[60:80] = 650.0
    mask, events = detect_contact(fz, threshold=20.0, min_gap=5)
    assert len(events) == 2
    assert events[0] == (10, 39)
    assert events[1] == (60, 79)
    assert mask[25]       # dropout bridged


def test_minjerk_endpoints():
    u = np.linspace(0, 1, 101)
    s = minjerk(u)
    assert s[0] == 0.0 and s[-1] == 1.0
    ds = np.gradient(s, u)
    assert abs(ds[0]) < 1e-3 and abs(ds[-1]) < 1e-3


def test_two_leg_matches_serial_and_conserves_energy():
    """Branched two-leg + shared-pelvis model: leg joints must reproduce the
    serial recursion exactly, the energy audit must close, and the mean
    vertical pelvis residual must equal the unmodelled weight above it."""
    import os
    import pytest as _pytest

    path = ("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
            "p1_5StridesData.mat")
    if not os.path.exists(path):
        _pytest.skip("local validation data not available")
    from boneid.core import (AnalysisParams, inverse_dynamics,
                             inverse_dynamics_two_legs, energy_audit_two_legs)
    from boneid.io_v3d import estimate_body_mass, load_v3d_trial, build_chain

    trial = load_v3d_trial(path, 13)
    params = AnalysisParams(lowpass_hz=12.0, force_lowpass_hz=0.0)
    skr, kinr, grr = build_chain(trial, side="r")
    skl, kinl, grl = build_chain(trial, side="l")
    two = inverse_dynamics_two_legs(skr, kinr, grr, skl, kinl, grl, params)
    serial = inverse_dynamics(skr, kinr, grr, params)
    np.testing.assert_array_equal(two.right.joint_torque[:, :2],
                                  serial.joint_torque[:, :2])

    audit = energy_audit_two_legs(skr, kinr, grr, skl, kinl, grl, two, params)
    i = slice(10, -10)
    assert (np.abs(audit.imbalance[i]).max()
            < 0.03 * np.abs(audit.de_dt).max())

    body_mass = estimate_body_mass(trial)
    unmodelled = body_mass - skr.mass.sum() - skl.mass[:3].sum()
    mean_fz = two.residual_force[i, 2].mean()
    assert abs(mean_fz + unmodelled * 9.81) < 0.05 * body_mass * 9.81


# ---------------------------------------------------------------------------
# io_c3d: static calibration carried as virtual points (synthetic)
# ---------------------------------------------------------------------------

def _synthetic_calibration():
    """A StaticCalibration with one triangle cluster and one virtual point."""
    from boneid.io_c3d import StaticCalibration

    neutral = np.array([[0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.12, 0.0]])
    point = np.array([0.05, 0.04, -0.3])          # off-cluster joint centre
    r0 = rot_y(np.array([0.3]))[0]                # static anatomical frame
    return StaticCalibration(
        path="synthetic", side="r", sex="male", body_mass=70.0,
        cluster_neutral={"foot": neutral}, points={"ankle": point},
        r_static=np.stack([r0] * 4), lengths={}, foot_heel_local=np.zeros(3))


def test_virtual_point_reconstruction_is_exact_for_rigid_motion():
    """A calibrated point carried by the cluster fit must reproduce the
    rigidly-moved point exactly, and the fit residual must be ~zero."""
    from boneid.core import fit_rigid_transform
    from boneid.io_c3d import virtual_points

    cal = _synthetic_calibration()
    neutral = cal.cluster_neutral["foot"]
    angles = np.linspace(0.0, 1.2, 40)
    r_true = rot_y(angles)                                   # [T,3,3]
    d_true = np.stack([0.3 * angles, 0.1 * angles, 0 * angles], axis=1)
    moving = np.einsum("tij,mj->tmi", r_true, neutral) + d_true[:, None]

    poses = {"foot": fit_rigid_transform(neutral, moving)}
    pts = virtual_points(poses, cal)
    expected = np.einsum("tij,j->ti", r_true, cal.points["ankle"]) + d_true
    np.testing.assert_allclose(pts["ankle"], expected, atol=1e-10)
    np.testing.assert_allclose(poses["foot"][2], 0.0, atol=1e-10)
    # and the carried frame is r_true @ r_static
    r_carried = np.einsum("tij,jk->tik", poses["foot"][0], cal.r_static[0])
    np.testing.assert_allclose(
        r_carried, np.einsum("tij,jk->tik", r_true, cal.r_static[0]),
        atol=1e-10)


def test_despike_removes_label_swap_but_keeps_real_motion():
    from boneid.io_c3d import _despike, _fill_gaps

    rate = 300.0
    t = np.arange(300) / rate
    x = np.stack([0.5 * np.sin(2 * np.pi * 1.5 * t),      # ~4.7 m/s peak: real
                  np.zeros_like(t), np.zeros_like(t)], axis=1)
    swapped = x.copy()
    swapped[100:130, 0] += 0.1                            # 30 m/s jumps in+out
    cleaned = _fill_gaps(_despike(swapped.copy(), rate))
    # the 0.1 m excursion is gone; what's left is the linear-bridge chord
    # error over the 0.1 s span of a 1.5 Hz sine (~0.03 m), not the swap
    assert np.abs(cleaned - x).max() < 0.03
    untouched = _fill_gaps(_despike(x.copy(), rate))
    np.testing.assert_array_equal(untouched, x)           # real motion kept


def test_unswap_labels_restores_permanent_swap():
    """Two markers that permanently exchange labels mid-trial are unswapped
    by trajectory continuity; a lone real fast motion is left alone."""
    from boneid.io_c3d import _unswap_labels

    rate = 300.0
    t = np.arange(600) / rate
    truth_a = np.stack([0.3 * np.sin(2 * np.pi * t), np.zeros_like(t),
                        0.6 + 0.02 * np.cos(2 * np.pi * t)], axis=1)
    truth_b = truth_a + np.array([0.02, 0.01, -0.09])     # cluster-mate below
    rec_a, rec_b = truth_a.copy(), truth_b.copy()
    rec_a[300:], rec_b[300:] = truth_b[300:], truth_a[300:]   # permanent swap

    markers = {"A": rec_a, "B": rec_b,
               "C": truth_a + np.array([0.5, 0.5, 0.0])}      # bystander
    swaps = _unswap_labels(markers, rate)
    assert [(s[1], s[2]) for s in swaps] in ([("A", "B")], [("B", "A")])
    np.testing.assert_allclose(markers["A"], truth_a, atol=1e-12)
    np.testing.assert_allclose(markers["B"], truth_b, atol=1e-12)
    np.testing.assert_allclose(markers["C"], truth_a + [0.5, 0.5, 0.0])


def test_quiet_standing_determination():
    """The no-full-stride fallback is only legitimate for quiet standing:
    sway-level ripple passes, gait-level fluctuation and unloaded plates
    do not."""
    from boneid.io_v3d import is_quiet_standing

    t = np.linspace(0, 5, 600)
    bw = 830.0
    sway = bw + 0.02 * bw * np.sin(2 * np.pi * 0.3 * t)      # 2% sway
    gait = bw + 0.30 * bw * np.sin(2 * np.pi * 0.9 * t)      # stride ripple
    assert is_quiet_standing(sway)
    assert not is_quiet_standing(gait)
    assert not is_quiet_standing(np.full_like(t, 3.0))       # unloaded plate


def test_body_mass_estimate_and_override_on_real_trial():
    """On real data: <2 contacts + gait fluctuation raises (no silently
    biased mass), and build_chain(body_mass=...) keeps the marker-measured
    skeleton while overriding only the mass."""
    import os
    import pytest as _pytest
    from dataclasses import replace

    path = ("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
            "p1_5StridesData.mat")
    if not os.path.exists(path):
        _pytest.skip("local validation data not available")
    from boneid.io_v3d import build_chain, estimate_body_mass, load_v3d_trial

    trial = load_v3d_trial(path, 13)
    # a partial-stride slice: too short for two right-foot contacts
    n_a = 900                                # 0.75 s of analog
    n_t = n_a // 10
    short = replace(trial, t=trial.t[:n_t], t_analog=trial.t_analog[:n_a],
                    markers={k: v[:n_t] for k, v in trial.markers.items()},
                    landmarks={k: v[:n_t] for k, v in trial.landmarks.items()},
                    force=trial.force[:n_a], cop=trial.cop[:n_a],
                    free_moment=trial.free_moment[:n_a])
    with _pytest.raises(ValueError, match="quiet standing"):
        estimate_body_mass(short)

    # the override: full auto-skeleton (marker lengths) at a supplied mass
    skel_a, _, _ = build_chain(trial, side="r")
    skel_b, _, _ = build_chain(trial, side="r", body_mass=70.0)
    est = estimate_body_mass(trial)
    np.testing.assert_allclose(skel_b.mass.sum() / skel_a.mass.sum(),
                               70.0 / est, rtol=1e-12)
    np.testing.assert_allclose(skel_b.length, skel_a.length)  # lengths kept


# ---------------------------------------------------------------------------
# Tutorial simulators (bricks, synthetic markers)
# ---------------------------------------------------------------------------

def test_brick_hand_wrench_is_the_residual():
    """One brick, zero ground wrench: the pipeline's residual is the hand
    wrench.  Corner markers reproduce the pose exactly; the residual matches
    the analytic wrench to finite-difference accuracy."""
    from boneid.core import (AnalysisParams, GroundWrench, energy_audit,
                             fit_rigid_transform, inverse_dynamics)
    from boneid.simulate import (BRICK_SIZE, segment_markers, simulate_brick_2d,
                                 tetrahedral_corners)

    skeleton, kin, markers, truth = simulate_brick_2d()
    corners = tetrahedral_corners(BRICK_SIZE)
    moving = np.stack([markers[f"c{k + 1}"] for k in range(4)], axis=1)
    r, d, res = fit_rigid_transform(corners, moving)
    np.testing.assert_allclose(r, kin.r_world[:, 0], atol=1e-12)
    np.testing.assert_allclose(d, kin.prox_pos[:, 0], atol=1e-12)
    assert res.max() < 1e-12

    raw = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)
    zero = GroundWrench(t=kin.t, rate=kin.rate, force=np.zeros((len(kin.t), 3)),
                        moment=np.zeros((len(kin.t), 3)))
    idres = inverse_dynamics(skeleton, kin, zero, raw)
    inner = slice(5, -5)
    f_scale = np.abs(truth["force"]).max()
    m_scale = np.abs(truth["torque"]).max()
    np.testing.assert_allclose(idres.residual_force[inner], truth["force"][inner],
                               atol=1e-4 * f_scale)
    np.testing.assert_allclose(idres.residual_torque[inner], truth["torque"][inner],
                               atol=1e-4 * m_scale)
    audit = energy_audit(skeleton, kin, zero, idres, raw)
    assert np.abs(audit.imbalance[inner]).max() < 1e-5 * np.abs(audit.de_dt).max()
    # the markers really ride the segment
    again = segment_markers(kin, {"c1": (0, corners[0])})["c1"]
    np.testing.assert_allclose(again, markers["c1"])


def test_two_bricks_top_down_equals_bottom_up():
    """Bottom-up through the exact ground wrench and top-down from the free
    upper brick give the same hinge wrench; the residual vanishes; a wrong
    lower-brick mass leaks into the residual (the alarm) and nowhere else."""
    from dataclasses import replace
    from boneid.core import AnalysisParams, GroundWrench, inverse_dynamics
    from boneid.simulate import simulate_two_bricks_2d

    skeleton, kin, ground, markers, hinge = simulate_two_bricks_2d()
    raw = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)
    up = inverse_dynamics(skeleton, kin, ground, raw)
    inner = slice(5, -5)
    bw = skeleton.mass.sum() * 9.81
    assert np.abs(up.residual_force[inner]).max() < 1e-6 * bw
    assert np.abs(up.residual_torque[inner]).max() < 1e-5 * np.abs(up.joint_torque).max()

    # top-down: the upper brick alone, turned upside down so that the hinge
    # is its PROXIMAL end (where the pipeline reports the residual)
    from boneid.core import SegmentKinematics, Skeleton
    L_b = skeleton.length[1]
    top_skel = Skeleton(segment_names=["upper"], joint_names=[],
                        mass=skeleton.mass[1:], com_local=np.array([[0, 0, 0.5 * L_b]]),
                        inertia_local=skeleton.inertia_local[1:], length=skeleton.length[1:])
    top_kin = SegmentKinematics(t=kin.t, rate=kin.rate, r_world=kin.r_world[:, 1:],
                                prox_pos=kin.dist_pos[:, 1:], dist_pos=kin.prox_pos[:, 1:])
    zero = GroundWrench(t=kin.t, rate=kin.rate, force=np.zeros((len(kin.t), 3)),
                        moment=np.zeros((len(kin.t), 3)))
    down = inverse_dynamics(top_skel, top_kin, zero, raw)
    # top-down: the wrench the hinge applies ON the upper brick is minus the
    # wrench on the lower brick reported by the bottom-up pass
    np.testing.assert_allclose(-down.residual_force[inner], up.joint_force[inner, 0],
                               atol=1e-5 * bw)
    np.testing.assert_allclose(-down.residual_torque[inner], up.joint_torque[inner, 0],
                               atol=1e-4 * np.abs(up.joint_torque).max())

    # the hinge as a virtual point agrees from both clusters
    np.testing.assert_allclose(
        kin.prox_pos[:, 0],
        kin.prox_pos[:, 1] + np.einsum("tij,j->ti", kin.r_world[:, 1], hinge["upper"]),
        atol=1e-12)

    heavy = replace(skeleton, mass=skeleton.mass * np.array([1.2, 1.0]))
    wrong = inverse_dynamics(heavy, kin, ground, raw)
    assert np.abs(wrong.residual_force[inner, 2]).max() > 0.5 * 0.2 * skeleton.mass[0] * 9.81


def test_s2s_markers_recover_frames_and_sway_exposes_legacy_physics():
    """Anatomical marker pairs rebuild the simulator's frames exactly; with
    torso sway the motion is 3-D and the legacy physics leaks a residual
    torque where the correct physics stays at round-off."""
    from boneid.core import AnalysisParams, LegacyPhysics, inverse_dynamics
    from boneid.io_v3d import orthonormal_frame
    from boneid.simulate import (s2s_marker_layout, s2s_skeleton, segment_markers,
                                 simulate_squat_to_stand)

    skeleton, kin, ground, truth = simulate_squat_to_stand(torso_sway=0.25)
    m = segment_markers(kin, s2s_marker_layout())
    hip = 0.5 * (m["hip_lat"] + m["hip_med"])
    top = 0.5 * (m["shoulder_l"] + m["shoulder_r"])
    r_torso = orthonormal_frame(top - hip, m["shoulder_l"] - m["shoulder_r"])
    np.testing.assert_allclose(r_torso, kin.r_world[:, 3], atol=1e-12)
    np.testing.assert_allclose(hip, kin.prox_pos[:, 2], atol=1e-12)
    lengths = np.linalg.norm(kin.dist_pos[-1] - kin.prox_pos[-1], axis=1)
    rebuilt = s2s_skeleton(80.0, lengths)
    np.testing.assert_allclose(rebuilt.inertia_local, skeleton.inertia_local)
    np.testing.assert_allclose(rebuilt.com_local, skeleton.com_local)

    raw = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)
    inner = slice(5, -5)
    good = inverse_dynamics(skeleton, kin, ground, raw)
    tq = np.abs(good.joint_torque).max()
    assert np.abs(good.residual_torque[inner]).max() < 1e-5 * tq
    for phys in (LegacyPhysics(one_sided_inertia=True),
                 LegacyPhysics(gyroscopic=False)):
        bad = inverse_dynamics(skeleton, kin, ground, raw, physics=phys)
        assert np.abs(bad.residual_torque[inner]).max() > 1e-3 * tq
    # planar motion forgives both errors exactly
    skeleton, kin, ground, _ = simulate_squat_to_stand()
    flat = inverse_dynamics(skeleton, kin, ground, raw,
                            physics=LegacyPhysics(one_sided_inertia=True, gyroscopic=False))
    assert np.abs(flat.residual_torque[inner]).max() < 1e-5 * tq


def test_s2s_c3d_knee_slip_6dof_power(tmp_path):
    """A knee that slides does translational work: the real c3d pipeline
    must recover it, and the energy audit must need it to close."""
    from boneid.core import AnalysisParams, energy_audit, inverse_dynamics
    from boneid.io_c3d import build_chain, calibrate_static, load_c3d
    from boneid.simulate import simulate_squat_to_stand_c3d

    truth = simulate_squat_to_stand_c3d(tmp_path, knee_slip_m=0.01)
    cal = calibrate_static(load_c3d(truth["static_path"]),
                           body_mass=truth["body_mass"])
    params = AnalysisParams(contact_threshold_n=50.0, lowpass_hz=12.0,
                            force_lowpass_hz=12.0)
    sk, kin, ground, _ = build_chain(load_c3d(truth["dynamic_path"]), cal,
                                     point=np.zeros(3), threshold_n=50.0)
    res = inverse_dynamics(sk, kin, ground, params)
    audit = energy_audit(sk, kin, ground, res, params)
    inner = slice(15, -15)
    trans_true = truth["joint_power_translational"][inner]
    assert np.abs(trans_true[:, 1]).max() > 5.0             # the knee slides
    np.testing.assert_allclose(audit.joint_power_translational[inner],
                               trans_true, atol=0.05)
    peak = np.abs(audit.de_dt).max()
    assert np.abs(audit.imbalance[inner]).max() < 1e-3 * peak
    three_dof = audit.imbalance + audit.joint_power_translational.sum(axis=1)
    assert np.abs(three_dof[inner]).max() > 5.0             # 3-DoF misses it


def test_varus_shift_moves_only_the_knee():
    """The varus counterfactual: knee moved d laterally, square to the
    hip-ankle line; hip, ankle, foot untouched; shank/thigh frames stay
    orthonormal (their lengths stretch a little, see `varus_rotations`)."""
    from boneid import io_v3d as io
    from boneid.simulate import varus_markers, varus_shift
    from pathlib import Path
    trial = io.load_v3d_trial(str(Path(__file__).resolve().parents[1]
                                  / "data/p1_trial13_sample.mat"), 0)
    _, kin, _ = io.build_chain(trial, side="r")
    d = 0.03
    k2 = varus_shift(kin, d)
    move = k2.prox_pos[:, 1] - kin.prox_pos[:, 1]
    np.testing.assert_allclose(np.linalg.norm(move, axis=1), d, atol=1e-12)
    axis = kin.prox_pos[:, 2] - kin.prox_pos[:, 0]
    assert np.max(np.abs(np.sum(move * axis, axis=1))) < 1e-12
    lateral = -kin.r_world[:, 2, :, 1]                  # right leg: -left
    assert np.all(np.sum(move * lateral, axis=1) > 0.9 * d)
    for s in (0, 2):                                    # ankle, hip fixed
        np.testing.assert_allclose(k2.prox_pos[:, s], kin.prox_pos[:, s])
    np.testing.assert_allclose(k2.r_world[:, 0], kin.r_world[:, 0])
    for s, (a, b) in ((1, (0, 1)), (2, (1, 2))):
        # the knee moves square to hip-ankle, not on a sphere: a small stretch
        stretch = (np.linalg.norm(k2.prox_pos[:, b] - k2.prox_pos[:, a], axis=1)
                   - np.linalg.norm(kin.prox_pos[:, b] - kin.prox_pos[:, a],
                                    axis=1))
        assert np.max(np.abs(stretch)) < 0.01
        r = np.einsum("tji,tjk->tik", k2.r_world[:, s], k2.r_world[:, s])
        np.testing.assert_allclose(r, np.broadcast_to(np.eye(3), r.shape),
                                   atol=1e-12)
    mk = varus_markers(trial.markers, kin, d)
    assert "RSH1" in mk and "RTH1" in mk and "RCAL" not in mk
    # a thigh marker keeps its distance to the hip
    np.testing.assert_allclose(
        np.linalg.norm(mk["RTH1"] - kin.prox_pos[:, 2], axis=1),
        np.linalg.norm(trial.markers["RTH1"] - kin.prox_pos[:, 2], axis=1),
        atol=1e-12)
    np.testing.assert_allclose(varus_shift(kin, 0.0).r_world, kin.r_world,
                               atol=1e-12)


def test_step_frequency_trial_keeps_speed_and_weight():
    """The cadence counterfactual: k=1 is the identity; for k != 1 the
    trial is k times shorter, a heel's fore-aft sweep is 1/k (same belt
    speed), and the force averaged over the trial still carries body
    weight."""
    from pathlib import Path
    from boneid import io_v3d as io
    from boneid.simulate import step_frequency_trial
    trial = io.load_v3d_trial(str(Path(__file__).resolve().parents[1]
                                  / "data/p1_trial13_sample.mat"), 0)
    m = 85.16
    same = step_frequency_trial(trial, 1.0, m)
    np.testing.assert_allclose(same.force, trial.force, atol=1e-9)
    np.testing.assert_allclose(same.markers["RCAL"], trial.markers["RCAL"],
                               atol=1e-12)
    for k in (0.9, 1.1):
        s = step_frequency_trial(trial, k, m)
        assert abs(s.t[-1] - trial.t[-1] / k) < 1.0 / trial.rate
        np.testing.assert_allclose(np.ptp(s.markers["RCAL"][:, 1]),
                                   np.ptp(trial.markers["RCAL"][:, 1]) / k,
                                   rtol=0.02)
        np.testing.assert_allclose(np.ptp(s.markers["RCAL"][:, 2]),
                                   np.ptp(trial.markers["RCAL"][:, 2]),
                                   rtol=0.02)
        fz0 = trial.force[..., 2].sum(axis=1).mean()
        fz1 = s.force[..., 2].sum(axis=1).mean()
        assert abs(fz1 - fz0) < 0.01 * fz0
