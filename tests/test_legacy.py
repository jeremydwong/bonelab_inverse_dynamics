"""boneid.legacy — the ported MATLAB routines against their text and data."""
import numpy as np
import pytest

from boneid import legacy as L

DATA = "data/asics"


def test_soder_matches_fit_rigid_transform_and_v2_transposes():
    from boneid.core import fit_rigid_transform
    rng = np.random.default_rng(1)
    neut = rng.normal(size=(4, 3))
    ang = np.linspace(0, 1.2, 50)
    r_true = np.stack([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in ang])
    d_true = rng.normal(size=(50, 3))
    mov = np.einsum("tij,mj->tmi", r_true, neut) + d_true[:, None]
    r, d, res = L.soder(neut, mov)
    r2, d2, _ = fit_rigid_transform(neut, mov)
    np.testing.assert_allclose(r, r2, atol=1e-12)
    np.testing.assert_allclose(d, d2, atol=1e-12)
    assert res.max() < 1e-12
    p, tm, _ = L.virtual_point_v2(neut, neut[0], mov)
    np.testing.assert_allclose(p, mov[:, 0], atol=1e-12)
    np.testing.assert_allclose(tm, np.swapaxes(r_true, 1, 2), atol=1e-12)


def test_new_cs_is_the_simm_frame_and_g2l_is_tm_times_global():
    lab = np.array([[1.0, 2.0, 3.0]])           # +x left, +y posterior, +z up
    simm = L.simm_from_lab(lab)[0]
    assert np.allclose(simm, [-2.0, 3.0, -1.0])  # x anterior, y up, z right
    np.testing.assert_allclose(L.lab_from_simm(L.simm_from_lab(lab)), lab)
    tm = np.swapaxes(np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]]), 1, 2)
    np.testing.assert_allclose(L.g2l(tm, np.array([[1.0, 0, 0]]))[0], tm[0] @ [1, 0, 0])


def test_room_seg_ang_planar_closed_form():
    """RoomSegAng on TM = Rᵀ for a pure rotation θ(t) about one axis returns
    −θ̇ cos 2θ, not θ̇: right size but WRONG SIGN at the static pose, zero at
    45°, the right sign (wrong size) beyond.  Planar motion does NOT forgive
    this one.  `true_seg_ang` gives θ̇."""
    sf = 200.0
    t = np.arange(400) / sf
    th = 2.0 * t
    r = np.stack([[[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]] for a in th])
    tm = np.swapaxes(r, 1, 2)
    av_leg, _ = L.room_seg_ang(tm, sf)
    av_true, _ = L.true_seg_ang(tm, sf)
    np.testing.assert_allclose(av_leg[5:-5, 1], -2.0 * np.cos(2 * th[5:-5]), atol=2e-3)
    np.testing.assert_allclose(av_true[5:-5, 1], 2.0, rtol=1e-3)
    ph = 1.5 * t
    rx = np.stack([[[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]] for a in ph])
    tm2 = np.swapaxes(np.einsum("tij,tjk->tik", r, rx), 1, 2)
    av_leg2, _ = L.room_seg_ang(tm2, sf)
    av_true2, _ = L.true_seg_ang(tm2, sf)
    assert np.abs(av_leg2[5:-5] - av_true2[5:-5]).max() > 0.3


def test_filterdata1_skips_leading_and_trailing_zeros():
    x = np.zeros((300, 1))
    x[100:200, 0] = 1.0 + 0.1 * np.sin(np.arange(100))
    y = L.filterdata1(x, 300.0, 12.0)
    assert np.all(y[:100] == 0) and np.all(y[200:] == 0) and np.abs(y[150, 0] - 1.0) < 0.05


@pytest.fixture(scope="module")
def s1_am_rp():
    from boneid.io_c3d import load_c3d
    return load_c3d(f"{DATA}/S1_AM_Static.c3d"), load_c3d(f"{DATA}/S1_AM_RP.c3d")


def test_cop_algebra_matches_ezc3d_without_the_filter(s1_am_rp):
    import ezc3d
    _, trial = s1_am_rp
    analog = L.raw_analog(trial)
    rd, md, cop, plate = L.get_vicon_forces_motek(
        analog, trial.analog_rate, np.stack([L.OFFSET_FP1, L.OFFSET_FP2]), lowpass_hz=0.0)
    c = ezc3d.c3d(trial.path, extract_forceplat_data=True)
    p = c["data"]["platform"][plate - 1]
    cop_e = np.asarray(p["center_of_pressure"]).T / 1000.0
    fz = np.asarray(p["force"])[2]
    on = fz > 300
    d = np.abs(L.lab_from_simm(cop)[on, :2] - cop_e[on, :2])
    assert plate == 2                              # both feet on the second plate
    assert d.max() < 1e-3                          # under a millimetre
    np.testing.assert_allclose(rd[on, 1], fz[on], rtol=1e-9)


def test_legacy_pipeline_reproduces_running_moments(s1_am_rp):
    """The port runs on S1 AM RP and its per-stance exported moments sit
    close to the corrected pipeline's; the physics switches move them by
    a few N m only, in the direction of ours."""
    from boneid.core import AnalysisParams, inverse_dynamics
    from boneid.io_c3d import build_chain, calibrate_static
    static, trial = s1_am_rp
    res = L.run_legacy_pipeline(static, trial, "AM", "RP")
    assert 35 <= len(res.hs) <= 50 and len(res.steps) == 18
    assert all(res.to > res.hs)
    assert res.steps[7]["bad"] and not res.steps[0]["bad"]
    assert res.mass["leg"] < res.mass["thigh"] and res.mass["foot"] < 1.0
    # the MATLAB's exported peaks: plantarflexor ankle (negative local z),
    # extensor knee (positive), hip (negative) — magnitudes in N m
    pk = {j: np.mean([s["peak_local_moment"][j][2] for s in res.steps if not s["bad"]])
          for j in ("ankle", "knee", "hip")}
    assert -150 < pk["ankle"] < -90 and 50 < pk["knee"] < 110 and -90 < pk["hip"] < -30

    cal = calibrate_static(static)
    skel, kin, ground, _ = build_chain(trial, cal)
    ours = inverse_dynamics(skel, kin, ground,
                            AnalysisParams(contact_threshold_n=50, lowpass_hz=12, force_lowpass_hz=12))
    mask = np.zeros(len(trial.t), bool)
    for s, e in zip(res.hs, res.to):
        mask[s:e + 1] = True
    leg = L.legacy_stance_torque_lab(res)
    for j in range(3):
        a, b = leg[mask, j, 0], ours.joint_torque[mask, j, 0]
        assert np.corrcoef(a, b)[0, 1] > 0.9
        assert np.sqrt(np.mean((a - b) ** 2)) < 0.15 * np.abs(b).max()
    fixed = L.run_legacy_pipeline(static, trial, "AM", "RP", physics="corrected", free_moment_axis="y")
    leg2 = L.legacy_stance_torque_lab(fixed)
    rms = lambda x, j: np.sqrt(np.mean((x[mask, j, 0] - ours.joint_torque[mask, j, 0]) ** 2))
    assert rms(leg2, 2) < rms(leg, 2)              # hip improves with the physics


def test_port_matches_the_executed_matlab(s1_am_rp):
    """MATLAB R2024b ran the ORIGINAL routines (harness/run_legacy_trial.m)
    on S1 AM RP; the committed reference holds its outputs.  The port must
    reproduce them to round-off: same events, same recursion, same
    per-stance exported curves."""
    import os
    from boneid.io_c3d import load_c3d
    ref = L.load_matlab_reference(f"{DATA}/matlab_reference/S1_AM_RP_legacy.mat")
    static = load_c3d(f"{DATA}/S1_AM_Static.c3d", repair=False)
    trial = load_c3d(f"{DATA}/S1_AM_RP.c3d", repair=False)
    res = L.run_legacy_pipeline(static, trial, "AM", "RP",
                                anthro_overrides={1: float(ref["A"][0])})
    np.testing.assert_array_equal(ref["HS"].astype(int) - 1, res.hs)
    np.testing.assert_array_equal(ref["TO"].astype(int) - 1, res.to)
    mask = np.zeros(len(res.t), bool)
    for s, e in zip(res.hs, res.to):
        mask[s:e + 1] = True
    for key, joint in (("GlobalMankle", "ankle"), ("GlobalMknee", "knee"), ("GlobalMhip", "hip")):
        np.testing.assert_allclose(res.global_moment[joint][mask], ref[key][mask], atol=1e-8)
    for key, joint in (("step_ankle_Mloc", "ankle"), ("step_knee_Mloc", "knee"),
                       ("step_hip_Mloc", "hip"), ("step_mtp_Mloc", "mtp")):
        ours = np.stack([s["local_moment"][joint] for s in res.steps])
        np.testing.assert_allclose(ours, np.asarray(ref[key]), atol=1e-8)
    np.testing.assert_allclose(np.stack([s["power"]["ankle"] for s in res.steps]),
                               np.asarray(ref["step_A_P"]), atol=1e-8)
    assert [s["bad"] for s in res.steps] == list(np.asarray(ref["step_bad"]).astype(bool))


@pytest.mark.skipif(not __import__("os").path.exists(L.MATLAB_BIN), reason="MATLAB not installed")
def test_matlab_harness_runs_and_matches_reference(tmp_path):
    """Where MATLAB is installed, re-execute the original routines and check
    the committed reference is what they produce (guards the harness and the
    exporter, ~1 min)."""
    ref = L.load_matlab_reference(f"{DATA}/matlab_reference/S1_AM_RP_legacy.mat")
    m = L.run_matlab_legacy(f"{DATA}/S1_AM_Static.c3d", f"{DATA}/S1_AM_RP.c3d", "AM", "RP",
                            tmp_path / "out.mat", body_mass=float(ref["A"][0]), workdir=tmp_path)
    np.testing.assert_array_equal(m["HS"], ref["HS"])
    np.testing.assert_allclose(m["GlobalMankle"], ref["GlobalMankle"], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(np.asarray(m["step_hip_Mloc"]), np.asarray(ref["step_hip_Mloc"]), atol=1e-9)
