"""Tests for boneid.io_v3d against the published Visual3D export.

Data: /Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat
(Fukuchi et al., Sci Data 2022).  Skipped when the file is absent.

Every threshold below is loose relative to the numbers actually observed; the
observed extremes over all 28 populated p1 trials x both legs are quoted in
comments so a regression shows up as a wide miss, not a coin flip.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from boneid import core
from boneid.io_v3d import (
    ANALOG_RATE,
    MOCAP_RATE,
    build_chain,
    crossover_flags,
    deleva_skeleton,
    estimate_body_mass,
    foot_centroid,
    joint_centers,
    load_v3d_trial,
    load_v3d_trials,
    plate_for_side,
    reference_series,
)

DATA = os.path.expanduser(
    "~/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat")

pytestmark = pytest.mark.skipif(not os.path.exists(DATA),
                                reason=f"test data not found: {DATA}")

PLATE_HEIGHT = -0.1746          # m, treadmill surface in the lab frame


@pytest.fixture(scope="module")
def trials():
    return load_v3d_trials(DATA)


@pytest.fixture(scope="module")
def trial(trials):
    return trials[0]


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2,
                                        axis=-1))))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_file_has_33_slots_28_populated():
    slots = load_v3d_trials(DATA, skip_empty=False)
    assert len(slots) == 33
    populated = [i for i, s in enumerate(slots) if s is not None]
    assert len(populated) == 28
    # p1 slots 25..29 are empty placeholders for conditions not collected
    assert [i for i in range(33) if i not in populated] == [25, 26, 27, 28, 29]


def test_load_single_trial_matches_bulk_load(trial):
    one = load_v3d_trial(DATA, 0)
    assert one.index == trial.index == 0
    assert np.allclose(one.markers["RASI"], trial.markers["RASI"])


def test_shapes_and_rates_consistent(trials):
    for tr in trials:
        n_t = len(tr.t)
        n_a = len(tr.t_analog)
        assert tr.rate == MOCAP_RATE == 120.0
        assert tr.analog_rate == ANALOG_RATE == 1200.0
        assert np.allclose(np.diff(tr.t), 1.0 / MOCAP_RATE)
        assert np.allclose(np.diff(tr.t_analog), 1.0 / ANALOG_RATE)
        # analog window is 10x the mocap window minus a few trailing samples
        assert 0 <= 10 * n_t - n_a <= 20
        assert len(tr.markers) == 35 and len(tr.markers_proc) == 35
        for name, m in tr.markers.items():
            assert m.shape == (n_t, 3), name
            assert np.isfinite(m).all(), name
        for name in ("HHL", "HHR"):
            assert tr.landmarks[name].shape == (n_t, 3)
        assert tr.force.shape == tr.cop.shape == tr.free_moment.shape
        assert tr.force.shape == (n_a, 2, 3)
        assert len(tr.plates) == 2
        for p in tr.plates:
            assert p.corners.shape == (4, 3)
            # plate surface height, and a right-handed plate frame
            assert np.allclose(p.corners[:, 2], PLATE_HEIGHT, atol=1e-3)
            assert np.isclose(np.linalg.det(p.r_lab), 1.0)
        # V3D reference series live on the mocap clock
        for key in ("rSkAngVel", "rShProxEndPos", "r_ank_moment"):
            assert reference_series(tr, key).shape == (n_t, 3)
        assert len(tr.t_full) == 7200 and len(tr.t_analog_full) == 72000


def test_markers_are_in_metres_and_plausible(trial):
    # the runner stands on a treadmill at z ~ -0.17; ASIS ~ 1 m up
    assert 0.8 < trial.markers["RASI"][:, 2].mean() < 1.2
    assert -0.2 < trial.markers["RCAL"][:, 2].mean() < 0.4


# ---------------------------------------------------------------------------
# Body mass
# ---------------------------------------------------------------------------

def test_body_mass_plausible_and_consistent(trials):
    masses = np.array([estimate_body_mass(tr) for tr in trials])
    assert np.all((masses > 40.0) & (masses < 120.0))
    # observed for p1: 84.60 .. 85.07 kg, i.e. +-0.3% about the mean
    assert np.ptp(masses) / masses.mean() < 0.05


# ---------------------------------------------------------------------------
# Force plates / alignment
# ---------------------------------------------------------------------------

def test_right_foot_is_on_plate_2(trials):
    for tr in trials:
        assert plate_for_side(tr, "r") == 1
        assert plate_for_side(tr, "l") == 0


def test_stance_cop_lies_under_the_foot(trials):
    """The crux of the alignment: if the cropped analog window were offset
    from the marker window, the COP would wander off the stance foot.

    Observed worst case over all trials/sides: RMS 0.100 m, 99th pct 0.142 m
    of horizontal COP-to-foot-centroid distance (the centroid sits between
    heel, 5th metatarsal and ankle, so ~0.1 m is the anatomy, not an error).
    """
    for tr in trials:
        for side in ("r", "l"):
            _, _, ground = build_chain(tr, side=side)
            cop, _ = core.cop_from_wrench(ground.force, ground.moment,
                                          ground.point,
                                          plane_height=PLATE_HEIGHT,
                                          min_fz=100.0)
            cent = foot_centroid(tr, side)
            stance = ground.force[:, 2] > 200.0
            d = np.linalg.norm(cop[stance, :2] - cent[stance, :2], axis=1)
            assert np.sqrt(np.mean(d ** 2)) < 0.15, (tr.index, side)
            assert np.percentile(d, 99) < 0.3, (tr.index, side)


def test_ground_wrench_zeroed_off_contact_and_on_mocap_clock(trial):
    _, kin, ground = build_chain(trial, side="r")
    assert ground.rate == trial.rate
    assert np.allclose(ground.t, kin.t)
    assert len(ground.force) == len(trial.t)
    quiet = ground.force[:, 2] < 1.0
    assert quiet.any()
    assert np.abs(ground.moment[quiet]).max() < 5.0
    # peak vertical GRF is 1-3 body weights for walking/running
    bw = estimate_body_mass(trial) * 9.81
    assert 1.0 * bw < ground.force[:, 2].max() < 3.0 * bw


def test_crossover_flags(trial):
    flags = crossover_flags(trial, "r")
    assert flags.dtype == bool and flags.shape == trial.t.shape
    # p1 trial 0 is a slow walk: double support is genuinely ~23% of frames
    assert 0.0 < flags.mean() < 0.5
    # a threshold above every measured force can never flag a crossover
    assert not crossover_flags(trial, "r", threshold_n=1e5).any()


# ---------------------------------------------------------------------------
# Kinematics vs Visual3D
# ---------------------------------------------------------------------------

def test_joint_centers_match_v3d_prox_end_pos(trials):
    """Observed worst case: ankle 4.7 mm, knee 5.1 mm, hip 29.3 mm.

    The hip is looser because Visual3D's thigh proximal end is its own hip
    model, while we take the exported HHR/HHL landmark verbatim.
    """
    for tr in trials:
        for side, pre in (("r", "r"), ("l", "l")):
            jc = joint_centers(tr, side)
            assert rms(jc["ankle"], reference_series(tr, pre + "FtProxEndPos")) < 0.01
            assert rms(jc["knee"], reference_series(tr, pre + "SkProxEndPos")) < 0.01
            assert rms(jc["hip"], reference_series(tr, pre + "ThProxEndPos")) < 0.035


def test_segment_frames_are_rotations(trial):
    _, kin, _ = build_chain(trial, side="r")
    r = kin.r_world
    assert r.shape == (len(trial.t), 4, 3, 3)
    eye = np.einsum("tsij,tskj->tsik", r, r)
    assert np.allclose(eye, np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(r), 1.0, atol=1e-9)


def test_segment_com_matches_v3d(trials):
    """de Leva COM vs Visual3D CGPos, worst case over p1 (both legs):

        foot 0.073 m, shank 0.0075 m, thigh 0.0285 m, pelvis 0.075 m

    Shank and thigh agree to well under 3 cm.  The foot and pelvis do not,
    and should not: Visual3D's foot runs ankle -> a virtual distal point and
    puts the COM at 48% of it, while de Leva measures 44% along heel -> toe;
    the V3D pelvis spans mid-ASIS -> a point 0.135 m below, while ours spans
    L5S1 -> mid-hip.  Both are documented segment-definition differences.
    """
    for tr in trials:
        for side, pre in (("r", "r"), ("l", "l")):
            skel, kin, _ = build_chain(tr, side=side)
            ck = core.chain_kinematics(skel, kin)
            names = [pre + s for s in ("Ft", "Sk", "Th", "Pv")]
            errs = [rms(ck["com"][:, i], reference_series(tr, n + "CGPos"))
                    for i, n in enumerate(names)]
            assert errs[1] < 0.03, (tr.index, side, "shank", errs)
            assert errs[2] < 0.03, (tr.index, side, "thigh", errs)
            assert errs[0] < 0.10 and errs[3] < 0.10, (tr.index, side, errs)


def test_angular_velocity_matches_v3d(trials):
    """Visual3D's *AngVel is a LAB-frame vector in rad/s (not local, not deg).

    Compared on the dominant component (lab x, the medio-lateral axis, i.e.
    sagittal-plane rotation), our shank omega correlates 0.976..0.995 with
    Visual3D's and the amplitude ratio is 0.98-1.02.
    """
    for tr in trials:
        for side, pre in (("r", "r"), ("l", "l")):
            skel, kin, _ = build_chain(tr, side=side)
            ck = core.chain_kinematics(skel, kin)
            v3d = reference_series(tr, pre + "SkAngVel")
            j = int(np.argmax(v3d.std(axis=0)))
            assert j == 0
            ours = ck["omega"][:, 1, j]
            r = np.corrcoef(ours, v3d[:, j])[0, 1]
            assert r > 0.95, (tr.index, side, r)
            assert 0.85 < v3d[:, j].std() / ours.std() < 1.15


# ---------------------------------------------------------------------------
# de Leva parameters
# ---------------------------------------------------------------------------

def test_deleva_skeleton_shapes_and_values():
    skel = deleva_skeleton(80.0, foot_length=0.25, shank_length=0.40,
                           thigh_length=0.42, pelvis_length=0.15)
    assert skel.segment_names == ["foot", "shank", "thigh", "pelvis"]
    assert skel.joint_names == ["ankle", "knee", "hip"]
    assert skel.mass.shape == (4,)
    assert skel.com_local.shape == (4, 3)
    assert skel.inertia_local.shape == (4, 3, 3)
    # de Leva male mass fractions
    assert np.allclose(skel.mass / 80.0, [0.0137, 0.0433, 0.1416, 0.1117])
    # shank COM 44.59% down the knee->ankle axis (local z points distal->prox)
    assert np.allclose(skel.com_local[1], [0.0, 0.0, -0.4459 * 0.40])
    # foot COM measured from the heel, which defaults to the segment origin
    assert np.allclose(skel.com_local[0], [0.0, 0.0, 0.4415 * 0.25])
    for i in range(4):
        d = skel.inertia_local[i]
        assert np.allclose(d, np.diag(np.diag(d)))
        assert np.all(np.diag(d) > 0)
    # I_shank about the ML (transverse) axis y = m (0.249 L)^2, and about the
    # AP (sagittal) axis x = m (0.255 L)^2
    assert np.isclose(skel.inertia_local[1][1, 1],
                      0.0433 * 80.0 * (0.249 * 0.40) ** 2)
    assert np.isclose(skel.inertia_local[1][0, 0],
                      0.0433 * 80.0 * (0.255 * 0.40) ** 2)
    # sanity: a woman's thigh is a larger fraction of body mass than a man's
    fem = deleva_skeleton(80.0, 0.25, 0.40, 0.42, 0.15, sex="female")
    assert fem.mass[2] > skel.mass[2]


def test_build_chain_skeleton_is_scaled_to_the_subject(trial):
    skel, kin, ground = build_chain(trial, side="r")
    assert np.isclose(skel.mass.sum() / estimate_body_mass(trial),
                      0.0137 + 0.0433 + 0.1416 + 0.1117)
    # measured segment lengths: shank/thigh ~0.4 m, foot ~0.23 m heel->toe
    assert 0.30 < skel.length[1] < 0.50
    assert 0.30 < skel.length[2] < 0.50
    assert len(ground.force) == len(kin.t) == len(trial.t)


def test_build_chain_accepts_a_supplied_skeleton(trial):
    given = deleva_skeleton(70.0, 0.24, 0.40, 0.42, 0.12)
    skel, _, _ = build_chain(trial, skeleton=given, side="r")
    assert skel is given


def test_inverse_dynamics_runs_and_stays_finite(trial):
    """Not a validation of the kinetics -- just that the loader's output is
    directly consumable by core.inverse_dynamics and produces sane peaks."""
    skel, kin, ground = build_chain(trial, side="r")
    res = core.inverse_dynamics(skel, kin, ground)
    assert np.isfinite(res.joint_torque).all()
    assert res.joint_torque.shape == (len(trial.t), 3, 3)
    peak = np.abs(res.joint_torque[:, 0]).max()
    assert 20.0 < peak < 500.0     # ankle moment, N m


def test_joint_torques_track_visual3d(trial):
    """End-to-end: the whole loader (crop alignment, units, plate choice, de
    Leva parameters, frames) plus core's Newton-Euler, against Visual3D's own
    lab-frame ProxEndTorque.

    p1 trial 0, right leg: RMS difference 1.6 / 1.7 / 8.6 N m at ankle / knee
    / hip against peaks of 105 / 70 / 79 N m, same sign convention.  Over all
    trials and both legs the worst case is 12.8 / 9.0 / 24.4 N m; the hip is
    the loosest because our pelvis and hip-centre definitions differ from
    Visual3D's.  Link_Model_Based.r_ank_moment equals this divided by body
    mass (peak 1.254 N m/kg x 84.8 kg = 106 N m), which is how the
    normalisation was identified.
    """
    skel, kin, ground = build_chain(trial, side="r")
    res = core.inverse_dynamics(skel, kin, ground,
                                core.AnalysisParams(lowpass_hz=12.0))
    for j, name in enumerate(["rFtProxEndTorque", "rSkProxEndTorque"]):
        assert rms(res.joint_torque[:, j], reference_series(trial, name)) < 15.0
    assert rms(res.joint_torque[:, 2],
               reference_series(trial, "rThProxEndTorque")) < 30.0
    mass = estimate_body_mass(trial)
    assert np.isclose(np.abs(trial.reference["r_ank_moment"]).max() * mass,
                      np.abs(reference_series(trial, "rFtProxEndTorque")).max(),
                      rtol=0.05)
