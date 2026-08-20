"""Whole-body power decompositions: COM power, peripheral power, UD foot power.

The simulated squat-to-stand is the instrument: its ground wrench is the EXACT
whole-body Newton-Euler wrench for the prescribed motion, every segment's mass
is modeled (nothing above the chain), and the foot is welded to the ground.
That makes three identities checkable to numerical precision:

TEST-COM-IDENTITY    F_grf . v_com == d/dt(1/2 M |v_com|^2 + M g h_com)
TEST-KOENIG          com_power + peripheral_power == d(KE+PE)/dt
TEST-COM-VEL-GRF     integrating the GRF recovers the kinematic COM velocity
TEST-UD-RIGID        a rigid welded foot has zero unified-deformable power
plus a real-data smoke test on p1 trial 13.
"""

import os

import numpy as np
import pytest

from boneid.core import (GRAVITY, AnalysisParams, chain_kinematics, com_power,
                         com_velocity_from_grf, detect_contact, energy_audit,
                         finite_difference, foot_power_ud, inverse_dynamics,
                         peripheral_power)
from boneid.simulate import simulate_squat_to_stand

RAW = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)

P1 = "/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat"
needs_p1 = pytest.mark.skipif(not os.path.exists(P1),
                              reason="validation data not present")

# central differences are only second-order accurate at the very ends; every
# identity below is checked on the interior.
EDGE = 5
INT = slice(EDGE, -EDGE)


def _s2s():
    """Squat-to-stand plus the KINEMATIC whole-body COM state."""
    skeleton, kin, ground, _ = simulate_squat_to_stand()
    ck = chain_kinematics(skeleton, kin, RAW.lowpass_hz, RAW.filter_order)
    total_mass = skeleton.mass.sum()
    com = np.einsum("s,tsi->ti", skeleton.mass, ck["com"]) / total_mass
    v_com = np.einsum("s,tsi->ti", skeleton.mass, ck["v_com"]) / total_mass
    return skeleton, kin, ground, total_mass, com, v_com


def test_com_identity():
    """TEST-COM-IDENTITY — Newton for the COM, nothing more.

    The whole external force on the body is the GRF plus gravity, so
    F_grf . v_com must equal the rate of change of the COM energy
    1/2 M |v_com|^2 + M g h_com. With every mass modeled and the exact
    simulated wrench this is an identity; the only error is the difference
    between the central-difference derivative of the energy and the analytic
    one, O(dt^2).
    """
    skeleton, kin, ground, total_mass, com, v_com = _s2s()
    power = com_power(ground.force, v_com)
    e_com = (0.5 * total_mass * np.sum(v_com ** 2, axis=1)
             - total_mass * (com @ GRAVITY))
    de_dt = finite_difference(e_com, kin.rate)

    err = np.abs(power[INT] - de_dt[INT]).max()
    scale = np.abs(de_dt[INT]).max()
    assert scale > 100.0                       # ~530 W peak, a real signal
    assert err < 1e-3                          # achieved 6.6e-4 W
    assert err / scale < 1e-5                  # achieved 1.2e-6 relative


def test_koenig_decomposition():
    """TEST-KOENIG — energy splits cleanly into COM + peripheral terms.

    Koenig: KE+PE = [COM part] + sum_i[1/2 m_i |v_i-v_com|^2 + 1/2 w I w].
    Differentiating and using Newton for the COM gives
    com_power + peripheral_power = d(KE+PE)/dt, with d(KE+PE)/dt taken from
    the independent `energy_audit` instrument.
    """
    skeleton, kin, ground, total_mass, com, v_com = _s2s()
    idres = inverse_dynamics(skeleton, kin, ground, RAW)
    audit = energy_audit(skeleton, kin, ground, idres, RAW)

    p_com = com_power(ground.force, v_com)
    p_per = peripheral_power(skeleton, kin, v_com, RAW)

    err = np.abs((p_com + p_per)[INT] - audit.de_dt[INT]).max()
    scale = np.abs(audit.de_dt[INT]).max()
    assert scale > 100.0                       # ~538 W peak
    assert err < 1e-3                          # achieved 6.6e-4 W
    assert err / scale < 1e-5                  # achieved 1.2e-6 relative
    # the peripheral term is not trivially zero: the segments really do move
    # relative to the COM (peak ~11 W here — small next to the 538 W COM term
    # because the 48 kg torso dominates the mass and rides with the COM)
    assert np.abs(p_per).max() > 5.0


def test_com_velocity_from_grf():
    """TEST-COM-VEL-GRF — integrate the GRF, recover the COM velocity.

    Tolerance achievable here is very tight (< 1e-5 m/s against a 0.48 m/s
    signal) for two reasons: the simulated wrench is exact rather than
    measured, and the squat-to-stand starts and ends at rest with a
    minimum-jerk profile, so removing the mean is exactly the right way to fix
    the integration constant. Real treadmill data only gets this if the window
    is an integer number of strides; the residual error there is drift and
    force-plate zero offset, not integration error.
    """
    skeleton, kin, ground, total_mass, com, v_com = _s2s()
    v_grf = com_velocity_from_grf([ground.force], total_mass, kin.rate)
    v_ref = v_com - v_com.mean(axis=0)         # same detrending

    err = np.abs(v_grf - v_ref).max()
    assert np.abs(v_ref).max() > 0.4
    assert err < 1e-4                          # achieved 7.7e-6 m/s
    assert np.sqrt(np.mean((v_grf - v_ref) ** 2)) < 1e-5   # achieved 2.3e-6

    # a bare [T,3] array is accepted as "one limb"
    assert np.allclose(v_grf, com_velocity_from_grf(ground.force, total_mass,
                                                    kin.rate))
    # without detrending the integration constant is simply missing: the
    # difference from the detrended version is a constant offset
    v_raw = com_velocity_from_grf([ground.force], total_mass, kin.rate,
                                  detrend=False)
    assert np.allclose(v_raw - v_raw.mean(axis=0), v_grf)


def test_ud_power_rigid_foot_is_zero():
    """TEST-UD-RIGID — the point of the unified-deformable measure.

    In the squat-to-stand the foot is welded to the ground: zero velocity,
    zero angular velocity, while the GRF runs to ~800 N and the joints deliver
    hundreds of watts. A rigid, motionless foot cannot transmit any power
    distally, so P_UD must be zero — any non-zero value would be measuring
    deformation that is not in the model.
    """
    skeleton, kin, ground, _ = simulate_squat_to_stand()
    idres = inverse_dynamics(skeleton, kin, ground, RAW)
    audit = energy_audit(skeleton, kin, ground, idres, RAW)
    p_ud = foot_power_ud(skeleton, kin, ground, RAW)

    peak_joint = np.abs(audit.joint_power).max()
    assert peak_joint > 100.0                  # ~335 W
    assert np.abs(ground.force[:, 2]).max() > 500.0
    assert np.abs(p_ud).max() < 1e-9 * max(peak_joint, 1.0)

    # P_UD does not depend on which horizontal plane the COP was reduced to:
    # sliding the COP along the force's line of action is a no-op.
    p_low = foot_power_ud(skeleton, kin, ground, RAW, plane_height=-0.1746)
    assert np.abs(p_ud - p_low).max() < 1e-9


# ---------------------------------------------------------------------------
# Real data: p1 trial 13 (fast walking, ~1.74 m/s belt)
# ---------------------------------------------------------------------------

@needs_p1
def test_real_com_and_ud_power():
    """Smoke test on p1 trial 13, both belts.

    Checks magnitudes rather than identities: with only legs+pelvis modeled
    there is no closed energy budget to check against. Recorded values for
    this trial (85.0 kg, belt ~1.74 m/s, 5 strides):

        peak |P_com|   right 510 W, left 425 W  (both plates, detrended v_com)
        peak |P_UD|    right 762 W, left 729 W in the LAB frame; ~280 W once
                       evaluated in the belt frame (see foot_power_ud docs)
        stance work    net negative on every stance, ~-45 J/stance in the belt
                       frame, with a clear negative (collision) phase early.
    """
    from boneid import io_v3d

    trial = io_v3d.load_v3d_trial(P1, 13)
    body_mass = io_v3d.estimate_body_mass(trial)
    assert 70.0 < body_mass < 100.0
    params = AnalysisParams()

    chains = {s: io_v3d.build_chain(trial, side=s) for s in ("r", "l")}
    grounds = {s: chains[s][2] for s in ("r", "l")}

    # --- COM power from both plates -------------------------------------
    v_com = com_velocity_from_grf([grounds["r"].force, grounds["l"].force],
                                  body_mass, trial.rate)
    assert np.isfinite(v_com).all()
    assert np.allclose(v_com.mean(axis=0), 0.0, atol=1e-9)
    # COM speed fluctuation of a walker: tens of cm/s, not metres/s
    assert 0.02 < v_com.std(axis=0).max() < 1.0

    peaks = {}
    for side in ("r", "l"):
        p = com_power(grounds[side].force, v_com)
        assert np.isfinite(p).all()
        peaks[side] = float(np.abs(p).max())
        # plausible whole-body external power for one limb of a fast walker
        assert 20.0 < peaks[side] < 600.0
    assert peaks["r"] > 100.0 and peaks["l"] > 100.0

    # --- unified-deformable foot power ----------------------------------
    for side in ("r", "l"):
        skeleton, kin, ground = chains[side]
        p_ud = foot_power_ud(skeleton, kin, ground, params,
                             plane_height=-0.1746)   # plate surface
        assert np.isfinite(p_ud).all()
        assert p_ud.shape == (len(trial.t),)
        # plane choice is irrelevant (see foot_power_ud docstring)
        assert np.abs(p_ud - foot_power_ud(skeleton, kin, ground,
                                           params)).max() < 1e-6

        _, events = detect_contact(ground.force[:, 2],
                                   params.contact_threshold_n, min_gap=12)
        # 5 strides -> at least 4 full stance phases plus edge fragments
        full = [(s, e) for s, e in events if e - s > 30]
        assert len(full) >= 4

        dt = 1.0 / trial.rate
        for s, e in full:
            seg = p_ud[s:e + 1]
            work = np.trapezoid(seg, dx=dt)
            neg = np.trapezoid(np.minimum(seg, 0.0), dx=dt)
            early = seg[:len(seg) // 2]
            early_neg = np.trapezoid(np.minimum(early, 0.0), dx=dt)
            assert np.isfinite(work)
            assert abs(work) > 1e-3            # not identically zero
            assert neg < -1.0                  # real absorption exists
            assert early_neg < -0.5            # collision phase absorbs
