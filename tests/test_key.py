"""The KEY test suite — small, intelligible, each one guards a distinct way
inverse dynamics can be wrong. Broader coverage lives in test_coverage.py.

KEY-STATIC        static equilibrium: torques = gravity moments, residual = 0
KEY-PENDULUM      analytic single-pendulum pivot torque
KEY-GYRO          the two legacy bugs: R I R^T transform + omega x (I omega)
KEY-WRENCH-EQUIV  COP formulation and origin-wrench formulation agree
TEST-SIMULATED-S2S  3-joint squat-to-stand: zero residual, static limits,
                    and an energy balance that closes across time
"""

import numpy as np
import pytest

from boneid.core import (AnalysisParams, GroundWrench, InverseDynamicsResult,
                         cop_from_wrench, energy_audit, inverse_dynamics,
                         wrench_from_cop)
from boneid.simulate import (simulate_gyro, simulate_pendulum,
                             simulate_squat_to_stand)

RAW = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)


def test_key_static():
    """Freeze the squat pose: every joint torque must equal the analytic
    gravity moment of the segments above it, and the residual must vanish."""
    skeleton, kin, ground, truth = simulate_squat_to_stand()
    # hold frame 0 for all time -> zero velocity/acceleration everywhere
    for arr in (kin.r_world, kin.prox_pos, kin.dist_pos):
        arr[:] = arr[0]
    # exact static ground wrench: supports total weight about the origin
    from boneid.core import GRAVITY
    com0 = kin.prox_pos[0] + np.einsum("sij,sj->si", kin.r_world[0],
                                       skeleton.com_local)
    ground.force[:] = -skeleton.mass.sum() * GRAVITY
    ground.moment[:] = -np.einsum(
        "si->i", np.cross(com0, skeleton.mass[:, None] * GRAVITY))

    res = inverse_dynamics(skeleton, kin, ground, RAW)
    mid = len(kin.t) // 2
    for j, name in enumerate(skeleton.joint_names):
        np.testing.assert_allclose(res.joint_torque[mid, j],
                                   truth["static_torque_start"][name],
                                   atol=1e-8)
    assert np.abs(res.residual_force[mid]).max() < 1e-8
    assert np.abs(res.residual_torque[mid]).max() < 1e-8


def test_key_pendulum():
    """Pivot torque of a swinging pendulum matches I*acc + m g d sin(theta)."""
    skeleton, kin, ground, truth_torque = simulate_pendulum()
    res = inverse_dynamics(skeleton, kin, ground, RAW)
    interior = slice(5, -5)  # one-sided differences pollute the edges
    err = res.residual_torque[interior] - truth_torque[interior]
    scale = np.abs(truth_torque).max()
    assert np.abs(err).max() < 1e-3 * scale


def test_key_gyro():
    """Constant-omega spin about a non-principal axis: required torque is
    omega x (I_w omega) with I_w = R I R^T. The legacy MATLAB gets ~zero
    (dropped gyroscopic term) with a non-symmetric I (one-sided transform)."""
    skeleton, kin, ground, truth = simulate_gyro()
    res = inverse_dynamics(skeleton, kin, ground, RAW)
    interior = slice(5, -5)
    scale = np.abs(truth).max()
    assert scale > 0.1  # the test must not be trivially zero
    err = res.residual_torque[interior] - truth[interior]
    assert np.abs(err).max() < 1e-2 * scale


def test_key_wrench_equiv():
    """Routing the ground load through a COP + free moment and back to a
    wrench about a DIFFERENT fixed point must not change joint torques."""
    skeleton, kin, ground, _ = simulate_squat_to_stand()
    res_a = inverse_dynamics(skeleton, kin, ground, RAW)

    cop, free = cop_from_wrench(ground.force, ground.moment, ground.point,
                                plane_height=0.0, min_fz=1.0)
    other_point = np.array([0.3, -0.2, 0.0])
    moment_b = wrench_from_cop(ground.force, cop, free, other_point)
    ground_b = GroundWrench(t=ground.t, rate=ground.rate, force=ground.force,
                            moment=moment_b, point=other_point)
    res_b = inverse_dynamics(skeleton, kin, ground_b, RAW)

    scale = np.abs(res_a.joint_torque).max()
    np.testing.assert_allclose(res_b.joint_torque, res_a.joint_torque,
                               atol=1e-9 * scale)


def test_simulated_s2s():
    """TEST-SIMULATED-S2S: squat-to-stand on a 3D 3-joint chain with a large
    torso. Checks, in order of physical meaning:
      1. residual wrench at the torso stays ~zero throughout the movement
      2. joint torques hit the analytic static values at both endpoints
      3. the energy audit closes: d(KE+PE)/dt == joint + ground + residual
         power, frame by frame."""
    skeleton, kin, ground, truth = simulate_squat_to_stand()
    res = inverse_dynamics(skeleton, kin, ground, RAW)
    audit = energy_audit(skeleton, kin, ground, res, RAW)
    interior = slice(5, -5)

    # 1. residual ~ 0 relative to body weight / peak torque
    bw = skeleton.mass.sum() * 9.81
    assert np.abs(res.residual_force[interior]).max() < 1e-3 * bw
    tq_scale = np.abs(res.joint_torque).max()
    assert np.abs(res.residual_torque[interior]).max() < 1e-3 * tq_scale

    # 2. static endpoint torques (min-jerk => zero vel/acc at ends)
    for j, name in enumerate(skeleton.joint_names):
        np.testing.assert_allclose(res.joint_torque[6, j],
                                   truth["static_torque_start"][name],
                                   atol=2e-2 * tq_scale)
        np.testing.assert_allclose(res.joint_torque[-7, j],
                                   truth["static_torque_end"][name],
                                   atol=2e-2 * tq_scale)

    # 3. energy balance across time
    power_scale = max(np.abs(audit.de_dt).max(), 1.0)
    assert np.abs(audit.imbalance[interior]).max() < 5e-3 * power_scale

    # sanity: the movement actually does work (rises ~0.4 m at 80 kg)
    rise = audit.potential[-1] - audit.potential[0]
    assert rise > 100.0
