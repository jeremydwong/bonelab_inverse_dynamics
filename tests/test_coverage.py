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
