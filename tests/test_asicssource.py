"""The legacy MATLAB's physics mistakes, reproduced on purpose.

`core.inverse_dynamics_whole_body_asicssource` is an A/B twin of
`inverse_dynamics_whole_body`: same data, same anthropometry, same joint
centres, same chain, same filtering — only the three calculations the Asics-era
MATLAB got wrong differ.

  (a) global inertia formed one-sidedly, I_G = R I  (not R I R^T)
  (b) Euler's equation missing the gyroscopic term omega x (I omega)
  (c) "angular acceleration" as the second derivative of zxy Cardan angles
      (PRESUMED — `RoomSegAng.m` is lost; switchable via `cardan_alpha`)

Two things are checked. First, that the reimplementation really does reproduce
bug (b): on the KEY-GYRO body, whose entire required torque IS the gyroscopic
term, the asicssource path must MISS it. Second, on real walking data, that the
difference is nonzero but bounded, and that the (unchanged, correct) energy
audit closes worse against the asics torques than against the correct ones —
the audit used as a bug detector.
"""

import os

import numpy as np
import pytest

from boneid.core import (AnalysisParams, LegacyPhysics, cardan_alpha,
                         energy_audit_whole_body, inverse_dynamics,
                         inverse_dynamics_whole_body,
                         inverse_dynamics_whole_body_asicssource)
from boneid.simulate import simulate_gyro

RAW = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)
ASICS_AB = LegacyPhysics(one_sided_inertia=True, gyroscopic=False)

DATA = ("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
        "p1_5StridesData.mat")
TRIAL = 13
EDGE = 10


# ---------------------------------------------------------------------------
# The bug, on the simulation built to expose it
# ---------------------------------------------------------------------------

def test_asicssource_misses_the_gyroscopic_torque():
    """KEY-GYRO with the legacy physics: constant world angular velocity about
    a non-principal axis, so alpha = 0 and the whole required torque is
    omega x (I_w omega). The correct recursion recovers it; the asicssource
    variant returns ~zero, i.e. it is wrong by the FULL gyroscopic magnitude.
    """
    skeleton, kin, ground, truth = simulate_gyro()
    interior = slice(5, -5)
    scale = np.abs(truth[interior]).max()
    assert scale > 0.1                       # not a trivially zero test

    good = inverse_dynamics(skeleton, kin, ground, RAW)
    bad = inverse_dynamics(skeleton, kin, ground, RAW, physics=ASICS_AB)

    good_err = np.abs(good.residual_torque[interior] - truth[interior]).max()
    bad_err = np.abs(bad.residual_torque[interior] - truth[interior]).max()
    assert good_err < 1e-2 * scale           # correct physics: matches truth
    assert np.abs(bad.residual_torque[interior]).max() < 1e-2 * scale
    assert bad_err > 0.9 * scale             # wrong by ~the whole term


def test_one_sided_inertia_is_the_legacy_product():
    """Flag (a) really is I_G = R I_local, and it is not symmetric."""
    skeleton, kin, ground, _ = simulate_gyro()
    r = kin.r_world[:, 0]
    i_local = skeleton.inertia_local[0]
    one_sided = np.einsum("tij,jk->tik", r, i_local)
    asym = np.abs(one_sided - np.swapaxes(one_sided, 1, 2)).max()
    assert asym > 0.05                       # the legacy "inertia" isn't symmetric
    # and it differs from the similarity transform by an order-1 amount
    from boneid.core import inertia_world
    assert np.abs(one_sided - inertia_world(r, i_local)).max() > 0.05


def test_cardan_alpha_differs_from_true_alpha():
    """Flag (c): the Cardan double derivative is NOT d(omega)/dt for a
    rotation whose axis is not a coordinate axis (here a fixed non-principal
    world axis at constant rate, where the true alpha is exactly zero)."""
    _, kin, _, _ = simulate_gyro()
    a_cardan = cardan_alpha(kin.r_world, kin.rate, cutoff=0.0)
    interior = slice(20, -20)
    assert np.abs(a_cardan[interior]).max() > 1.0     # true alpha is 0


# ---------------------------------------------------------------------------
# The A/B on real data
# ---------------------------------------------------------------------------

pytestmark_data = pytest.mark.skipif(
    not os.path.exists(DATA), reason="local validation data not available")


@pytest.fixture(scope="module")
def ab():
    from boneid.io_v3d import build_chain, build_upper_body, load_v3d_trial

    trial = load_v3d_trial(DATA, TRIAL)
    params = AnalysisParams(lowpass_hz=12.0, force_lowpass_hz=0.0)
    skr, kinr, grr = build_chain(trial, side="r")
    skl, kinl, grl = build_chain(trial, side="l")
    sku, kinu = build_upper_body(trial)
    chain = (skr, kinr, grr, skl, kinl, grl, sku, kinu)
    return {
        "chain": chain, "params": params,
        "correct": inverse_dynamics_whole_body(*chain, params),
        "abc": inverse_dynamics_whole_body_asicssource(*chain, params,
                                                       cardan_alpha=True),
        "ab": inverse_dynamics_whole_body_asicssource(*chain, params,
                                                      cardan_alpha=False),
    }


@pytestmark_data
def test_asicssource_torques_differ_but_stay_bounded(ab):
    """Every leg joint torque moves — by more than rounding, by less than the
    signal. Walking is near-planar, so the legacy mistakes are real but small:
    the assertion window is deliberately wide (0.01 to 30 N m RMS) because the
    point is that the difference EXISTS and is BOUNDED, not its exact size."""
    sl = slice(EDGE, -EDGE)
    for variant in ("ab", "abc"):
        for side in ("right", "left"):
            good = getattr(ab["correct"], side)
            bad = getattr(ab[variant], side)
            for j in range(3):
                d = (bad.joint_torque - good.joint_torque)[sl, j]
                rms = float(np.sqrt((d ** 2).sum(axis=1).mean()))
                assert 0.01 < rms < 30.0, (variant, side, j, rms)


@pytestmark_data
def test_asicssource_leaves_the_ground_reaction_alone(ab):
    """Sanity: only the moment balance is touched. Joint FORCES are pure
    F = m a - F_d - m g and must be identical to the last bit, which is what
    makes the torque comparison an isolation of the Euler equation."""
    for variant in ("ab", "abc"):
        for side in ("right", "left"):
            np.testing.assert_allclose(
                getattr(ab[variant], side).joint_force,
                getattr(ab["correct"], side).joint_force, atol=1e-12)


@pytestmark_data
def test_energy_audit_detects_the_asics_torques(ab):
    """The audit is computed with the CORRECT physics in both cases and only
    the torques are swapped, so it is an independent instrument: feeding it
    torques from a broken Euler equation must make d(KE+PE)/dt fail to match
    the summed joint powers by MORE than it does with correct torques."""
    sl = slice(EDGE, -EDGE)

    def imbalance(res):
        audit = energy_audit_whole_body(*ab["chain"], res, ab["params"])
        return float(np.sqrt((audit.imbalance[sl] ** 2).mean()))

    good = imbalance(ab["correct"])
    for variant in ("ab", "abc"):
        bad = imbalance(ab[variant])
        assert bad >= good
    # and the failure is not marginal: the correct run closes to a few watts
    assert good < 5.0
    assert imbalance(ab["abc"]) > 2.0 * good
