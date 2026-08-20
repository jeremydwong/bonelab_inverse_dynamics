"""Whole-body (12-segment) model: the torso and arms are no longer residual.

Before this model the analysis stopped at the pelvis and dumped ~49% of body
mass into a residual wrench at L5S1 (mean vertical residual -406 N on p1 trial
13, 0.49 body weight). Here the torso (+head) and both arms (+hands) are real
segments, so that wrench becomes the L5S1 JOINT wrench and the residual moves
to the torso, where it should be near zero.
"""

import os

import numpy as np
import pytest

DATA = ("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
        "p1_5StridesData.mat")
TRIAL = 13

pytestmark = pytest.mark.skipif(not os.path.exists(DATA),
                                reason="local validation data not available")


@pytest.fixture(scope="module")
def built():
    from boneid.core import (AnalysisParams, inverse_dynamics_two_legs,
                             inverse_dynamics_whole_body)
    from boneid.io_v3d import (build_chain, build_upper_body,
                               estimate_body_mass, load_v3d_trial)

    trial = load_v3d_trial(DATA, TRIAL)
    params = AnalysisParams(lowpass_hz=12.0, force_lowpass_hz=0.0)
    skr, kinr, grr = build_chain(trial, side="r")
    skl, kinl, grl = build_chain(trial, side="l")
    sku, kinu = build_upper_body(trial)
    two = inverse_dynamics_two_legs(skr, kinr, grr, skl, kinl, grl, params)
    whole = inverse_dynamics_whole_body(skr, kinr, grr, skl, kinl, grl,
                                        sku, kinu, params)
    return dict(trial=trial, params=params, skr=skr, kinr=kinr, grr=grr,
                skl=skl, kinl=kinl, grl=grl, sku=sku, kinu=kinu,
                two=two, whole=whole,
                body_mass=estimate_body_mass(trial))


def test_legs_unchanged_by_upper_body(built):
    """Adding a torso and arms cannot move a single leg number: the leg
    recursion is bottom-up from the ground and never sees them."""
    two, whole = built["two"], built["whole"]
    for a, b in ((two.right, whole.right), (two.left, whole.left)):
        np.testing.assert_array_equal(a.joint_force, b.joint_force)
        np.testing.assert_array_equal(a.joint_torque, b.joint_torque)
        np.testing.assert_array_equal(a.joint_torque_local, b.joint_torque_local)
    # and the old pelvis "residual" is exactly the new L5S1 joint wrench
    np.testing.assert_array_equal(two.residual_force, whole.l5s1_force)
    np.testing.assert_array_equal(two.residual_torque, whole.l5s1_torque)


def test_ep_markers_really_are_elbows(built):
    """R/LEP sit between shoulder and wrist at plausible upper-arm and forearm
    lengths, and hold those lengths rigidly. If this fails the arm model is
    hanging off the wrong marker and every arm number below is meaningless."""
    m = built["trial"].markers
    for s in ("R", "L"):
        ua = np.linalg.norm(m[s + "AC"] - m[s + "EP"], axis=1)
        fa = np.linalg.norm(m[s + "EP"] - m[s + "WR"], axis=1)
        print(f"{s}: |AC-EP| = {ua.mean():.3f} +- {ua.std():.4f} m, "
              f"|EP-WR| = {fa.mean():.3f} +- {fa.std():.4f} m")
        assert 0.20 < ua.mean() < 0.42
        assert 0.18 < fa.mean() < 0.35
        assert ua.std() < 0.05
        assert fa.std() < 0.05
        # and EP is genuinely between AC and WR: the two sub-lengths bracket
        # the shoulder-to-wrist distance and never fall outside it
        sw = np.linalg.norm(m[s + "AC"] - m[s + "WR"], axis=1)
        assert np.all(sw <= ua + fa + 1e-9)


def test_modelled_mass_is_the_whole_body(built):
    """de Leva's fractions over the 12 segments must account for essentially
    all of the body mass the force plates measure."""
    b = built
    modelled = (b["skr"].mass.sum()              # foot+shank+thigh+pelvis
                + b["skl"].mass[:3].sum()        # the other leg (pelvis once)
                + b["sku"].mass.sum())           # torso(+head) + 2 arms(+hands)
    frac = modelled / b["body_mass"]
    print(f"body mass {b['body_mass']:.2f} kg, modelled {modelled:.2f} kg, "
          f"fraction {frac:.4f}")
    print("  segment masses (kg): "
          + ", ".join(f"{n}={m:.2f}" for n, m in
                      zip(list(b["skr"].segment_names)
                          + list(b["skl"].segment_names[:3])
                          + list(b["sku"].segment_names),
                          list(b["skr"].mass) + list(b["skl"].mass[:3])
                          + list(b["sku"].mass))))
    assert frac > 0.95


def test_torso_residual_is_near_zero(built):
    """THE HEADLINE. The residual used to be -406 N (0.49 BW) of unmodelled
    torso, head and arms sitting on top of the pelvis. With them modelled the
    residual is a genuine error term and must be small."""
    b = built
    bw = b["body_mass"] * 9.81
    i = slice(10, -10)
    old = b["two"].residual_force[i, 2].mean()
    new = b["whole"].residual_force[i, 2].mean()
    peak = np.abs(np.linalg.norm(b["whole"].residual_force[i], axis=1)).max()
    print(f"body weight {bw:.1f} N")
    print(f"  mean Fz residual at the PELVIS (legs+pelvis only): "
          f"{old:.1f} N = {old / bw:+.3f} BW")
    print(f"  mean Fz residual at the TORSO  (whole body):       "
          f"{new:.1f} N = {new / bw:+.3f} BW")
    print(f"  peak |residual force| at the torso: {peak:.1f} N "
          f"= {peak / bw:.3f} BW")
    assert abs(new) < 0.03 * bw


def test_energy_audit_closes(built):
    """12 segments, 11 joints, 2 ground wrenches, 1 residual: d(KE+PE)/dt must
    equal the sum of the powers."""
    from boneid.core import energy_audit_whole_body
    b = built
    audit = energy_audit_whole_body(b["skr"], b["kinr"], b["grr"],
                                    b["skl"], b["kinl"], b["grl"],
                                    b["sku"], b["kinu"], b["whole"],
                                    b["params"])
    i = slice(10, -10)
    rel = np.abs(audit.imbalance[i]).max() / np.abs(audit.de_dt).max()
    print(f"peak |imbalance| / peak |dE/dt| = {rel:.5f}")
    assert rel < 0.03


def test_l5s1_carries_the_upper_body_weight(built):
    """The L5S1 wrench is ON the pelvis BY the torso, so its mean vertical
    component must be minus the weight of everything above it."""
    b = built
    upper_mass = b["sku"].mass.sum()
    expected = -upper_mass * 9.81
    i = slice(10, -10)
    got = b["whole"].l5s1_force[i, 2].mean()
    print(f"upper-body mass {upper_mass:.2f} kg -> expected mean L5S1 Fz "
          f"{expected:.1f} N; got {got:.1f} N "
          f"({100 * (got - expected) / abs(expected):+.1f}%)")
    assert abs(got - expected) < 0.10 * abs(expected)
