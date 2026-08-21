"""Whole-body inverse dynamics in 10 steps.

This is the canonical run-through: every step is one high-level call, in the
order the physics needs them. It is the same shape as the legacy Asics batch
script (study config up top, one linear pipeline below) with the hardcoding
removed — swap the config for your study and the ten steps do not change.

Run:    uv run python examples/quickstart.py [mat_path] [trial_index]
Output: reports/quickstart_report.html  (the example report)
"""

import sys
from pathlib import Path

import numpy as np

from boneid.core import (AnalysisParams, detect_contact,
                         energy_audit_whole_body, inverse_dynamics_whole_body)
from boneid.io_v3d import (build_chain, build_upper_body, crossover_flags,
                           estimate_body_mass, load_v3d_trial)
from boneid.report import p1_viewer_html, quickstart_report_html

# ---- study config (the only thing you edit) -------------------------------
MAT_PATH = "/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat"
TRIAL_INDEX = 13
CONTACT_THRESHOLD_N = 20.0      # vertical force marking ground contact
KINEMATIC_LOWPASS_HZ = 12.0     # segment-pose filter before differentiation
TREADMILL_ORIGIN = np.zeros(3)  # fixed lab point the ground wrench is about


def main(mat_path: str = MAT_PATH, trial_index: int = TRIAL_INDEX):
    # Step 1 — mocap data: markers, landmarks, rates (Visual3D export).
    trial = load_v3d_trial(mat_path, trial_index)

    # Step 2 — force data: dual-belt plate forces/COP at the analog rate ride
    # along on the trial; nothing to do but look at them.
    #   trial.force [Ta, plate, 3], trial.cop, trial.free_moment, trial.plates

    # Step 3 — skeleton: body mass from the force record (mean vertical GRF),
    # segment parameters from de Leva (1996) regressions. build_chain in
    # step 5 does this automatically; done explicitly here so you see it.
    body_mass = estimate_body_mass(trial, threshold_n=CONTACT_THRESHOLD_N)

    # Step 4 — treadmill definition & analysis parameters: torques are taken
    # about a fixed lab point (the treadmill origin) — no center-of-pressure
    # division anywhere. Force anti-alias filtering already happened in the
    # loader, so force_lowpass_hz stays 0.
    params = AnalysisParams(contact_threshold_n=CONTACT_THRESHOLD_N,
                            lowpass_hz=KINEMATIC_LOWPASS_HZ,
                            force_lowpass_hz=0.0)

    # Step 5 — segment kinematics: joint centers from marker pairs/landmarks,
    # segment frames from anatomical axes, ground wrench about the origin —
    # one call per side, plus one for the upper body (torso+head and both
    # arms, each arm a contiguous distal->proximal sub-chain).
    skel_r, kin_r, ground_r = build_chain(trial, side="r",
                                          point=TREADMILL_ORIGIN,
                                          threshold_n=CONTACT_THRESHOLD_N)
    skel_l, kin_l, ground_l = build_chain(trial, side="l",
                                          point=TREADMILL_ORIGIN,
                                          threshold_n=CONTACT_THRESHOLD_N)
    skel_u, kin_u = build_upper_body(trial, threshold_n=CONTACT_THRESHOLD_N)

    # Step 6 — contact events and crossover flags.
    contact = {side: detect_contact(g.force[:, 2], CONTACT_THRESHOLD_N)
               for side, g in (("r", ground_r), ("l", ground_l))}
    crossover = {side: crossover_flags(trial, side, CONTACT_THRESHOLD_N)
                 for side in ("r", "l")}

    # Step 7 — inverse dynamics: both legs bottom-up into a shared pelvis,
    # both arms bottom-up into the torso, then the torso's own balance.
    whole = inverse_dynamics_whole_body(skel_r, kin_r, ground_r,
                                        skel_l, kin_l, ground_l,
                                        skel_u, kin_u, params)

    # Step 8 — the outputs: net joint torques, the L5S1 joint wrench, and the
    # residual wrench — now at the TORSO, with every segment modelled, so its
    # mean is near zero instead of carrying the missing upper body.
    #   whole.right/.left .joint_torque [T, (ankle,knee,hip), 3]  (lab frame)
    #   whole.l5s1_force / .l5s1_torque   [T,3]  ON pelvis BY torso
    #   whole.residual_force / .residual_torque  [T,3] at the torso COM
    torques = {"r": whole.right.joint_torque, "l": whole.left.joint_torque}

    # Step 9 — energy audit (always): d(KE+PE)/dt must equal the summed
    # wrench powers; the imbalance is the correctness instrument.
    audit = energy_audit_whole_body(skel_r, kin_r, ground_r,
                                    skel_l, kin_l, ground_l,
                                    skel_u, kin_u, whole, params)

    # Step 10 — report (figures + numbers + 3-D animation).
    viewer_html = p1_viewer_html(
        trial, (skel_r, kin_r, ground_r, skel_l, kin_l, ground_l),
        skel_u, kin_u, params)
    html = quickstart_report_html(
        trial=trial, body_mass=body_mass, params=params,
        chains={"r": (skel_r, kin_r, ground_r), "l": (skel_l, kin_l, ground_l)},
        upper=(skel_u, kin_u),
        contact=contact, crossover=crossover, whole=whole, audit=audit,
        viewer_html=viewer_html)
    out = Path(__file__).resolve().parents[1] / "reports" / "quickstart_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else MAT_PATH,
         int(args[1]) if len(args) > 1 else TRIAL_INDEX)
