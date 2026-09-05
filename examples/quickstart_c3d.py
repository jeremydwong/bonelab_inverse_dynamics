"""Static-calibrated inverse dynamics from raw c3d, in 8 steps.

The c3d variant of examples/quickstart.py, for studies that ship a STATIC
calibration trial alongside the dynamic ones (here: the Asics running study,
``S1_<shoe>_{Static,PI,RP}.c3d``).  The static trial carries anatomical
markers that are removed before running (medial ankle/knee), so joint centres
and segment frames are measured once in the static pose and carried through
the dynamic trials as virtual points on the tracking-cluster fits.

Run:    uv run python examples/quickstart_c3d.py [data_dir] [condition] [trial]
        e.g.  uv run python examples/quickstart_c3d.py ~/Downloads/asics AM RP
Output: reports/c3d_quickstart_report.html
"""

import sys
from pathlib import Path

import numpy as np

from boneid.core import AnalysisParams, energy_audit, inverse_dynamics
from boneid.io_c3d import build_chain, calibrate_static, load_c3d
from boneid.report import c3d_report_html, c3d_viewer_html, right_step_events

# ---- study config (the only thing you edit) -------------------------------
DATA_DIR = Path.home() / "Downloads" / "asics"
SUBJECT = "S1"
CONDITION = "AM"                # shoe condition: AG / AH / AM
TRIAL = "RP"                    # dynamic trial: PI / RP
SEX = "male"                    # not recorded in the export; de Leva table
CONTACT_THRESHOLD_N = 50.0      # vertical force marking ground contact
KINEMATIC_LOWPASS_HZ = 12.0     # segment-pose filter before differentiation
TREADMILL_ORIGIN = np.zeros(3)  # fixed lab point the ground wrench is about


def main(data_dir=DATA_DIR, condition: str = CONDITION, trial_name: str = TRIAL):
    data_dir = Path(data_dir)

    # Step 1 — static trial: joint centres, segment frames, cluster shapes,
    # body mass and segment lengths, all measured in the standing pose.
    static = load_c3d(data_dir / f"{SUBJECT}_{condition}_Static.c3d")
    cal = calibrate_static(static, sex=SEX)

    # Step 2 — dynamic trial: markers (de-spiked, gaps bridged) and plate
    # forces/COP/free moment in the lab frame, reaction on the subject.
    trial = load_c3d(data_dir / f"{SUBJECT}_{condition}_{trial_name}.c3d")

    # Step 3 — analysis parameters: torques about a fixed lab point, no COP
    # division.  Force and kinematics share the SAME low-pass cutoff: with
    # running impacts, filtering the kinematics at 12 Hz but not the force
    # leaves large artifact oscillations in the knee/hip moments
    # (Bisseling & Hof 2006; Kristianslund et al. 2012).
    params = AnalysisParams(contact_threshold_n=CONTACT_THRESHOLD_N,
                            lowpass_hz=KINEMATIC_LOWPASS_HZ,
                            force_lowpass_hz=KINEMATIC_LOWPASS_HZ)

    # Step 4 — the chain: the static calibration carried by per-frame SVD
    # cluster fits (virtual joint centres), a de Leva skeleton from the
    # STATIC lengths, and the ground wrench attributed bout-by-bout to the
    # instrumented right foot (both feet strike the same belt here).
    skeleton, kin, ground, quality = build_chain(
        trial, cal, point=TREADMILL_ORIGIN, threshold_n=CONTACT_THRESHOLD_N)

    # Step 5 — inverse dynamics, bottom-up foot -> shank -> thigh -> pelvis.
    # Joints are [ankle, knee, hip]; the residual at the top carries the
    # unmodelled upper body + left leg (only one leg is instrumented).
    idres = inverse_dynamics(skeleton, kin, ground, params)

    # Step 6 — energy audit (always): d(KE+PE)/dt must equal the summed
    # wrench powers; the imbalance is the correctness instrument.
    audit = energy_audit(skeleton, kin, ground, idres, params)

    # Step 7 — the outputs at a glance (skip the first/last second: the
    # record starts mid-stance and zero-lag filtering rings at the edges).
    steps = right_step_events(ground, CONTACT_THRESHOLD_N)
    edge = slice(int(kin.rate), -int(kin.rate))
    print(f"{condition}/{trial_name}: {cal.body_mass:.1f} kg, "
          f"{len(steps)} right steps; peak torques "
          + ", ".join(f"{name} {np.abs(idres.joint_torque[edge, j, 0]).max():.0f} N m"
                      for j, name in enumerate(skeleton.joint_names)))

    # Step 8 — report: figures, the animation, the energy audit, the tables.
    viewer_html = c3d_viewer_html(trial, cal, skeleton, kin, ground, quality,
                                  params)
    html = c3d_report_html(
        trial, static, cal, skeleton, kin, ground, quality, idres, audit,
        params, viewer_html=viewer_html,
        source_note=f"Subject {SUBJECT}, shoe condition {condition}, "
                    f"trial {trial_name}.")
    out = (Path(__file__).resolve().parents[1] / "reports"
           / "c3d_quickstart_report.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else DATA_DIR,
         args[1] if len(args) > 1 else CONDITION,
         args[2] if len(args) > 2 else TRIAL)
