"""Export P1 trial 13 (in-repo sample) to OpenSim inputs + boneid's own ID.

Run in the REPO's uv env (it imports boneid):

    uv run python scripts/opensim/export_inputs.py

Other trials (e.g. the step-frequency study) take a source .mat, a 0-based
slot, a file-name tag and an output directory; the defaults are trial 13:

    uv run python scripts/opensim/export_inputs.py \
        --mat .../p1_5StridesData.mat --slot 19 --tag p1_slot19 \
        --out data/opensim/stepfreq/work

Writes into data/opensim/work/:
    p1_trial13_markers.trc   markers [m], OpenSim lab frame, 120 Hz
    p1_trial13_grf.mot       per-foot GRF / COP / free moment, 1200 Hz
    p1_trial13_boneid.npz    boneid right-leg sagittal ID + misc metadata

LAB -> OPENSIM ROTATION (documented once, used everywhere):

    boneid / Visual3D lab: +x subject-left, +y posterior, +z up
    OpenSim ground:        +X anterior (walking dir), +Y up, +Z subject-right

        [X]   [ 0 -1  0] [x]
        [Y] = [ 0  0  1] [y]        (det = +1, a proper rotation, no shift)
        [Z]   [-1  0  0] [z]

COP: Visual3D's COP sits at the plate-origin height z = -0.1746 m, ~0.2 m
below where the feet actually are (the heel marker never goes below
z = +0.056 m).  A point force is equivalent anywhere along its line of action
with the SAME free moment (moving the point along F adds (dr x F) = 0), so we
slide the COP along the force line to lab z = 0 (about the marker floor):
cop' = cop + F * (0 - cop_z) / F_z.  The wrench on the body is unchanged;
only the drawn arrow moves up to the foot.

Unloaded samples (F_z < 20 N, boneid's contact threshold) get zero force
and zero free moment, and the COP is held at the nearest loaded value so it
never jumps to the junk values Visual3D writes there.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from boneid import io_v3d as io
from boneid import report

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLE = os.path.join(ROOT, "data", "p1_trial13_sample.mat")
WORK = os.path.join(ROOT, "data", "opensim", "work")
THRESH = 20.0

R_OS_LAB = np.array([[0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0]])

# data marker name -> name used in the OpenSim model (same name unless noted)
MARKERS = ["RASI", "LASI", "SACR",
           "RGTR", "RTH1", "RTH2", "RTH3", "RLEP", "RMEP",
           "RSH1", "RSH2", "RSH3", "RLML", "RMML", "RCAL", "R5TH",
           "LGTR", "LTH1", "LTH2", "LTH3", "LLEP", "LMEP",
           "LSH1", "LSH2", "LSH3", "LLML", "LMML", "LCAL", "L5TH",
           "RAC", "LAC", "REP", "LEP", "RWR", "LWR"]


def to_os(x: np.ndarray) -> np.ndarray:
    return x @ R_OS_LAB.T


def write_trc(path, t, names, xyz, rate):
    n = len(t)
    with open(path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:g}\t{rate:g}\t{n}\t{len(names)}\tm\t{rate:g}\t1\t{n}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\n")
        f.write("\t\t" + "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}"
                                   for i in range(len(names))) + "\n\n")
        for i in range(n):
            row = [f"{i+1}", f"{t[i]:.6f}"] + [f"{v:.6f}" for v in
                                               xyz[i].reshape(-1)]
            f.write("\t".join(row) + "\n")


def write_mot(path, t, cols: dict[str, np.ndarray]):
    names = list(cols)
    data = np.column_stack([t] + [cols[k] for k in names])
    with open(path, "w") as f:
        f.write(f"{os.path.basename(path)}\nversion=1\nnRows={len(t)}\n"
                f"nColumns={len(names)+1}\ninDegrees=no\nendheader\n")
        f.write("\t".join(["time"] + names) + "\n")
        np.savetxt(f, data, fmt="%.8f", delimiter="\t")


def hold_nearest(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(valid)
    out = x.copy()
    if len(idx) == 0:
        return np.zeros_like(x)
    nearest = idx[np.clip(np.searchsorted(idx, np.arange(len(x))), 0,
                          len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(len(x))) - 1, 0,
                       len(idx) - 1)]
    pick = np.where(np.abs(prev - np.arange(len(x))) <
                    np.abs(nearest - np.arange(len(x))), prev, nearest)
    out[~valid] = x[pick[~valid]]
    return out


def right_stride_s(trial) -> float:
    """Mean interval between right-plate onsets (Fz > THRESH, mocap clock)."""
    p = io.plate_for_side(trial, "r", THRESH)
    on = io.analog_to_mocap(trial, trial.force[:, :, 2])[:, p] > THRESH
    hs = np.flatnonzero(~on[:-1] & on[1:]) + 1
    return float(np.mean(np.diff(trial.t[hs])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mat", default=SAMPLE, help="Visual3D *_5StridesData.mat")
    ap.add_argument("--slot", type=int, default=0, help="0-based trial slot")
    ap.add_argument("--tag", default="p1_trial13", help="output file prefix")
    ap.add_argument("--out", default=WORK, help="output directory")
    ap.add_argument("--syn-d-spm", type=float, default=None,
                    help="synthetic cadence: re-time the trial with "
                         "simulate.step_frequency_trial to f0 + d steps/min "
                         "(f0 = its own right-stride step frequency)")
    ap.add_argument("--syn-body-mass", type=float, default=85.16)
    a = ap.parse_args(argv)
    trial = io.load_v3d_trial(a.mat, a.slot)
    extra = {}
    if a.syn_d_spm is not None:
        from boneid import simulate
        f0 = 120.0 / right_stride_s(trial)              # steps/min
        k = (f0 + a.syn_d_spm) / f0
        trial = simulate.step_frequency_trial(trial, k, a.syn_body_mass)
        extra = {"syn_k": k, "syn_d_spm": a.syn_d_spm, "syn_f0_spm": f0,
                 "syn_source_slot": a.slot}
        print(f"synthetic: f0 {f0:.2f} spm, d {a.syn_d_spm:+g}, k {k:.5f}, "
              f"stride now {right_stride_s(trial):.4f} s")
    export_trial(trial, a.tag, a.out, extra)


def export_trial(trial, tag: str, work: str, extra: dict | None = None):
    """Write `{tag}_markers.trc`, `{tag}_grf.mot`, `{tag}_boneid.npz` into
    `work` for any V3dTrial-like object (measured or synthetic)."""
    os.makedirs(work, exist_ok=True)
    t = trial.t

    # ---- markers (+ the Visual3D hip-centre landmarks as virtual markers)
    names = MARKERS + ["RHJC", "LHJC"]
    xyz = np.stack([trial.markers[k] for k in MARKERS]
                   + [trial.landmarks["HHR"], trial.landmarks["HHL"]], axis=1)
    if np.isnan(xyz).any():
        sys.exit("NaN markers in export; gap-fill first")
    write_trc(os.path.join(work, f"{tag}_markers.trc"), t, names,
              to_os(xyz), trial.rate)

    # ---- GRF per foot (plate assignment by COP proximity, as boneid does)
    plate = {s: io.plate_for_side(trial, s, THRESH) for s in "rl"}
    cols = {}
    for s in "rl":
        p = plate[s]
        f = trial.force[:, p].astype(float).copy()
        cop = trial.cop[:, p].astype(float).copy()
        fm = trial.free_moment[:, p].astype(float).copy()
        on = f[:, 2] > THRESH
        cop[on] = cop[on] + f[on] * ((0.0 - cop[on, 2]) / f[on, 2])[:, None]
        cop = hold_nearest(cop, on)
        f[~on] = 0.0
        fm[~on] = 0.0
        f, cop, fm = to_os(f), to_os(cop), to_os(fm)
        for k, a in (("v", f), ("p", cop)):
            for c, ax in enumerate("xyz"):
                cols[f"ground_force_{s}_{k}{ax}"] = a[:, c]
        for c, ax in enumerate("xyz"):
            cols[f"ground_torque_{s}_{ax}"] = fm[:, c]
    write_mot(os.path.join(work, f"{tag}_grf.mot"), trial.t_analog, cols)

    # ---- boneid inverse dynamics (whole body, right leg reported)
    params = report.p1_params()
    chain, skel_u, kin_u, whole, audit = report.whole_body_run(trial, params)
    tl = whole.right.joint_torque_local          # [T,3(ankle,knee,hip),3] N m
    # boneid: torque ON the distal segment BY the proximal one, in the
    # proximal segment frame (x anterior, y subject-LEFT, z up).  OpenSim ID
    # gives the generalized force of each coordinate = the moment the parent
    # exerts on the child about the coordinate axis, positive in the
    # coordinate's positive direction.  In the Rajagopal model
    #   ankle_angle_r  + = dorsiflexion  -> rotation about ~subject-RIGHT
    #   knee_angle_r   + = FLEXION       -> rotation about subject-LEFT
    #                     (walker knee; unlike gait2392 where + = extension)
    #   hip_flexion_r  + = flexion       -> rotation about subject-RIGHT
    # so OpenSim = [-y_ankle, +y_knee, -y_hip] of boneid's local torque
    # (checked: correlations with OpenSim ID 0.99 / 0.99 / 0.96).
    id_boneid = tl[:, :, 1] * np.array([-1.0, 1.0, -1.0])
    body_mass = float(io.estimate_body_mass(trial, threshold_n=THRESH))
    fz = io.analog_to_mocap(trial, trial.force[:, :, 2])
    np.savez(os.path.join(work, f"{tag}_boneid.npz"),
             t=t, id_moment_boneid=id_boneid, joint_torque_local=tl,
             body_mass=body_mass, plate_r=plate["r"], plate_l=plate["l"],
             fz_r=fz[:, plate["r"]], fz_l=fz[:, plate["l"]],
             lowpass_hz=params.lowpass_hz,
             energy_imbalance=audit.imbalance,
             R_os_lab=R_OS_LAB, **(extra or {}))
    print(f"T={len(t)} ({t[-1]:.3f} s), plates r={plate['r']} l={plate['l']}, "
          f"body mass {body_mass:.2f} kg")
    print("boneid peak |moment| ankle/knee/hip [N m]:",
          np.round(np.abs(id_boneid[10:-10]).max(0), 1))


if __name__ == "__main__":
    main()
