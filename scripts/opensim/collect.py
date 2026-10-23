"""Collect SO + MocoInverse + ID into data/opensim/p1_trial13_muscles.{npz,json}.

Called by `pipeline.py collect` (conda env `opensim`, cwd = data/opensim).
Every series is put on the trial's mocap clock t = i/120 (the boneid
`V3dTrial.t`) by linear interpolation; samples outside a solve's window are
NaN.
"""

from __future__ import annotations

import json
import os

import numpy as np
import opensim as osim

import pipeline as P

JOINTS = ["ankle", "knee", "hip"]
COORDS = ["ankle_angle_r", "knee_angle_r", "hip_flexion_r"]
CROSS_TOL = 2e-3          # m: |moment arm| above this somewhere => crosses


def read_sto(path):
    tbl = osim.TimeSeriesTable(path)
    return (np.array(tbl.getIndependentColumn()),
            list(tbl.getColumnLabels()), tbl.getMatrix().to_numpy())


def interp_cols(t, tsrc, data, inside_only=True):
    out = np.stack([np.interp(t, tsrc, data[:, j])
                    for j in range(data.shape[1])], axis=1)
    if inside_only:
        out[(t < tsrc[0] - 1e-6) | (t > tsrc[-1] + 1e-6)] = np.nan
    return out


def moment_arms(t, muscles):
    """[T,M,3] moment arms on the processed model at low-passed IK poses."""
    model = osim.Model(P.PROCESSED)
    state = model.initSystem()
    tp = osim.TableProcessor(P.IK)
    tp.append(osim.TabOpLowPassFilter(P.LOWPASS_HZ))
    tbl = tp.process(model)                   # radians after processing
    tk = np.array(tbl.getIndependentColumn())
    labels = list(tbl.getColumnLabels())
    q = interp_cols(t, tk, tbl.getMatrix().to_numpy(), inside_only=False)
    cs = model.getCoordinateSet()
    # TableProcessor keeps the IK file's degrees; convert the Rotational
    # coordinates (the Coupled knee_angle_*_beta is already in radians).
    for j, lab in enumerate(labels):
        if cs.contains(lab) and cs.get(lab).getMotionType() == 1:
            q[:, j] = np.deg2rad(q[:, j])
    cols = [(cs.get(lab), j) for j, lab in enumerate(labels)
            if cs.contains(lab)]
    coords = [cs.get(c) for c in COORDS]
    mus = [model.getMuscles().get(m) for m in muscles]
    r = np.zeros((len(t), len(muscles), 3))
    for i in range(len(t)):
        for c, j in cols:
            c.setValue(state, float(q[i, j]), False)
        model.realizePosition(state)
        for m, mu in enumerate(mus):
            for k, c in enumerate(coords):
                r[i, m, k] = mu.computeMomentArm(state, c)
    return r, q, labels


def main():
    b = np.load(P.BONEID)
    t = b["t"]
    summ = P.load_summary()

    # ---- right-leg muscles
    model = osim.Model(P.PROCESSED)
    model.initSystem()
    allm = [model.getMuscles().get(i).getName()
            for i in range(model.getMuscles().getSize())]
    muscles = [m for m in allm if m.endswith("_r")]
    M = len(muscles)

    # ---- static optimization
    tso, lab, F = read_sto("so/p1_trial13_so_StaticOptimization_force.sto")
    _, laba, A = read_sto("so/p1_trial13_so_StaticOptimization_activation.sto")
    so_force_all = interp_cols(t, tso, F)
    so_act_all = interp_cols(t, tso, A)
    so_force = so_force_all[:, [lab.index(m) for m in muscles]]
    so_act = so_act_all[:, [laba.index(m) for m in muscles]]

    # ---- MocoInverse
    tm, labo, O = read_sto("p1_trial13_mocoinverse_outputs.sto")
    dyn_force = interp_cols(t, tm, O)[:, [labo.index(f"/forceset/{m}|tendon_force")
                                         for m in muscles]]
    dyn_act = interp_cols(t, tm, O)[:, [labo.index(f"/forceset/{m}|activation")
                                       for m in muscles]]
    sol = osim.MocoTrajectory("p1_trial13_mocoinverse_solution.sto")
    tc = np.asarray(sol.getTimeMat())
    cnames = list(sol.getControlNames())
    C = np.asarray(sol.getControlsTrajectoryMat())

    # ---- moment arms, groups
    r, q, qlab = moment_arms(t, muscles)
    crosses = np.abs(r).max(axis=0) > CROSS_TOL                    # [M,3]
    mean_abs = np.abs(r).mean(axis=0) * crosses
    group = np.argmax(mean_abs, axis=1).astype(int)

    so_moment = np.einsum("tm,tmk->tk", so_force, r)
    dyn_moment = np.einsum("tm,tmk->tk", dyn_force, r)

    # ---- ID: OpenSim and boneid
    tid, labid, ID = read_sto(P.ID)
    id_os = interp_cols(t, tid, ID, inside_only=False)[
        :, [labid.index(c + "_moment") for c in COORDS]]
    id_b = b["id_moment_boneid"]

    window = np.array([tm[0], tm[-1]])
    np.savez(
        os.path.join(P.OUT, "p1_trial13_muscles.npz"),
        t=t, muscle_names=np.array(muscles),
        so_force=so_force, dyn_force=dyn_force,
        so_activation=so_act, dyn_activation=dyn_act,
        joint_names=np.array(JOINTS), moment_arm=r, group=group,
        crosses=crosses, so_moment=so_moment, dyn_moment=dyn_moment,
        id_moment_opensim=id_os, id_moment_boneid=id_b, window=window)

    # ---- metrics for the sidecar
    def stats(a, ref, mask):
        a, ref = a[mask], ref[mask]
        return {"rmse_Nm": float(np.sqrt(np.mean((a - ref) ** 2))),
                "corr": float(np.corrcoef(a, ref)[0, 1]),
                "peak_abs_Nm": float(np.abs(a).max()),
                "ref_peak_abs_Nm": float(np.abs(ref).max())}

    inside = (t >= window[0]) & (t <= window[1])
    inside_so = np.isfinite(so_force[:, 0])
    edge = np.zeros_like(t, bool)
    edge[12:-12] = True
    both = inside & inside_so
    m_id = {j: stats(id_b[:, k], id_os[:, k], edge)
            for k, j in enumerate(JOINTS)}
    m_so = {j: stats(so_moment[:, k], id_os[:, k], inside_so)
            for k, j in enumerate(JOINTS)}
    m_dyn = {j: stats(dyn_moment[:, k], id_os[:, k], inside)
             for k, j in enumerate(JOINTS)}
    peaks = {m: {"so_peak_N": float(np.nanmax(so_force[:, i])),
                 "dyn_peak_N": float(np.nanmax(dyn_force[:, i])),
                 "so_vs_dyn_corr": float(np.corrcoef(
                     so_force[both, i], dyn_force[both, i])[0, 1])
                 if np.std(so_force[both, i]) > 0 and
                 np.std(dyn_force[both, i]) > 0 else None,
                 "group": JOINTS[group[i]],
                 "crosses": [JOINTS[k] for k in range(3) if crosses[i, k]]}
             for i, m in enumerate(muscles)}

    def reserve_usage(names, values, scale):
        out = {}
        for n in names:
            v = values[:, cnames.index(n)] * scale.get(n, 1.0)
            out[n] = {"max_abs": float(np.abs(v).max()),
                      "rms": float(np.sqrt(np.mean(v ** 2)))}
        return out

    so_res = [l for l in lab if "reserve" in l or "residual" in l]
    so_reserve = {n: {"max_abs": float(np.abs(F[:, lab.index(n)]).max()),
                      "rms": float(np.sqrt(np.mean(F[:, lab.index(n)] ** 2)))}
                  for n in so_res}
    opt = {}
    for i in range(model.getForceSet().getSize()):
        f = model.getForceSet().get(i)
        ca = osim.CoordinateActuator.safeDownCast(f)
        if ca is not None:
            opt["/forceset/" + f.getName()] = ca.getOptimalForce()
    dyn_res_names = [n for n in cnames if "reserve" in n or "residual" in n]
    dyn_reserve = reserve_usage(dyn_res_names, C, opt)

    # IK summary in the requested shape
    side = {
        "model": "RajagopalLaiUhlrich2023 (opensim-models master; Rajagopal "
                 "2016 + Lai 2017 + Uhlrich 2022), 80 Millard muscles in the "
                 "scaled model; for SO + MocoInverse muscles replaced by "
                 "DeGrooteFregly2016 (rigid tendon, passive fibre force "
                 "ignored, active F-L width x1.5), subtalar/mtp/wrist welded",
        "opensim_version": osim.__version__,
        "env": "conda env 'opensim' (/Users/jeremy/.julia/conda/3/aarch64/"
               "envs/opensim), opensim-org::opensim 4.6 py311 osx-arm64",
        "trial": "P1 trial slot 13 (data/p1_trial13_sample.mat, index 0), "
                 "walking, 637 frames @120 Hz",
        "body_mass_kg": float(b["body_mass"]),
        "scaling": summ.get("scale"),
        "ik": summ.get("ik"),
        "id": {**summ.get("id", {}),
               "boneid_vs_opensim_sagittal": m_id,
               "comparison_frames": "all but 12 frames at each end"},
        "static_optimization": {**summ.get("so", {}),
                                "reserve_and_residual_usage_N_or_Nm": so_reserve,
                                "muscle_moment_vs_opensim_id": m_so},
        "dynamic_optimization": {**summ.get("moco", {}),
                                 "reserve_and_residual_usage_N_or_Nm":
                                     dyn_reserve,
                                 "muscle_moment_vs_opensim_id": m_dyn},
        "muscles": peaks,
        "conventions": {
            "lab_to_opensim_rotation": b["R_os_lab"].tolist(),
            "lab_to_opensim_text": "X_os = -y_lab (anterior), Y_os = z_lab "
                                   "(up), Z_os = -x_lab (subject right); no "
                                   "translation",
            "cop": "COP slid along the force line from the plate-origin "
                   "height z=-0.1746 m to lab z=0 (wrench unchanged)",
            "moment_sign": "OpenSim generalized-force convention: + = "
                           "ankle dorsiflexion, knee FLEXION (Rajagopal "
                           "walker knee), hip flexion moment on the child "
                           "segment",
            "id_moment_boneid": "boneid whole_body_run(trial, p1_params()) "
                                "right-leg joint_torque_local[:, j, 1] "
                                "(torque on distal by proximal about the "
                                "proximal frame's subject-LEFT axis) times "
                                "[-1, +1, -1] for [ankle, knee, hip]",
            "moment_arm": "OpenSim Muscle.computeMomentArm (= -dL/dq), m; "
                          "so/dyn_moment = sum_m force_m * moment_arm_m",
            "group": "argmax over crossed joints of mean |moment arm|",
            "crosses": f"max |moment arm| > {CROSS_TOL} m",
            "time": "t = i/120 s, the trial's mocap clock; SO and dyn "
                    "series are NaN outside their windows",
        },
        "so_window_s": [float(tso[0]), float(tso[-1])],
        "dyn_window_s": window.tolist(),
        "notes": [
            "No static trial in the export: segment scale factors are "
            "trial-mean per-frame marker distances over the generic model's; "
            "markers were placed by iterating IK + trial-mean body-frame "
            "placement (see ik.procedure).",
            "Visual3D's HJC landmarks are 20% wider apart than the scaled "
            "Rajagopal hips while ASIS width matches within 3%, so pelvis "
            "markers carry ~13-17 mm RMS of systematic error.",
            "Pelvis residual forces/moments are what ID needs to close "
            "without RRA; reserves on leg joints are optimal force 1 N m "
            "(penalised like a muscle at control 1), pelvis residuals "
            "optimal force 100.",
            "SO and MocoInverse use the identical processed model, external "
            "loads and 12 Hz-filtered IK, so differences come from "
            "activation dynamics + the excitation cost only.",
        ],
    }
    with open(os.path.join(P.OUT, "p1_trial13_muscles.json"), "w") as f:
        json.dump(side, f, indent=2, default=float)
    print(json.dumps({"id": m_id, "so": m_so, "dyn": m_dyn}, indent=1))
    big = sorted(peaks, key=lambda m: -peaks[m]["so_peak_N"])[:12]
    for m in big:
        print(m, {k: (round(v, 2) if isinstance(v, float) else v)
                  for k, v in peaks[m].items()})
    print("dyn reserves:", {k: round(v["max_abs"], 1)
                            for k, v in dyn_reserve.items()})


if __name__ == "__main__":
    os.chdir(P.OUT)
    main()
