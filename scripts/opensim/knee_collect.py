"""Collect knee contact (baseline + varus sweep) -> data/opensim/p1_trial13_knee_contact.{npz,json}.

Called by `knee_contact.py collect` (conda env `opensim`).  Needs
`knee_contact.py baseline`, `step ...`, optionally `moco ...`, and
`uv run python scripts/opensim/knee_kam_boneid.py` to have run.

Every [T] series is on the trial's mocap clock t = i/120 and is NaN outside
the solve window.  Contact forces are in body weights (BW = body_mass * g);
moments are in N m.
"""

from __future__ import annotations

import json
import os

import numpy as np
import opensim as osim

import knee_contact as K
import pipeline as P

G = 9.80665
CITATIONS = {  # Semantic Scholar citation counts fetched 2026-10-23
    "winby2009": {"cite": "Winby CR, Lloyd DG, Besier TF, Kirk TB (2009) Muscle and external load contribution to knee joint contact loads during normal gait. J Biomech 42(14):2294-2300",
                  "doi": "10.1016/j.jbiomech.2009.06.019", "s2_citations": 387,
                  "used_for": "two point contacts, one per condyle, medial/lateral force from moment balance about the other condyle; muscles provide frontal-plane support"},
    "lerner2015": {"cite": "Lerner ZF, DeMers MS, Delp SL, Browning RC (2015) How tibiofemoral alignment and contact locations affect predictions of medial and lateral tibiofemoral contact forces. J Biomech 48(4):644-650",
                   "doi": "10.1016/j.jbiomech.2014.12.049", "s2_citations": 228,
                   "used_for": "generic contact points +-20 mm from the knee centre (40 mm apart) on the plateau, compartment force = component normal to the plateau; sensitivity: 51 N (medial) / 30 N (lateral) per degree of alignment, 41/33 N per mm of contact location"},
    "gerus2013": {"cite": "Gerus P et al. (2013) Subject-specific knee joint geometry improves predictions of medial tibiofemoral contact forces. J Biomech 46:2778-2786",
                  "doi": "10.1016/j.jbiomech.2013.09.005", "s2_citations": 269,
                  "used_for": "generic intercondylar distance 40 mm (from Zhao et al. 2007 instrumented-knee data)"},
    "kutzner2010": {"cite": "Kutzner I et al. (2010) Loading of the knee joint during activities of daily living measured in vivo in five subjects. J Biomech 43(11):2164-2173",
                    "doi": "10.1016/j.jbiomech.2010.03.046", "s2_citations": 817,
                    "used_for": "in vivo walking peak resultant 261% BW (instrumented TKA, n=5)"},
    "kutzner2013": {"cite": "Kutzner I, Trepczynski A, Heller MO, Bergmann G (2013) Knee adduction moment and medial contact force - facts about their correlation during gait. PLoS One 8(12):e81036",
                    "doi": "10.1371/journal.pone.0081036", "s2_citations": 253,
                    "used_for": "in vivo medial peaks 176+-27% BW (1st) / 187+-44% BW (2nd); medial ratio = 9.2*KAM(%BW*Ht) + 51 % at peaks"},
    "halder2012": {"cite": "Halder A et al. (2012) Influence of limb alignment on mediolateral loading in total knee replacement. J Bone Joint Surg Am 94(11):1023-1029",
                   "doi": "10.2106/JBJS.K.00927", "s2_citations": 155,
                   "used_for": "in vivo medial share 73% (1st peak) / 65% (2nd); ~5% more medial share per degree of varus"},
    "kutzner2017": {"cite": "Kutzner I et al. (2017) Mediolateral force distribution at the knee joint shifts across activities and is driven by tibiofemoral alignment. Bone Joint J 99-B(6):779-787",
                    "doi": "10.1302/0301-620X.99B6.BJJ-2016-0713.R1", "s2_citations": 84,
                    "used_for": "medial ratio tracks HKA alignment (R^2 0.59 dynamic single-limb), up to 88% medial in varus"},
    "fregly2012": {"cite": "Fregly BJ et al. (2012) Grand challenge competition to predict in vivo knee loads. J Orthop Res 30(4):503-513",
                   "doi": "10.1002/jor.22023", "s2_citations": 622,
                   "used_for": "literature total walking peaks 1.8-3.0 BW, medial share 55-88%; models tend to over-predict total force"},
    "baggaley2024": {"cite": "Baggaley M, Haider I, Bruce O, Khassetarash A, Edwards WB (2024) Tibial strains are sensitive to speed perturbations, but not grade perturbations, during running. J Exp Biol 227(10):jeb246770",
                     "doi": "10.1242/jeb.246770", "s2_citations": 11,
                     "used_for": "Edwards lab choices: static optimization minimizing summed squared muscle STRESS (Crowninshield & Brand 1981), force bounds from force-length-velocity; contact force = muscles + ID reaction in the tibia frame; no medial/lateral split"},
    "baggaley2022": {"cite": "Baggaley M, Derrick TR, Vernillo G, Millet GY, Edwards WB (2022) Internal tibial forces and moments during graded running. J Biomech Eng 144(1):011009",
                     "doi": "10.1115/1.4051924", "s2_citations": 20,
                     "used_for": "Edwards lab: SO + static equilibrium of all applied loads for internal tibial loads"},
    "miller2014": {"cite": "Miller RH, Edwards WB, Brandon SC, Morton AM, Deluzio KJ (2014) Why don't most runners get knee osteoarthritis? Med Sci Sports Exerc 46(3):572-579",
                   "doi": "10.1249/MSS.0000000000000135", "s2_citations": 91,
                   "used_for": "modelled peak knee contact 2.72 BW walking (8.02 BW running)"},
}


def stances(fz, lo, hi, t):
    on = fz > 20.0
    edges = np.flatnonzero(np.diff(on.astype(int)))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if on[a + 1] and t[a + 1] >= lo and t[b] <= hi:
            out.append((a + 1, b))
    return out


def on_clock(t, ts, x):
    y = np.interp(t, ts, x)
    y[(t < ts[0] - 1e-6) | (t > ts[-1] + 1e-6)] = np.nan
    return y


def stance_peaks(x, st):
    p = np.array([np.nanmax(x[a:b + 1]) for a, b in st])
    return {"mean_of_stance_peaks": float(p.mean()), "sd": float(p.std()),
            "max": float(p.max()), "n_stances": len(st)}


def main():
    b = np.load(P.BONEID)
    t = b["t"]
    mass = float(b["body_mass"])
    bw = mass * G
    d_ic, f_tib = K.d_ic_scaled()
    summ_base = P.load_summary()

    base = np.load(os.path.join(K.VARUS, "baseline_wrench.npz"))
    ts = base["t"]
    win_so = (ts[0], ts[-1])
    tm0, tm1 = ts[base["dyn_inside"]][[0, -1]]
    st = stances(b["fz_r"], max(win_so[0], tm0), min(win_so[1], tm1), t)
    out = {"t": t, "body_mass": mass, "d_ic": d_ic,
           "stance_frames": np.array(st)}
    side = {"baseline": {}, "sweep": {}}
    for name in ("so", "dyn"):
        sp = K.split(base[name + "_F"], base[name + "_M"], d_ic)
        keep = np.ones_like(ts, bool) if name == "so" else base["dyn_inside"]
        for k in ("total", "medial", "lateral", "lig_lateral", "lig_medial",
                  "medial_raw", "lateral_raw"):
            x = np.where(keep, sp[k], np.nan) / bw
            out[f"{name}_contact_{k}" if k in ("total", "medial", "lateral")
                else f"{name}_{k}"] = on_clock(t, ts, x)
        out[f"{name}_frontal_moment"] = on_clock(t, ts, np.where(keep, sp["frontal_moment"], np.nan))
        tot = out[f"{name}_contact_total"]
        med = out[f"{name}_contact_medial_raw" if False else f"{name}_medial_raw"]
        share = np.where(tot > 0.5, med / tot, np.nan)
        out[f"{name}_medial_share"] = share
        inst = [a + np.nanargmax(tot[a:b + 1]) for a, b in st]
        side["baseline"][name] = {
            "peak_total_BW": stance_peaks(tot, st),
            "peak_medial_BW": stance_peaks(out[f"{name}_contact_medial"], st),
            "peak_lateral_BW": stance_peaks(out[f"{name}_contact_lateral"], st),
            "medial_share_at_total_peak": float(np.nanmean(share[inst])),
            "medial_share_mean_stance": float(np.nanmean(np.concatenate(
                [share[a:b + 1] for a, b in st]))),
            "lateral_liftoff_frames_in_stance": int(sum(
                np.sum(out[f"{name}_lig_lateral"][a:b + 1] > 0) for a, b in st)),
            "stance_frames": int(sum(b_ - a + 1 for a, b_ in st)),
            "peak_lateral_ligament_tension_BW_in_stance": float(max(
                np.nanmax(out[f"{name}_lig_lateral"][a:b + 1]) for a, b in st)),
        }
    out["ext_frontal_moment"] = on_clock(t, ts, -base["ext_M"][:, 0])

    # ---------------- sweep
    kb = np.load(os.path.join(K.VARUS, "kam_boneid.npz"))
    D = len(K.D_MM)
    keys = ["so_total", "so_medial", "so_lateral", "so_lig_lateral",
            "frontal_moment_os", "contact_frontal_moment_so",
            "knee_sagittal_os"]
    sw = {k: np.full((D, len(t)), np.nan) for k in keys}
    theta, ikver = [], []
    off0 = None
    for i, dmm in enumerate(K.D_MM):
        step = os.path.join(K.VARUS, f"d{dmm:02d}")
        z = np.load(os.path.join(step, "wrench.npz"))
        info = json.load(open(os.path.join(step, "info.json")))
        theta.append(info["theta_deg"])
        tz = z["t"]
        sp = K.split(z["so_F"], z["so_M"], d_ic)
        sw["so_total"][i] = on_clock(t, tz, sp["total"] / bw)
        sw["so_medial"][i] = on_clock(t, tz, sp["medial"] / bw)
        sw["so_lateral"][i] = on_clock(t, tz, sp["lateral"] / bw)
        sw["so_lig_lateral"][i] = on_clock(t, tz, sp["lig_lateral"] / bw)
        sw["frontal_moment_os"][i] = on_clock(t, tz, -z["ext_M"][:, 0])
        sw["contact_frontal_moment_so"][i] = on_clock(t, tz, -sp["frontal_moment"])
        tid, lab, ID = K.read_sto(os.path.join(step, "id.sto"))
        sw["knee_sagittal_os"][i] = np.interp(t, tid, ID[:, lab.index("knee_angle_r_moment")])
        geo = {k[4:]: z[k] for k in z.files if k.startswith("geo_")}
        off = K.frontal_offset(geo)
        if off0 is None:
            off0 = off
        inst = np.concatenate([np.arange(a, b + 1) for a, b in st])
        dk = on_clock(t, tz, off - off0)
        ikver.append({"d_mm": dmm, "theta_deg": info["theta_deg"],
                      "model_knee_lateral_shift_mean_mm": float(np.nanmean(dk) * 1000),
                      "model_knee_lateral_shift_sd_mm": float(np.nanstd(dk) * 1000),
                      "model_knee_lateral_shift_stance_mean_mm": float(np.nanmean(dk[inst]) * 1000),
                      "marker_knee_lateral_shift_mean_mm": info["marker_knee_lateral_shift_mean_m"] * 1000,
                      "ik_rms_mean_mm": info["ik"]["rms_mean_m"] * 1000,
                      "ik_max_marker_error_max_mm": info["ik"]["max_marker_error_max_m"] * 1000,
                      "default_pose_ankle_shift_m": info["default_pose_ankle_shift_m"],
                      "runtime_s": info["runtime_s"], "so_runtime_s": info["so_runtime_s"]})
    for k, v in sw.items():
        out[f"sweep_{k}"] = v
    out["sweep_d_m"] = np.array(K.D_MM) / 1000.0
    out["sweep_theta_deg"] = np.array(theta)
    out["sweep_kam_boneid"] = kb["kam"]
    out["sweep_knee_sagittal_boneid"] = kb["knee_sagittal"]
    out["sweep_model_knee_shift_mm"] = np.array([v["model_knee_lateral_shift_mean_mm"] for v in ikver])

    dyn_d, dyn = [], {k: [] for k in ("total", "medial", "lateral", "lig_lateral")}
    moco_info = {}
    for dmm in K.D_MM:
        f = os.path.join(K.VARUS, f"d{dmm:02d}", "wrench_dyn.npz")
        if dmm == 0 and not os.path.exists(f):
            # d = 0 is the baseline MocoInverse (same model, same IK: the
            # d = 0 sweep step reproduces the baseline SO peaks exactly)
            sp = K.split(base["dyn_F"], base["dyn_M"], d_ic)
            dyn_d.append(0.0)
            for k in dyn:
                dyn[k].append(on_clock(t, ts, np.where(base["dyn_inside"], sp[k], np.nan) / bw))
            moco_info[0] = {"runtime_s": summ_base.get("moco", {}).get("runtime_s"),
                            "note": "baseline MocoInverse (data/opensim/p1_trial13_mocoinverse_*.sto)"}
            continue
        if not os.path.exists(f):
            continue
        z = np.load(f)
        sp = K.split(z["dyn_F"], z["dyn_M"], d_ic)
        inside = (z["t"] >= z["window"][0] - 1e-9) & (z["t"] <= z["window"][1] + 1e-9)
        dyn_d.append(dmm / 1000)
        for k in dyn:
            dyn[k].append(on_clock(t, z["t"], np.where(inside, sp[k], np.nan) / bw))
        moco_info[dmm] = json.load(open(os.path.join(K.VARUS, f"d{dmm:02d}", "info_moco.json")))
    out["sweep_dyn_d_m"] = np.array(dyn_d)
    for k, v in dyn.items():
        out[f"sweep_dyn_{k}"] = np.array(v) if v else np.zeros((0, len(t)))

    # sweep summary table
    rows = []
    for i, dmm in enumerate(K.D_MM):
        r = {"d_mm": dmm, "theta_deg": theta[i]}
        for k in ("so_total", "so_medial", "so_lateral"):
            r[f"peak_{k}_BW"] = stance_peaks(sw[k][i], st)["mean_of_stance_peaks"]
        lig = sw["so_lig_lateral"][i]
        r["lateral_liftoff_stance_frames"] = int(sum(np.sum(lig[a:b + 1] > 0) for a, b in st))
        r["peak_lateral_ligament_tension_BW"] = float(max(np.nanmax(lig[a:b + 1]) for a, b in st))
        r["peak_kam_opensim_Nm"] = stance_peaks(sw["frontal_moment_os"][i], st)["mean_of_stance_peaks"]
        r["peak_kam_boneid_Nm"] = stance_peaks(kb["kam"][i], st)["mean_of_stance_peaks"]
        r["peak_knee_extension_moment_opensim_Nm"] = stance_peaks(-sw["knee_sagittal_os"][i], st)["mean_of_stance_peaks"]
        r["peak_knee_extension_moment_boneid_Nm"] = stance_peaks(-kb["knee_sagittal"][i], st)["mean_of_stance_peaks"]
        if dmm / 1000 in dyn_d:
            j = dyn_d.index(dmm / 1000)
            for k in ("total", "medial", "lateral"):
                r[f"peak_dyn_{k}_BW"] = stance_peaks(dyn[k][j], st)["mean_of_stance_peaks"]
        rows.append(r)
    side["sweep"]["table"] = rows
    side["sweep"]["ik_verification"] = ikver
    side["sweep"]["moco"] = moco_info
    np.savez(os.path.join(P.OUT, "p1_trial13_knee_contact.npz"), **out)

    side.update({
        "body_mass_kg": mass, "BW_N": bw, "d_ic_m": d_ic,
        "d_ic_note": f"40 mm generic (Lerner 2015 generic +-20 mm; Gerus 2013 40 mm) x tibia_r uniform scale factor {f_tib:.4f}",
        "stances": {"definition": "right Fz > 20 N, complete stances inside both solve windows",
                    "frames": [list(map(int, s)) for s in st],
                    "times_s": [[float(t[a]), float(t[b_])] for a, b_ in st]},
        "peak_definition": "mean over the complete right stances of each stance's maximum (sd and max also given for the baseline)",
        "assumptions": [
            "Knee reaction = Newton-Euler of the subtree below the knee (tibia, talus, calcn, toes): IK accelerations (12 Hz low-pass, quintic spline), gravity, measured GRF + free moment, muscle path forces (tension along every path segment touching a subtree body, wrap points included). No JointReaction analysis (that realizes forward dynamics).",
            "Reaction taken about the walker knee's tibial child-frame origin and expressed in the tibia_r body frame: compressive force = -F_y (along the tibial Y axis, which is within ~1.4 deg of the knee-ankle line), AP shear F_x, frontal (varus/valgus) moment M_x.",
            "Medial/lateral split: two point contacts on a flat plateau perpendicular to tibial Y, at z = -/+ d_ic/2 through the knee centre. F_med + F_lat = Fc; (F_med - F_lat) d_ic/2 = -M_x (varus loads medial). Ligaments carry nothing; a negative condyle is reported as lift-off: contact clamped to 0, the tension a collateral ligament at that condyle would need is reported separately (= -raw), the other condyle keeps its raw value.",
            "Sagittal moment along the knee DOF is not used by the contact model (it is the muscles' job); the knee reserve stays below 0.05 N m in SO.",
            "Muscle forces: SO = OpenSim StaticOptimization (sum of squared activations, DeGrooteFregly rigid tendon, force-length-velocity); dyn = MocoInverse tendon forces. Edwards-lab SO minimizes summed squared muscle STRESS instead; with fixed fibre lengths/velocities that is activation squared weighted by 1/(fl*fv)^2 rather than a different structure, so I kept the existing solves.",
            "external knee adduction moment (OpenSim) = -M_x of the reaction WITHOUT muscle forces (inertia + gravity + GRF), + = varus.",
            "Varus sweep: right thigh markers (RGTR, RTH1-3, RLEP, RMEP) rigidly rotated about the hip centre and shank markers (RSH1-3, RLML, RMML) about the mid-malleoli so the mid-epicondyle point moves laterally by d, perpendicular to the hip-ankle line (lateral = epicondyle axis); hip, foot markers and GRF untouched. Model: walker_knee_r child (tibia) offset frame rotated by theta = d(1/L_thigh + 1/L_shank) about tibial X through the knee centre (L from the data, 0.412/0.394 m), i.e. a fixed varus of the tibia on the femur. Then a single IK pass with the final placed markers and the baseline weights, ID, SO.",
        ],
        "citations": CITATIONS,
        "in_vivo_reference": {"total_walking_peak_BW": 2.61, "source": "Kutzner 2010 (261% BW)",
                              "medial_peaks_BW": [1.76, 1.87], "medial_share": [0.73, 0.65],
                              "medial_share_source": "Halder 2012 (1st/2nd peak); Kutzner 2013 medial peaks",
                              "modelled_range_BW": [1.8, 3.0], "modelled_range_source": "Fregly 2012 review"},
        "runtimes_s": {"baseline_so_moco_solves": {"so": summ_base.get("so", {}).get("runtime_s"),
                                                    "moco": summ_base.get("moco", {}).get("runtime_s")},
                       "sweep_steps": {v["d_mm"]: v["runtime_s"] for v in ikver},
                       "sweep_moco": {k: v["runtime_s"] for k, v in moco_info.items()}},
        "units": "contact *_BW arrays in body weights; moments in N m; sagittal moments in OpenSim sign (+ = knee flexion moment on the shank, so extension is negative)",
    })
    with open(os.path.join(P.OUT, "p1_trial13_knee_contact.json"), "w") as f:
        json.dump(side, f, indent=2, default=float)
    print(json.dumps(side["baseline"], indent=1))
    for r in rows:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    for v in ikver:
        print(v["d_mm"], round(v["model_knee_lateral_shift_mean_mm"], 2), round(v["model_knee_lateral_shift_sd_mm"], 2), round(v["ik_rms_mean_mm"], 2))


if __name__ == "__main__":
    os.chdir(P.OUT)
    main()
