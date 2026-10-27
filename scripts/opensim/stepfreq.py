"""Knee contact for P1's nine fixed-speed (1.25 m/s) step-frequency trials.

Run in the conda env `opensim` (like pipeline.py), after the uv-env export:

    MAT=/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat
    for s in 16 17 18 19 20 21 22 23 24; do
      uv run python scripts/opensim/export_inputs.py --mat $MAT --slot $s \\
          --tag p1_slot$s --out data/opensim/stepfreq/work
    done
    PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
    $PY scripts/opensim/stepfreq.py run 19          # one slot per process; run them in parallel
    # synthetic cadences from slot 19 (simulate.step_frequency_trial), d in steps/min:
    for d in -10 -8 -6 -4 -2 0 2 4 6 8 10; do
      uv run python scripts/opensim/export_inputs.py --mat $MAT --slot 19 \
          --syn-d-spm $d --tag $(printf "p1_syn%+d" $d) --out data/opensim/stepfreq/work
    done
    $PY scripts/opensim/stepfreq.py run syn-10 syn+0 ...   # same pipeline
    $PY scripts/opensim/stepfreq.py collect         # -> data/opensim/p1_stepfreq_knee_contact.{npz,json}

Per slot (intermediates in data/opensim/stepfreq/slotNN/): ONE IK pass on the
already scaled + marker-placed trial-13 model (data/opensim/p1_scaled.osim,
same subject and markerset, so no rescaling or re-placement), with the final
trial-13 IK weights; then ID, SO and the knee contact model, all through the
SAME functions as trial 13 (pipeline.stage_id / stage_so, knee_contact.
knee_wrench / split) with the module paths re-pointed at this trial.  The SO
window is (0.10 s, T_end - 0.10 s), the same 0.10 s edge trim as trial 13.

van der Zee et al. 2022 Table 2 (fixed speed 1.25 m/s, step frequency
f* = f / f_preferred): slots 16/17/18 = 0.7/0.8/0.9, 19/20/21 = preferred
(three repeats), 22/23/24 = 1.1/1.2/1.3.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knee_collect as KC  # noqa: E402
import knee_contact as K  # noqa: E402
import pipeline as P  # noqa: E402

G = 9.80665
SF = os.path.join(P.OUT, "stepfreq")
SF_WORK = os.path.join(SF, "work")
SLOTS = [16, 17, 18, 19, 20, 21, 22, 23, 24]
SYN_D = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]   # steps/min added to slot 19's f0
F_STAR = {16: 0.7, 17: 0.8, 18: 0.9, 19: 1.0, 20: 1.0, 21: 1.0,
          22: 1.1, 23: 1.2, 24: 1.3}
OUT_NPZ = os.path.join(P.OUT, "p1_stepfreq_knee_contact.npz")
OUT_JSON = os.path.join(P.OUT, "p1_stepfreq_knee_contact.json")
EDGE = 0.10           # s trimmed off both ends for SO (trial 13: 0.10 / 5.20 of 5.30)


def log(msg):
    print(f"[stepfreq] {msg}", flush=True)


def paths(slot):
    """slot: int (measured trial) or 'syn<+d>' (synthetic cadence of slot 19)."""
    if isinstance(slot, str) and slot.startswith("syn"):
        tag, sub = f"p1_{slot}", slot
    else:
        tag, sub = f"p1_slot{int(slot)}", f"slot{int(slot):02d}"
    return (tag, os.path.join(SF, sub),
            os.path.join(SF_WORK, f"{tag}_markers.trc"),
            os.path.join(SF_WORK, f"{tag}_grf.mot"),
            os.path.join(SF_WORK, f"{tag}_boneid.npz"))


def reserve_usage(sto, window_frames=None):
    """Max |value| of the right-leg reserves and the pelvis residuals (SO)."""
    ts, lab, F = K.read_sto(sto)
    out = {}
    for j, n in enumerate(lab):
        if n.startswith("residual_") or (n.startswith("reserve_") and n.endswith("_r")):
            out[n] = float(np.abs(F[:, j]).max())
    return out


def stage_run(slot):
    tag, step, trc, grf, bnz = paths(slot)
    os.makedirs(step, exist_ok=True)
    os.chdir(step)
    t = np.load(bnz)["t"]
    t_end = float(t[-1])
    P.save_summary = lambda _d: None          # trial-13 summary stays untouched
    P.load_summary = lambda: {}
    P.TRC, P.GRF = trc, grf
    P.SCALED = os.path.join(P.OUT, "p1_scaled.osim")
    P.IK, P.ID = "ik.mot", "id.sto"
    P.PROCESSED, P.EXT_LOADS = "processed.osim", "external_loads.xml"
    P.T_END = t_end
    P.TAG = tag.replace("+", "p")             # OpenSim names reject "+"
    P.WINDOW = (EDGE, round(t_end - EDGE, 6))
    K.GRF_ABS = grf
    info = {"slot": slot, "tag": tag, "f_star": F_STAR.get(slot),
            "t_end_s": t_end, "so_window_s": list(P.WINDOW),
            "model": "data/opensim/p1_scaled.osim (trial-13 scale + marker placement, reused)"}
    t0 = time.time()
    w2 = {n: P.WEIGHTS.get(n, 1.0) for n in P.marker_table()}
    P.run_ik(P.SCALED, P.IK, "setup_ik.xml", w2)
    info["ik"] = P.ik_errors("ik_ik_marker_errors.sto")
    info["ik"]["per_marker_rms_m"] = P.per_marker_rms(P.SCALED, P.IK)
    info["ik"]["runtime_s"] = time.time() - t0
    log(f"slot {slot}: IK mean RMS {info['ik']['rms_mean_m']*1000:.1f} mm, "
        f"worst max {info['ik']['max_marker_error_max_m']*1000:.1f} mm")
    t1 = time.time()
    P.stage_id()
    info["id_runtime_s"] = time.time() - t1
    t1 = time.time()
    P.stage_so()
    info["so_runtime_s"] = time.time() - t1
    sto = f"so/{P.TAG}_so_StaticOptimization_force.sto"
    info["so_reserve_max_abs"] = reserve_usage(sto)
    t1 = time.time()
    muscles = K.right_muscles(P.PROCESSED)
    tso = K.read_sto(sto)[0]
    so, _ = K.so_forces(sto, tso, muscles)
    W, geo = K.knee_wrench(P.PROCESSED, P.IK, tso, {"so": so})
    info["wrench_runtime_s"] = time.time() - t1
    info["runtime_s"] = time.time() - t0
    np.savez("wrench.npz", t=tso,
             **{f"{k}_F": v[0] for k, v in W.items()},
             **{f"{k}_M": v[1] for k, v in W.items()},
             **{f"geo_{k}": v for k, v in geo.items()})
    with open("info.json", "w") as f:
        json.dump(info, f, indent=2)
    for x in glob.glob("so/*controls.xml"):
        os.remove(x)
    log(f"slot {slot}: done in {info['runtime_s']:.0f} s "
        f"(SO {info['so_runtime_s']:.0f} s)")


MIN_STANCE_S = 0.20   # drop sub-0.2 s / sub-0.3 BW "stances" (slot 17 has a
MIN_PEAK_BW = 0.30    # 20 N blip at t = 7.48 s that is not a foot strike)


def real_stances(fz, t, bw, thresh=20.0):
    """All right stances (on, off) frames with Fz > thresh, blips dropped."""
    on = fz > thresh
    up = np.flatnonzero(~on[:-1] & on[1:]) + 1
    dn = np.flatnonzero(on[:-1] & ~on[1:])
    out = []
    for a in up:
        b = dn[dn >= a]
        b = int(b[0]) if len(b) else len(fz) - 1
        if (t[b] - t[a] >= MIN_STANCE_S and fz[a:b + 1].max() >= MIN_PEAK_BW * bw):
            out.append((int(a), b, b < len(fz) - 1))
    return out


def heel_strikes(fz, t, bw):
    return np.array([a for a, _, _ in real_stances(fz, t, bw)])


def window_stances(fz, t, bw, lo, hi):
    """Complete stances inside [lo, hi] (same rule as knee_collect.stances,
    plus the blip filter)."""
    return [(a, b) for a, b, ended in real_stances(fz, t, bw)
            if ended and a > 0 and t[a] >= lo and t[b] <= hi]


def stance_metrics(x, st, dt):
    peaks = np.array([np.nanmax(x[a:b + 1]) for a, b in st])
    imp = np.array([np.trapezoid(x[a:b + 1], dx=dt) for a, b in st])
    return peaks, imp


def stage_collect():
    os.chdir(P.OUT)
    d_ic, f_tib = K.d_ic_scaled()
    mass = float(np.load(P.BONEID)["body_mass"])     # trial 13, 85.16 kg
    bw = mass * G
    base = json.load(open(os.path.join(P.OUT, "p1_trial13_knee_contact.json")))
    out = {"slots": np.array(SLOTS), "f_star": np.array([F_STAR[s] for s in SLOTS]),
           "body_mass": mass, "d_ic": d_ic}
    side = {"trials": {}}
    stride_s, sf_hz = [], []
    syn_keys = [f"syn{d:+d}" for d in SYN_D]
    syn_stride, syn_k = [], []
    for slot in SLOTS + syn_keys:
        tag, step, trc, grf, bnz = paths(slot)
        if not os.path.exists(os.path.join(step, "wrench.npz")):
            sys.exit(f"missing {step}/wrench.npz: run `stepfreq.py run {slot}` first")
        b = np.load(bnz)
        t, fz = b["t"], b["fz_r"]
        dt = float(t[1] - t[0])
        z = np.load(os.path.join(step, "wrench.npz"))
        info = json.load(open(os.path.join(step, "info.json")))
        ts = z["t"]
        sp = K.split(z["so_F"], z["so_M"], d_ic)
        st = window_stances(fz, t, bw, ts[0], ts[-1])
        hs = heel_strikes(fz, t, bw)
        stride = float(np.mean(np.diff(t[hs])))
        syn = isinstance(slot, str)
        if syn:
            syn_stride.append(stride)
            syn_k.append(float(b["syn_k"]))
        else:
            stride_s.append(stride)
            sf_hz.append(2.0 / stride)
        p = f"{slot}_" if syn else f"s{slot}_"
        out[p + "t"] = t
        for k in ("total", "medial", "lateral", "lig_lateral"):
            out[p + k] = KC.on_clock(t, ts, sp[k] / bw)
        out[p + "fz_r"] = fz
        out[p + "stance_frames"] = np.array(st, dtype=int)
        model = osim.Model(os.path.join(step, "processed.osim"))
        model.initSystem()
        spl = K.coordinate_splines(model, os.path.join(step, "ik.mot"))
        out[p + "knee_flexion_deg"] = np.degrees(spl["knee_angle_r"](t))
        out[p + "knee_adduction_moment"] = KC.on_clock(t, ts, -z["ext_M"][:, 0])
        tid, lab, ID = K.read_sto(os.path.join(step, "id.sto"))
        out[p + "knee_sagittal_moment"] = np.interp(
            t, tid, ID[:, lab.index("knee_angle_r_moment")])
        row = {"f_star": None if syn else F_STAR[slot], "stride_s": stride,
               **({"synthetic_from_slot": int(b["syn_source_slot"]),
                   "syn_k": float(b["syn_k"]), "syn_d_spm": float(b["syn_d_spm"]),
                   "syn_f0_spm": float(b["syn_f0_spm"])} if syn else {}),
               "step_freq_hz": 2.0 / stride, "steps_per_min": 120.0 / stride,
               "n_frames": len(t), "t_end_s": float(t[-1]),
               "n_complete_stances_in_window": len(st),
               "stance_frames": [list(map(int, s)) for s in st],
               "stance_s_mean": float(np.mean([(b_ - a + 1) * dt for a, b_ in st])),
               "ik_rms_mean_mm": info["ik"]["rms_mean_m"] * 1000,
               "ik_rms_max_mm": info["ik"]["rms_max_m"] * 1000,
               "ik_max_marker_error_max_mm": info["ik"]["max_marker_error_max_m"] * 1000,
               "ik_per_marker_rms_mm": {k: v * 1000 for k, v in
                                        info["ik"]["per_marker_rms_m"].items()},
               "so_window_s": info["so_window_s"],
               "so_reserve_max_abs": info["so_reserve_max_abs"],
               "runtimes_s": {"total": info["runtime_s"], "ik": info["ik"]["runtime_s"],
                              "id": info["id_runtime_s"], "so": info["so_runtime_s"],
                              "contact": info["wrench_runtime_s"]}}
        for k in ("total", "medial", "lateral", "lig_lateral"):
            pk, imp = stance_metrics(out[p + k], st, dt)
            row[f"peak_{k}_BW_mean"] = float(pk.mean())
            row[f"peak_{k}_BW_sd"] = float(pk.std())
            row[f"impulse_{k}_BWs_mean"] = float(imp.mean())
            row[f"impulse_{k}_BWs_sd"] = float(imp.std())
        pk, _ = stance_metrics(out[p + "knee_adduction_moment"], st, dt)
        row["peak_knee_adduction_moment_Nm_mean"] = float(pk.mean())
        pk, _ = stance_metrics(-out[p + "knee_sagittal_moment"], st, dt)
        row["peak_knee_extension_moment_Nm_mean"] = float(pk.mean())
        pk, _ = stance_metrics(fz, st, dt)
        row["peak_fz_r_BW_mean"] = float(pk.mean() / bw)
        row["liftoff_frames_in_stance"] = int(sum(
            np.sum(out[p + "lig_lateral"][a:b_ + 1] > 0) for a, b_ in st))
        side["trials"][str(slot)] = row
    out["stride_s"] = np.array(stride_s)
    out["step_freq_hz"] = np.array(sf_hz)
    out["syn_d_spm"] = np.array(SYN_D, float)
    out["syn_k"] = np.array(syn_k)
    out["syn_stride_s"] = np.array(syn_stride)
    out["syn_step_freq_hz"] = 2.0 / np.array(syn_stride)
    np.savez_compressed(OUT_NPZ, **out)

    side.update({
        "body_mass_kg": mass, "BW_N": bw,
        "body_mass_source": "trial 13 boneid estimate (data/opensim/work/p1_trial13_boneid.npz), same as the trial-13 knee contact",
        "d_ic_m": d_ic,
        "d_ic_note": f"40 mm generic x tibia_r uniform scale factor {f_tib:.4f} (same as trial 13)",
        "units": {"s{slot}_total/medial/lateral/lig_lateral": "BW (body_mass x 9.80665), NaN outside the SO window",
                  "s{slot}_fz_r": "N (right-foot vertical GRF, mocap clock)",
                  "s{slot}_knee_flexion_deg": "deg, OpenSim knee_angle_r (+ = flexion), 12 Hz low-passed IK",
                  "s{slot}_knee_adduction_moment": "N m, EXTERNAL knee adduction moment (+ = varus) = -M_x of the inertia+gravity+GRF reaction at the knee centre in the tibia frame, NaN outside the SO window. The walker knee has one DOF, so OpenSim ID has no frontal knee coordinate; this is the same frontal-moment definition as trial 13's ext_frontal_moment",
                  "s{slot}_knee_sagittal_moment": "N m, OpenSim ID knee_angle_r_moment (+ = flexion moment, so extension is negative)",
                  "s{slot}_stance_frames": "[n,2] (heel strike, toe off) frame indices on s{slot}_t of complete right stances (Fz > 20 N, lasting >= 0.2 s and peaking >= 0.3 BW) inside the SO window",
                  "stride_s / step_freq_hz": "mean right heel-strike interval over the whole trial; step frequency = 2 / stride"},
        "peak_definition": "mean over the complete right stances of each stance's maximum; impulse = trapezoid integral over each stance (BW s), mean over stances",
        "settings": {"model": "trial-13 p1_scaled.osim reused (no rescale, no marker re-placement); processed model rebuilt by pipeline.processed_model() from it (DGF rigid tendon, no passive, F-L width x1.5, welded subtalar/mtp/wrist, pelvis residuals 100, 1 N m reserves)",
                     "ik": "one pass, trial-13 final weights, accuracy 1e-5",
                     "id": "12 Hz coordinate low-pass, muscles excluded",
                     "so": "sum of activation^2, muscle physiology on, convergence 1e-4, max 200 iterations, 12 Hz coordinate low-pass, window 0.10 s to T_end - 0.10 s",
                     "contact": "knee_contact.knee_wrench + split: subtree Newton-Euler below the knee, two-point medial/lateral split, lateral lift-off clamped and reported as ligament tension"},
        "trial13_reference_1p8ms": {"so_peak_total_BW_mean": base["baseline"]["so"]["peak_total_BW"]["mean_of_stance_peaks"],
                                    "so_peak_medial_BW_mean": base["baseline"]["so"]["peak_medial_BW"]["mean_of_stance_peaks"],
                                    "so_peak_lateral_BW_mean": base["baseline"]["so"]["peak_lateral_BW"]["mean_of_stance_peaks"]},
        "conditions": "van der Zee et al. 2022 Table 2: fixed speed 1.25 m/s (belt measured ~1.17 m/s), f* = step frequency / preferred",
        "moco": "not run (SO only)",
        "synthetic": "syn<+d>_* keys: slot 19 re-timed by boneid.simulate.step_frequency_trial(trial, k, 85.16) to f0 + d steps/min at the same belt speed (f0 = slot 19's mean right-plate-onset stride, k = (f0 + d)/f0); exported with export_inputs.py --syn-d-spm d and run through the identical IK/ID/SO/contact pipeline. Hip centres (HHR/HHL landmarks) are transformed with the markers. d = +0 reproduces slot 19's inputs to 1e-8 (print rounding).",
    })
    with open(OUT_JSON, "w") as f:
        json.dump(side, f, indent=2, default=float)
    print("trial   f*/k    spm IKrms  pkTot  pkMed  pkLat  pkLig impTot impMed impLat   KAM nSt")
    for s in SLOTS + syn_keys:
        r = side["trials"][str(s)]
        fk = r["f_star"] if r["f_star"] is not None else r["syn_k"]
        print(f"{str(s):>6} {fk:6.3f} {r['steps_per_min']:6.1f} {r['ik_rms_mean_mm']:5.1f} "
              f"{r['peak_total_BW_mean']:6.2f} {r['peak_medial_BW_mean']:6.2f} "
              f"{r['peak_lateral_BW_mean']:6.2f} {r['peak_lig_lateral_BW_mean']:6.2f} "
              f"{r['impulse_total_BWs_mean']:6.3f} {r['impulse_medial_BWs_mean']:6.3f} "
              f"{r['impulse_lateral_BWs_mean']:6.3f} {r['peak_knee_adduction_moment_Nm_mean']:5.1f} "
              f"{r['n_complete_stances_in_window']:3d}")

if __name__ == "__main__":
    osim.Logger.setLevelString("warn")
    cmd = sys.argv[1]
    if cmd == "run":
        for a in sys.argv[2:]:
            stage_run(a if a.startswith("syn") else int(a))
    elif cmd == "collect":
        stage_collect()
