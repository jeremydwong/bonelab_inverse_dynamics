"""Right-leg fibre kinematics + muscle parameters -> p1_trial13_muscles.npz.

Run in the conda env `opensim` (cwd doesn't matter):

    PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
    $PY scripts/opensim/fiber_kinematics.py

With rigid tendons and prescribed kinematics, fibre length and velocity are
pure geometry: l_MT(q) - l_slack = l_M cos(alpha), with l_M sin(alpha) held
at l_opt sin(alpha_opt). So SO and MocoInverse should see the SAME fibre
kinematics. This script measures both and checks it:

- SO side: an AnalyzeTool + MuscleAnalysis on exactly the SO setup (same
  processed model, IK file, 12 Hz low-pass, window). AnalyzeTool's own
  kinematics prep is what SO saw.
- MocoInverse side: the saved MocoInverse solution re-analysed through its
  own MocoStudy (`study.analyze`, same as MocoInverse does for its outputs).

New npz keys (all [T,M] on t = i/120, NaN outside the Moco window; the
values stored are the MocoInverse ones): fiber_length_norm, fiber_velocity_norm
(v_M / (v_max l_opt), + = lengthening), mtu_length, tendon_length (m),
pennation (rad); per muscle f_iso (N), l_opt, l_slack (m), v_max (l_opt/s),
alpha_opt (rad). Existing keys are copied through untouched. Sidecar JSON
gets a "fiber_kinematics" block (SO vs dyn agreement, model constants).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

osim.Logger.setLevelString("warn")

NPZ = os.path.join(P.OUT, "p1_trial13_muscles.npz")
JSN = os.path.join(P.OUT, "p1_trial13_muscles.json")
MA_DIR = "fiber"                         # under data/opensim
NAME = "p1_trial13_fiber"
OUTS = ["normalized_fiber_length", "normalized_fiber_velocity",
        "fiber_length", "tendon_length", "pennation_angle",
        "active_force_length_multiplier", "force_velocity_multiplier"]


def read_sto(path):
    tbl = osim.TimeSeriesTable(path)
    return (np.array(tbl.getIndependentColumn()),
            list(tbl.getColumnLabels()), tbl.getMatrix().to_numpy())


def interp(t, ts, y):
    out = np.stack([np.interp(t, ts, y[:, j]) for j in range(y.shape[1])], 1)
    out[(t < ts[0] - 1e-6) | (t > ts[-1] + 1e-6)] = np.nan
    return out


def so_side(muscles):
    """MuscleAnalysis on the SO setup (same AnalyzeTool kinematics)."""
    at = osim.AnalyzeTool("p1_setup_so.xml")
    at.setName(NAME)
    at.setResultsDir(MA_DIR)
    aset = at.getAnalysisSet()
    while aset.getSize():
        aset.remove(0)
    ma = osim.MuscleAnalysis()
    ma.setName("MuscleAnalysis")
    ma.setStartTime(P.WINDOW[0])
    ma.setEndTime(P.WINDOW[1])
    ma.setComputeMoments(False)
    names = osim.ArrayStr()
    for m in muscles:
        names.append(m)
    ma.setMuscles(names)
    aset.cloneAndAppend(ma)
    at.printToXML("p1_setup_fiber_analysis.xml")
    t0 = time.time()
    osim.AnalyzeTool("p1_setup_fiber_analysis.xml").run()
    rt = time.time() - t0
    out = {}
    for key, fn, deg in (("nfl", "NormalizedFiberLength", False),
                         ("nfv", "NormFiberVelocity", False),
                         ("lmt", "Length", False),
                         ("lt", "TendonLength", False),
                         ("penn", "PennationAngle", True)):
        ts, lab, y = read_sto(f"{MA_DIR}/{NAME}_MuscleAnalysis_{fn}.sto")
        y = y[:, [lab.index(m) for m in muscles]]
        if deg and np.nanmax(np.abs(y)) > 2.0:     # MuscleAnalysis -> degrees
            y = np.deg2rad(y)
        out[key] = (ts, y)
    keep = {f"{NAME}_MuscleAnalysis_{fn}.sto" for fn in (
        "NormalizedFiberLength", "NormFiberVelocity", "Length",
        "TendonLength", "PennationAngle", "FiberLength", "FiberVelocity")}
    for fn in os.listdir(MA_DIR):           # forces/powers: not needed
        if fn.endswith(".sto") and fn not in keep:
            os.remove(os.path.join(MA_DIR, fn))
    return out, rt


def dyn_side(muscles):
    """Re-analyse the saved MocoInverse solution through its own study."""
    inv = osim.MocoInverse.safeDownCast(
        osim.OpenSimObject.makeObjectFromFile("p1_setup_mocoinverse.xml"))
    study = inv.initialize()
    sol = osim.MocoTrajectory("p1_trial13_mocoinverse_solution.sto")
    paths = osim.StdVectorString()
    for o in OUTS:
        paths.append(rf".*_r\|{o}")
    paths.append(r".*_r/path\|length")
    t0 = time.time()
    tbl = study.analyze(sol, paths)
    rt = time.time() - t0
    ts = np.array(tbl.getIndependentColumn())
    lab = list(tbl.getColumnLabels())
    y = tbl.getMatrix().to_numpy()

    def cols(fmt):
        return y[:, [lab.index(fmt.format(m)) for m in muscles]]
    out = {o: (ts, cols("/forceset/{}|" + o)) for o in OUTS}
    out["length"] = (ts, cols("/forceset/{}/path|length"))
    return out, rt


# DeGrooteFregly2016Muscle curve constants (OpenSim 4.6 header)
B = [(0.8150671134243542, 1.055033428970575, 0.162384573599574,
      0.063303448465465),
     (0.433004984392647, 0.716775413397760, -0.029947116970696,
      0.200356847296188),
     (0.1, 1.0, 0.353553390593274, 0.0)]
D1, D2, D3, D4 = -0.3211346127989808, -8.149, -0.374, 0.8825327733249912


def f_l(l, width=1.5):
    x = (l - 1.0) / width + 1.0
    return sum(b1 * np.exp(-0.5 * (x - b2) ** 2 / (b3 + b4 * x) ** 2)
               for b1, b2, b3, b4 in B)


def f_v(v):
    u = D2 * v + D3
    return D1 * np.log(u + np.sqrt(u * u + 1.0)) + D4


def activation_ode_check(model, name, shared):
    """OpenSim's own da/dt vs the DGF formula at a few (a, e) pairs."""
    ta = shared["activation_time_constant_s"][0]
    td = shared["deactivation_time_constant_s"][0]
    b = shared["activation_dynamics_smoothing_b"][0]
    state = model.initSystem()
    mu = osim.DeGrooteFregly2016Muscle.safeDownCast(model.getMuscles().get(name))
    acts = model.getActuators()
    idx = [i for i in range(acts.getSize()) if acts.get(i).getName() == name][0]
    worst = 0.0
    for a, e in ((0.01, 1.0), (0.5, 0.9), (0.9, 0.01), (0.3, 0.3), (0.2, 0.05)):
        mu.setActivation(state, a)
        model.realizeVelocity(state)
        c = model.getControls(state)
        c.set(idx, e)
        model.setControls(state, c)
        model.realizeAcceleration(state)
        f = 0.5 * np.tanh(b * (e - a))
        z = 0.5 + 1.5 * a
        ours = ((f + 0.5) / (ta * z) + (0.5 - f) * z / td) * (e - a)
        worst = max(worst, abs(ours - mu.getStateVariableDerivativeValue(
            state, "activation")))
    return {"muscle": name, "pairs": 5, "max_abs_diff_per_s": worst,
            "formula": "da/dt = [(f+0.5)/(tau_a (0.5+1.5a)) + (0.5+1.5a)"
                       "(0.5-f)/tau_d] (e-a), f = 0.5 tanh(b (e-a))"}


def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[ok] - b[ok])
    return {"max_abs_diff": float(d.max()), "rms_diff": float(np.sqrt(
        np.mean(d ** 2))), "n": int(ok.sum())}


def main():
    os.chdir(P.OUT)
    os.makedirs(MA_DIR, exist_ok=True)
    old = dict(np.load(NPZ, allow_pickle=False))
    t = old["t"]
    muscles = [str(m) for m in old["muscle_names"]]

    model = osim.Model(P.PROCESSED)
    model.initSystem()
    dgf = [osim.DeGrooteFregly2016Muscle.safeDownCast(
        model.getMuscles().get(m)) for m in muscles]
    par = {k: np.array([f(d) for d in dgf]) for k, f in {
        "f_iso": lambda d: d.get_max_isometric_force(),
        "l_opt": lambda d: d.get_optimal_fiber_length(),
        "l_slack": lambda d: d.get_tendon_slack_length(),
        "v_max": lambda d: d.get_max_contraction_velocity(),
        "alpha_opt": lambda d: d.get_pennation_angle_at_optimal(),
    }.items()}
    shared = {k: sorted({f(d) if isinstance(f(d), bool) else float(f(d))
                         for d in dgf}) for k, f in {
        "activation_time_constant_s": lambda d: d.get_activation_time_constant(),
        "deactivation_time_constant_s":
            lambda d: d.get_deactivation_time_constant(),
        "activation_dynamics_smoothing_b":
            lambda d: d.get_activation_dynamics_smoothing(),
        "active_force_width_scale": lambda d: d.get_active_force_width_scale(),
        "fiber_damping": lambda d: d.get_fiber_damping(),
        "ignore_passive_fiber_force":
            lambda d: bool(d.get_ignore_passive_fiber_force()),
        "ignore_tendon_compliance":
            lambda d: bool(d.get_ignore_tendon_compliance()),
        "ignore_activation_dynamics":
            lambda d: bool(d.get_ignore_activation_dynamics()),
        "tendon_strain_at_one_norm_force":
            lambda d: d.get_tendon_strain_at_one_norm_force(),
    }.items()}
    e0 = np.array([d.get_passive_fiber_strain_at_one_norm_force()
                            for d in dgf])

    act_check = activation_ode_check(model, muscles[0], shared)
    so, rt_so = so_side(muscles)
    dy, rt_dyn = dyn_side(muscles)

    D = {"fiber_length_norm": interp(t, *dy["normalized_fiber_length"]),
         "fiber_velocity_norm": interp(t, *dy["normalized_fiber_velocity"]),
         "mtu_length": interp(t, *dy["length"]),
         "tendon_length": interp(t, *dy["tendon_length"]),
         "pennation": interp(t, *dy["pennation_angle"])}
    S = {"fiber_length_norm": interp(t, *so["nfl"]),
         "fiber_velocity_norm": interp(t, *so["nfv"]),
         "mtu_length": interp(t, *so["lmt"]),
         "tendon_length": interp(t, *so["lt"]),
         "pennation": interp(t, *so["penn"])}

    # geometry identity check on the Moco side: l_MT = l_s + l_M cos(alpha)
    lm = interp(t, *dy["fiber_length"])
    geo = np.nanmax(np.abs(D["mtu_length"] - D["tendon_length"]
                           - lm * np.cos(D["pennation"])))
    # tendon really rigid?
    rigid = np.nanmax(np.abs(D["tendon_length"] - par["l_slack"][None]))

    # force reconstruction on each solver's OWN time grid (no interpolation):
    # which terms does each solver's reported force contain?
    fiso, beta = par["f_iso"][None], shared["fiber_damping"][0]
    tm, lab, O = read_sto("p1_trial13_mocoinverse_outputs.sto")
    Fd = O[:, [lab.index(f"/forceset/{m}|tendon_force") for m in muscles]]
    ad = O[:, [lab.index(f"/forceset/{m}|activation") for m in muscles]]
    ld, vd = dy["normalized_fiber_length"][1], dy["normalized_fiber_velocity"][1]
    cd = np.cos(dy["pennation_angle"][1])
    curve_err = max(
        np.abs(f_l(ld, shared["active_force_width_scale"][0])
               - dy["active_force_length_multiplier"][1]).max(),
        np.abs(f_v(vd) - dy["force_velocity_multiplier"][1]).max())
    tso, lab, Fs = read_sto("so/p1_trial13_so_StaticOptimization_force.sto")
    Fs = Fs[:, [lab.index(m) for m in muscles]]
    _, lab, As = read_sto("so/p1_trial13_so_StaticOptimization_activation.sto")
    As = As[:, [lab.index(m) for m in muscles]]
    ls, vs = so["nfl"][1], so["nfv"][1]
    cs = np.cos(so["penn"][1])
    w = shared["active_force_width_scale"][0]
    act_s = As * f_l(ls, w) * f_v(vs)
    act_d = ad * f_l(ld, w) * f_v(vd)
    check = {
        "curves_numpy_vs_opensim_max_abs": float(curve_err),
        "dyn: Fiso (a fL fV + beta vM) cos(alpha)": float(np.abs(
            fiso * (act_d + beta * vd) * cd - Fd).max()),
        "dyn: Fiso a fL fV cos(alpha) (no damping)": float(np.abs(
            fiso * act_d * cd - Fd).max()),
        "so: Fiso a fL fV (no cos, no damping)": float(np.abs(
            fiso * act_s - Fs).max()),
        "so: Fiso a fL fV cos(alpha)": float(np.abs(
            fiso * act_s * cs - Fs).max()),
        "so: Fiso (a fL fV + beta vM) cos(alpha)": float(np.abs(
            fiso * (act_s + beta * vs) * cs - Fs).max()),
        "units": "max |reconstruction - reported force| over all right-leg "
                 "muscles and samples of that solver's own grid, N",
    }

    # SO vs dyn fibre kinematics on the Moco grid (SO's 120 Hz series
    # interpolated onto the 513 Moco times)
    td = dy["normalized_fiber_length"][0]
    agree = {}
    for k, (sk, dk) in {"fiber_length_norm": ("nfl", "normalized_fiber_length"),
                        "fiber_velocity_norm": ("nfv", "normalized_fiber_velocity"),
                        "mtu_length": ("lmt", "length"),
                        "tendon_length": ("lt", "tendon_length"),
                        "pennation": ("penn", "pennation_angle")}.items():
        a_ = interp(td, *so[sk])
        d_ = np.abs(a_ - dy[dk][1])
        i, j = np.unravel_index(np.nanargmax(d_), d_.shape)
        agree[k] = {**stats(a_, dy[dk][1]),
                    "p99_abs_diff": float(np.nanpercentile(d_, 99)),
                    "argmax_t_s": float(td[i]), "argmax_muscle": muscles[j]}
    S = {"fiber_length_norm": interp(t, *so["nfl"])}
    inside = np.isfinite(D["fiber_length_norm"][:, 0])
    ranges = {}
    for i, m in enumerate(muscles):
        l = D["fiber_length_norm"][inside, i]
        v = D["fiber_velocity_norm"][inside, i]
        ranges[m] = {"lM_norm_min": float(l.min()), "lM_norm_max": float(l.max()),
                     "lM_norm_frac_above_1": float(np.mean(l > 1.0)),
                     "vM_norm_min": float(v.min()), "vM_norm_max": float(v.max())}

    new = {**old, **{k: v for k, v in D.items()},
           "f_iso": par["f_iso"], "l_opt": par["l_opt"],
           "l_slack": par["l_slack"], "v_max": par["v_max"],
           "alpha_opt": par["alpha_opt"],
           "passive_strain_e0": e0}
    np.savez(NPZ, **new)

    with open(JSN) as f:
        side = json.load(f)
    # the earlier sidecar quoted DGF's default 0.015/0.060 s; the processed
    # model actually carries the Rajagopal model's 0.010/0.040 s
    side["dynamic_optimization"]["muscle_model"] = (
        "DeGrooteFregly2016, rigid tendon, passive fibre force ignored, "
        "active force-length width x1.5, fibre damping 0.1, activation "
        "dynamics tau_act/tau_deact = {:.3f}/{:.3f} s (inherited from the "
        "Rajagopal model by ModOpReplaceMusclesWithDeGrooteFregly2016), "
        "tanh smoothing b = {:g}".format(
            shared["activation_time_constant_s"][0],
            shared["deactivation_time_constant_s"][0],
            shared["activation_dynamics_smoothing_b"][0]))
    side["fiber_kinematics"] = {
        "keys": {
            "fiber_length_norm": "[T,M] l_M / l_opt (MocoInverse output "
                                 "normalized_fiber_length)",
            "fiber_velocity_norm": "[T,M] v_M / (v_max l_opt), + = "
                                   "lengthening (normalized_fiber_velocity)",
            "mtu_length": "[T,M] m, musculotendon path length",
            "tendon_length": "[T,M] m (rigid: = l_slack)",
            "pennation": "[T,M] rad",
            "f_iso": "[M] N", "l_opt": "[M] m", "l_slack": "[M] m",
            "v_max": "[M] l_opt/s", "alpha_opt": "[M] rad",
            "passive_strain_e0": "[M] passive_fiber_strain_at_one_norm_force "
                                 "(only matters if passive force were on)",
        },
        "source": "values stored are from the MocoInverse solution "
                  "(study.analyze); the SO side is an AnalyzeTool + "
                  "MuscleAnalysis on the identical SO setup",
        "so_vs_dyn_agreement": agree,
        "so_vs_dyn_grid": "SO/MuscleAnalysis 613 samples (AnalyzeTool "
                          "grid) interpolated onto the 513 MocoInverse "
                          "mesh times",
        "geometry_identity_max_abs_m": float(geo),
        "tendon_minus_slack_max_abs_m": float(rigid),
        "force_reconstruction": check,
        "model_constants": shared,
        "activation_ode_check": act_check,
        "v_max_lopt_per_s": sorted(set(par["v_max"].tolist())),
        "operating_range_dyn_window": ranges,
        "runtime_s": {"muscle_analysis": rt_so, "moco_analyze": rt_dyn},
    }
    with open(JSN, "w") as f:
        json.dump(side, f, indent=2, default=float)
    print(json.dumps({"agree": agree, "geo": geo, "rigid": rigid,
                      "check": check, "shared": shared}, indent=1))


if __name__ == "__main__":
    main()
