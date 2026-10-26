"""Tendon-compliance ladder for P1 trial 13: MocoInverse with compliant tendons.

Run in the conda env `opensim` (like pipeline.py), cwd anywhere:

    PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
    $PY scripts/opensim/compliance.py list            # l_slack / l_opt table + C2 list
    $PY scripts/opensim/compliance.py c1              # 58 min / 540 iters, warm start = rigid
    $PY scripts/opensim/compliance.py c2              # 63 min / 578 iters, warm start = C1
    $PY scripts/opensim/compliance.py c3              # 85 min / 104 iters, warm start = C2
    # variants: tag=_cold guess=cold | eqgoal=0 | prev=rigid | mesh=0.04 (-> c3_coarse)
    $PY scripts/opensim/compliance.py collect         # -> p1_trial13_muscles.npz c*_ keys + JSON
    $PY scripts/opensim/compliance.py contact         # -> p1_trial13_knee_contact.npz c*_ keys

The ladder (everything else is the rigid MocoInverse of pipeline.py: same
processed model, IK, 12 Hz filter, window 0.10-5.20 s, 256 mesh intervals
Hermite-Simpson, tolerances 1e-3, reserves 1 N m, pelvis residuals):

- C1: compliant tendons on soleus, gasmed, gaslat (BOTH legs: the MocoInverse
  solves the whole model, both legs' muscles, so the left triceps surae goes
  compliant too).
- C2: compliant tendons on every muscle with l_slack / l_opt > 1 (46 of 80).
- C3: every muscle compliant AND passive fibre force on.

DGF tendon compliance with IMPLICIT tendon dynamics (state =
normalized_tendon_force, its derivative is a solver variable with bounds
[-1000, 1000] / s and MocoInverse's default 0.01-weighted squared-derivative
cost).  MocoInverse does not add MocoInitialVelocityEquilibriumDGFGoal on its
own, so I add it (endpoint constraint), otherwise the tendon-force derivative
is free at t0.  Tendon stiffness is the model's (tendon_strain_at_one_norm_force
0.049 for every muscle: the Rajagopal Millard muscles use the default tendon
curve, and ModOpReplaceMusclesWithDeGrooteFregly2016 keeps DGF's default).

Warm start: each stage's guess is the previous stage's solution (rigid -> C1
-> C2 -> C3) copied by name onto the new mesh; tendons that are newly
compliant start at the previous stage's tendon force / F_iso with a finite
difference derivative.

Intermediates (solution, outputs, logs, info) go to data/opensim/compliant/
(gitignored).  Only the collected npz keys and JSON blocks are committed.
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

CDIR = os.path.join(P.OUT, "compliant")
NPZ = os.path.join(P.OUT, "p1_trial13_muscles.npz")
JSN = os.path.join(P.OUT, "p1_trial13_muscles.json")
KNPZ = os.path.join(P.OUT, "p1_trial13_knee_contact.npz")
KJSN = os.path.join(P.OUT, "p1_trial13_knee_contact.json")
RIGID_SOL = os.path.join(P.OUT, "p1_trial13_mocoinverse_solution.sto")
RIGID_OUT = os.path.join(P.OUT, "p1_trial13_mocoinverse_outputs.sto")
TRICEPS = ("soleus", "gasmed", "gaslat")
RATIO_THRESHOLD = 1.0
STAGES = ("c1", "c2", "c3")
PREV = {"c1": "rigid", "c2": "c1", "c3": "c2"}
# DGF tendon curve constants (DeGrooteFregly2016Muscle.h, OpenSim 4.6)
TC1, TC2, TC3 = 0.2, 1.0, 0.2
OUTS = ["tendon_force", "activation", "normalized_fiber_length",
        "normalized_fiber_velocity", "tendon_length", "tendon_strain",
        "tendon_velocity", "fiber_velocity_along_tendon",
        "fiber_force_along_tendon", "passive_fiber_force_along_tendon",
        "pennation_angle", "speed"]


def log(msg):
    print(f"[compliance] {msg}", flush=True)


def read_sto(path):
    tbl = osim.TimeSeriesTable(path)
    return (np.array(tbl.getIndependentColumn()),
            list(tbl.getColumnLabels()), tbl.getMatrix().to_numpy())


def dgf_list(model):
    ms = model.getMuscles()
    return [osim.DeGrooteFregly2016Muscle.safeDownCast(ms.get(i))
            for i in range(ms.getSize())]


def ratio_table():
    m = osim.Model(os.path.join(P.OUT, P.PROCESSED))
    m.initSystem()
    rows = []
    for d in dgf_list(m):
        ls, lo = d.get_tendon_slack_length(), d.get_optimal_fiber_length()
        eT = d.get_tendon_strain_at_one_norm_force()
        kT = np.log((1 + TC3) / TC1) / eT
        k_norm = TC1 * kT * np.exp(kT * eT)          # dfT/dlT~ at F = F_iso
        rows.append({"muscle": d.getName(), "l_slack_m": ls, "l_opt_m": lo,
                     "ratio": ls / lo, "f_iso_N": d.get_max_isometric_force(),
                     "tendon_strain_at_one_norm_force": eT,
                     "kT": kT,
                     "stiffness_at_f_iso_N_per_m":
                         k_norm * d.get_max_isometric_force() / ls,
                     "tendon_stretch_at_f_iso_mm": 1000 * eT * ls,
                     "passive_fiber_strain_at_one_norm_force":
                         d.get_passive_fiber_strain_at_one_norm_force()})
    return rows


def compliant_names(stage, rows):
    if stage == "c1":
        return [r["muscle"] for r in rows
                if r["muscle"].rsplit("_", 1)[0] in TRICEPS]
    if stage == "c2":
        return [r["muscle"] for r in rows if r["ratio"] > RATIO_THRESHOLD]
    return [r["muscle"] for r in rows]


def build_model(stage):
    rows = ratio_table()
    names = set(compliant_names(stage, rows))
    mp = P.processed_model()                       # rigid, passive off
    mp.append(osim.ModOpAddExternalLoads(P.EXT_LOADS))
    model = mp.process()
    model.initSystem()
    ms = model.updMuscles()
    for i in range(ms.getSize()):
        d = osim.DeGrooteFregly2016Muscle.safeDownCast(ms.get(i))
        if d.getName() in names:
            d.set_ignore_tendon_compliance(False)
            d.set_tendon_compliance_dynamics_mode("implicit")
        if stage == "c3":
            d.set_ignore_passive_fiber_force(False)
    model.finalizeConnections()
    return model, sorted(names)


def prev_files(stage, prev_stage=None):
    p = prev_stage or PREV[stage]
    if p == "rigid":
        return RIGID_SOL, RIGID_OUT
    return (os.path.join(CDIR, f"{p}_solution.sto"),
            os.path.join(CDIR, f"{p}_outputs.sto"))


def warm_start(solver, stage, model, prev_stage=None):
    guess = solver.createGuess()
    tg = np.asarray(guess.getTimeMat())
    sol_f, out_f = prev_files(stage, prev_stage)
    prev = osim.MocoTrajectory(sol_f)
    tp = np.asarray(prev.getTimeMat())

    def interp(y):
        return osim.Vector(np.interp(tg, tp, y).tolist())
    copied, from_force = [], []
    pst = list(prev.getStateNames())
    for n in guess.getStateNames():
        if n in pst:
            guess.setState(n, interp(np.asarray(prev.getStateMat(n))))
            copied.append(n)
    for n in guess.getControlNames():
        if n in list(prev.getControlNames()):
            guess.setControl(n, interp(np.asarray(prev.getControlMat(n))))
            copied.append(n)
    for n in guess.getMultiplierNames():
        if n in list(prev.getMultiplierNames()):
            guess.setMultiplier(n, interp(np.asarray(prev.getMultiplierMat(n))))
            copied.append(n)
    pder = list(prev.getDerivativeNames())
    to, lab, O = read_sto(out_f)
    for n in guess.getStateNames():
        if not n.endswith("/normalized_tendon_force"):
            continue
        mus = n.split("/")[2]
        d = osim.DeGrooteFregly2016Muscle.safeDownCast(
            model.getMuscles().get(mus))
        if n in pst:
            ntf = np.asarray(prev.getStateMat(n))
            ntf = np.interp(tg, tp, ntf)
        else:
            F = O[:, lab.index(f"/forceset/{mus}|tendon_force")]
            ntf = np.clip(np.interp(tg, to, F) / d.get_max_isometric_force(),
                          0.0, 5.0)
            guess.setState(n, osim.Vector(ntf.tolist()))
            from_force.append(mus)
        dn = n.replace("normalized_tendon_force",
                       "implicitderiv_normalized_tendon_force")
        if dn in pder:
            dv = np.interp(tg, tp, np.asarray(prev.getDerivativeMat(dn)))
        else:
            dv = np.clip(np.gradient(ntf, tg), -1000, 1000)
        guess.setDerivative(dn, osim.Vector(dv.tolist()))
    return guess, {"copied_by_name": len(copied),
                   "tendon_state_from_previous_force": from_force,
                   "previous_solution": os.path.relpath(sol_f, P.OUT)}


def kinematics_table():
    """Kinematic states of the rigid MocoInverse solution (MocoInverse puts
    them in its solution; the study solved directly does not)."""
    tbl = osim.MocoTrajectory(RIGID_SOL).exportToStatesTable()
    sub = osim.TimeSeriesTable(tbl.getIndependentColumn())
    for lab in tbl.getColumnLabels():
        if lab.startswith("/jointset"):
            sub.appendColumn(lab, tbl.getDependentColumn(lab))
    return sub


def stage_solve(stage, mesh=None, tol=None, max_iter=3000, tag="",
                guess="warm", eqgoal=1, prev=None):
    """Solve one stage. `tag` suffixes the output names (variants), `guess`
    = warm | cold (MocoInverse's own default guess), `eqgoal` = 0 drops the
    initial velocity-equilibrium goal, `prev` overrides the warm-start
    stage (e.g. prev=rigid)."""
    os.makedirs(CDIR, exist_ok=True)
    os.chdir(P.OUT)
    model, names = build_model(stage)
    inv = osim.MocoInverse()
    inv.setName(f"p1_trial13_mocoinverse_{stage}")
    inv.setModel(osim.ModelProcessor(model))
    tp = osim.TableProcessor(P.IK)
    tp.append(osim.TabOpLowPassFilter(P.LOWPASS_HZ))
    tp.append(osim.TabOpUseAbsoluteStateNames())
    inv.setKinematics(tp)
    inv.set_initial_time(P.WINDOW[0])
    inv.set_final_time(P.WINDOW[1])
    inv.set_mesh_interval(mesh or P.MESH)
    inv.set_kinematics_allow_extra_columns(True)
    inv.set_minimize_sum_squared_activations(True)
    inv.set_convergence_tolerance(tol or 1e-3)
    inv.set_constraint_tolerance(tol or 1e-3)
    inv.set_max_iterations(max_iter)
    study = inv.initialize()
    problem = study.updProblem()
    if int(eqgoal):
        problem.addGoal(osim.MocoInitialVelocityEquilibriumDGFGoal(
            "initial_velocity_equilibrium"))
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    # the Python copy of the study needs its solver re-pointed at the problem
    solver.resetProblem(study.getProblem())
    if guess == "warm":
        g, ginfo = warm_start(solver, stage, model, prev)
        solver.setGuess(g)
    else:
        ginfo = {"cold": "MocoInverse default guess (bounds midpoints)"}
    study.printToXML(os.path.join(CDIR, f"{stage}{tag}_study.xml"))
    settings = {
        "stage": stage, "compliant_muscles": names,
        "n_compliant": len(names),
        "passive_fiber_force": stage == "c3",
        "tendon_compliance_dynamics_mode": "implicit",
        "solver": "MocoCasADiSolver (CasADi + IPOPT), MocoInverse problem "
                  "solved through study.solve() so a guess can be set",
        "transcription_scheme": solver.get_transcription_scheme(),
        "num_mesh_intervals": solver.get_num_mesh_intervals(),
        "mesh_interval_s": mesh or P.MESH,
        "optim_convergence_tolerance": solver.get_optim_convergence_tolerance(),
        "optim_constraint_tolerance": solver.get_optim_constraint_tolerance(),
        "optim_max_iterations": solver.get_optim_max_iterations(),
        "optim_hessian_approximation": solver.get_optim_hessian_approximation(),
        "minimize_implicit_auxiliary_derivatives":
            bool(solver.get_minimize_implicit_auxiliary_derivatives()),
        "implicit_auxiliary_derivatives_weight":
            solver.get_implicit_auxiliary_derivatives_weight(),
        "extra_goals": (["MocoInitialVelocityEquilibriumDGFGoal (endpoint "
                         "constraint)"] if int(eqgoal) else []),
        "variant_tag": tag,
        "window_s": list(P.WINDOW),
        "guess": ginfo,
    }
    log(f"{stage}: {len(names)} compliant muscles; settings "
        f"{ {k: v for k, v in settings.items() if k != 'compliant_muscles'} }")
    t0 = time.time()
    sol = study.solve()
    rt = time.time() - t0
    success = bool(sol.success())
    sol.unseal()
    sol.insertStatesTrajectory(kinematics_table(), False)
    sol.write(os.path.join(CDIR, f"{stage}{tag}_solution.sto"))
    info = {**settings, "runtime_s": rt, "success": success,
            "status": sol.getStatus(),
            "num_iterations": sol.getNumIterations(),
            "objective": sol.getObjective()}
    try:
        info["objective_terms"] = {
            n: sol.getObjectiveTerm(n) for n in sol.getObjectiveTermNames()}
    except Exception:                     # noqa: BLE001
        pass
    with open(os.path.join(CDIR, f"{stage}{tag}_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)
    log(f"{stage}: success={success} status={sol.getStatus()} iters="
        f"{sol.getNumIterations()} obj={sol.getObjective():.5g} runtime "
        f"{rt:.0f} s")
    analyze(stage + tag, study, sol)


def analyze(stage, study=None, sol=None):
    if study is None:
        model, _ = build_model(stage[:2])
        inv = osim.MocoInverse()
        inv.setModel(osim.ModelProcessor(model))
        tp = osim.TableProcessor(P.IK)
        tp.append(osim.TabOpLowPassFilter(P.LOWPASS_HZ))
        tp.append(osim.TabOpUseAbsoluteStateNames())
        inv.setKinematics(tp)
        inv.set_initial_time(P.WINDOW[0])
        inv.set_final_time(P.WINDOW[1])
        inv.set_mesh_interval(P.MESH)
        inv.set_kinematics_allow_extra_columns(True)
        study = inv.initialize()
        sol = osim.MocoTrajectory(os.path.join(CDIR, f"{stage}_solution.sto"))
    paths = osim.StdVectorString()
    for o in OUTS:
        paths.append(rf".*\|{o}")
    t0 = time.time()
    tbl = study.analyze(sol, paths)
    osim.STOFileAdapter.write(tbl, os.path.join(CDIR, f"{stage}_outputs.sto"))
    log(f"{stage}: outputs analysed in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------
# collect -> p1_trial13_muscles.npz / .json
# ---------------------------------------------------------------------------

def interp(t, ts, y):
    out = np.stack([np.interp(t, ts, y[:, j]) for j in range(y.shape[1])], 1)
    out[(t < ts[0] - 1e-6) | (t > ts[-1] + 1e-6)] = np.nan
    return out


def stances(fz, t, lo, hi):
    on = fz > 20.0
    e = np.flatnonzero(np.diff(on.astype(int)))
    return [(a + 1, b) for a, b in zip(e[:-1], e[1:])
            if on[a + 1] and t[a + 1] >= lo and t[b] <= hi]


def work_split(F, vT, vM, t, st):
    """Positive / negative work (J) per stance of the summed unit, delivered
    to the skeleton: P = -F v (v + = lengthening)."""
    out = {}
    for key, v in (("mtu", vT + vM), ("tendon", vT), ("fibre", vM)):
        p = -np.sum(F * v, axis=1)
        pos, neg = [], []
        for a, b in st:
            pp = p[a:b + 1]
            tt = t[a:b + 1]
            pos.append(float(np.trapezoid(np.maximum(pp, 0), tt)))
            neg.append(float(np.trapezoid(np.minimum(pp, 0), tt)))
        out[key] = {"positive_J_mean": float(np.mean(pos)),
                    "negative_J_mean": float(np.mean(neg)),
                    "net_J_mean": float(np.mean(pos) + np.mean(neg)),
                    "positive_J_per_stance": pos,
                    "negative_J_per_stance": neg}
    return out


def peak_timing(x, t, st):
    pk, tm = [], []
    for a, b in st:
        i = a + int(np.nanargmax(x[a:b + 1]))
        pk.append(float(x[i]))
        tm.append(float((t[i] - t[a]) / (t[b] - t[a]) * 100))
    return {"peak_N_mean": float(np.mean(pk)), "peak_N_max": float(np.max(pk)),
            "peak_time_pct_stance_mean": float(np.mean(tm)),
            "peak_time_pct_stance": tm}


def rigid_speed():
    """Path lengthening speed of every muscle on the rigid MocoInverse
    solution (study.analyze, cached in compliant/rigid_speed.sto)."""
    f = os.path.join(CDIR, "rigid_speed.sto")
    if not os.path.exists(f):
        inv = osim.MocoInverse.safeDownCast(osim.OpenSimObject.makeObjectFromFile(
            os.path.join(P.OUT, "p1_setup_mocoinverse.xml")))
        study = inv.initialize()
        paths = osim.StdVectorString()
        paths.append(r".*\|speed")
        tbl = study.analyze(osim.MocoTrajectory(RIGID_SOL), paths)
        osim.STOFileAdapter.write(tbl, f)
    return read_sto(f)


def ipopt_log_tail(name):
    """Last IPOPT iteration line of compliant/<name>.log (for a full-mesh
    solve that was stopped before it converged)."""
    f = os.path.join(CDIR, f"{name}.log")
    if not os.path.exists(f):
        return None
    last, status = None, None
    for line in open(f):
        p = line.split()
        if len(p) >= 9 and p[0].isdigit():
            last = p
        if line.startswith("EXIT:") or "success=" in line:
            status = line.strip()
    if last is None:
        return None
    return {"log": f"compliant/{name}.log", "last_iteration": int(last[0]),
            "objective": float(last[1]), "inf_pr": float(last[2]),
            "inf_du": float(last[3]), "lg_mu": float(last[4]),
            "exit": status or "stopped by hand (not converged)"}


def coarse_summary(stage):
    """Side info on a coarse-mesh run of `stage` (a fallback started while
    the full-mesh one was slow), or None."""
    f = os.path.join(CDIR, f"{stage}_coarse_info.json")
    if not os.path.exists(f):
        return None
    i = json.load(open(f))
    return {k: i[k] for k in ("success", "status", "num_iterations",
                              "runtime_s", "objective", "num_mesh_intervals",
                              "mesh_interval_s")}


def source(stage):
    """Which solve stands for `stage`: the full-mesh one if it converged,
    else a converged coarse-mesh fallback (`<stage>_coarse`), else the
    full-mesh one (reported as not converged)."""
    for name in (stage, f"{stage}_coarse"):
        f = os.path.join(CDIR, f"{name}_info.json")
        if os.path.exists(f) and json.load(open(f))["success"]:
            return name
    return stage


def stage_collect():
    os.chdir(P.OUT)
    old = dict(np.load(NPZ, allow_pickle=False))
    t = old["t"]
    muscles = [str(m) for m in old["muscle_names"]]
    r = old["moment_arm"]
    b = np.load(P.BONEID)
    rows = {x["muscle"]: x for x in ratio_table()}
    new = {}
    side_c = {}
    st = stances(b["fz_r"], t, P.WINDOW[0], P.WINDOW[1])

    # rigid tendon/fibre velocities for the work split (tendon_velocity = 0)
    ref = {"dyn": {"force": old["dyn_force"],
                   "vT": np.zeros_like(old["dyn_force"]),
                   "vM": None}}
    summ_side = {}
    for stage in STAGES:
        src = source(stage)
        info_f = os.path.join(CDIR, f"{src}_info.json")
        out_f = os.path.join(CDIR, f"{src}_outputs.sto")
        if not (os.path.exists(info_f) and os.path.exists(out_f)):
            log(f"{stage}: no solution yet, skipped")
            continue
        info = json.load(open(info_f))
        ts, lab, O = read_sto(out_f)

        def col(o):
            return interp(t, ts, O[:, [lab.index(f"/forceset/{m}|{o}")
                                       for m in muscles]])
        F = col("tendon_force")
        D = {"force": F, "activation": col("activation"),
             "fiber_length_norm": col("normalized_fiber_length"),
             "fiber_velocity_norm": col("normalized_fiber_velocity"),
             "tendon_length": col("tendon_length"),
             "tendon_strain": col("tendon_strain"),
             "moment": np.einsum("tm,tmk->tk", F, r)}
        mask = np.array([m in info["compliant_muscles"] for m in muscles])
        if info["success"]:
            for k, v in D.items():
                new[f"{stage}_{k}"] = v
            new[f"compliant_mask_{stage}"] = mask
        # MTU speed from the path (same as the rigid one), tendon speed from
        # the solution; the fibre part is the remainder, so MTU = tendon + fibre
        vT = col("tendon_velocity")
        vM = col("speed") - vT
        # work split over the right triceps surae (+ every right muscle)
        ts_idx = [muscles.index(f"{m}_r") for m in TRICEPS]
        inside = np.isfinite(F[:, 0])
        work = {}
        for name, idx in (("triceps_surae_r", ts_idx),
                          ("soleus_r", [muscles.index("soleus_r")]),
                          ("gastrocnemius_r", ts_idx[1:])):
            work[name] = work_split(np.nan_to_num(F[:, idx]),
                                    np.nan_to_num(vT[:, idx]),
                                    np.nan_to_num(vM[:, idx]), t, st)
        timing = {m: peak_timing(F[:, muscles.index(m)], t, st)
                  for m in ("soleus_r", "gasmed_r", "gaslat_r", "tibant_r",
                            "vaslat_r", "vasmed_r", "vasint_r", "recfem_r",
                            "glmax2_r", "glmed1_r", "semimem_r", "bflh_r")}
        rng = {}
        for m in ("soleus_r", "gasmed_r", "gaslat_r", "vaslat_r", "tibant_r"):
            i = muscles.index(m)
            rng[m] = {"lM_norm": [float(np.nanmin(D["fiber_length_norm"][:, i])),
                                  float(np.nanmax(D["fiber_length_norm"][:, i]))],
                      "vM_norm": [float(np.nanmin(D["fiber_velocity_norm"][:, i])),
                                  float(np.nanmax(D["fiber_velocity_norm"][:, i]))],
                      "tendon_strain_max": float(np.nanmax(D["tendon_strain"][:, i])),
                      "tendon_stretch_max_mm": float(1000 * (np.nanmax(
                          D["tendon_length"][:, i]) - rows[m]["l_slack_m"]))}
        # reserve / residual usage from the solution controls
        sol = osim.MocoTrajectory(os.path.join(CDIR, f"{src}_solution.sto"))
        cn = list(sol.getControlNames())
        C = np.asarray(sol.getControlsTrajectoryMat())
        m_ = osim.Model(P.PROCESSED)
        opt = {}
        for i in range(m_.getForceSet().getSize()):
            f = m_.getForceSet().get(i)
            ca = osim.CoordinateActuator.safeDownCast(f)
            if ca is not None:
                opt["/forceset/" + f.getName()] = ca.getOptimalForce()
        res = {n: {"max_abs": float(np.abs(C[:, cn.index(n)] * opt[n]).max()),
                   "rms": float(np.sqrt(np.mean((C[:, cn.index(n)]
                                                 * opt[n]) ** 2)))}
               for n in cn if ("reserve" in n or "residual" in n)}
        id_os = old["id_moment_opensim"]
        ok = np.isfinite(D["moment"][:, 0])
        mom = {j: {"rmse_Nm": float(np.sqrt(np.mean(
                    (D["moment"][ok, k] - id_os[ok, k]) ** 2))),
                   "corr": float(np.corrcoef(D["moment"][ok, k],
                                             id_os[ok, k])[0, 1])}
               for k, j in enumerate(["ankle", "knee", "hip"])}
        side_c[stage] = {
            **{k: v for k, v in info.items()},
            "converged": bool(info["success"]),
            "source_solve": src,
            "full_mesh_attempt": (None if src == stage else
                                  ipopt_log_tail(stage)),
            "coarse_mesh_run": coarse_summary(stage),
            "stored_in_npz": bool(info["success"]),
            "reserve_and_residual_usage_N_or_Nm": res,
            "muscle_moment_vs_opensim_id": mom,
            "peak_force_timing_right": timing,
            "operating_range_right": rng,
            "stance_work_J": work,
        }
        summ_side[stage] = (info["success"], info["num_iterations"],
                            info["runtime_s"])
    # rigid reference numbers (same definitions) for the comparison
    F0 = old["dyn_force"]
    rigid = {"peak_force_timing_right": {
        m: peak_timing(F0[:, muscles.index(m)], t, st)
        for m in ("soleus_r", "gasmed_r", "gaslat_r", "tibant_r", "vaslat_r",
                  "vasmed_r", "vasint_r", "recfem_r", "glmax2_r", "glmed1_r",
                  "semimem_r", "bflh_r")}}
    # rigid fibre velocity along the tendon = MTU (path) speed
    ts0, lab0, S0 = rigid_speed()
    vM0 = interp(t, ts0, S0[:, [lab0.index(f"/forceset/{m}|speed")
                                for m in muscles]])
    ts_idx = [muscles.index(f"{m}_r") for m in TRICEPS]
    rigid["stance_work_J"] = {
        "triceps_surae_r": work_split(np.nan_to_num(F0[:, ts_idx]),
                                      np.zeros((len(t), 3)),
                                      np.nan_to_num(vM0[:, ts_idx]), t, st)}
    rigid["operating_range_right"] = {
        m: {"lM_norm": [float(np.nanmin(old["fiber_length_norm"][:, i])),
                        float(np.nanmax(old["fiber_length_norm"][:, i]))],
            "vM_norm": [float(np.nanmin(old["fiber_velocity_norm"][:, i])),
                        float(np.nanmax(old["fiber_velocity_norm"][:, i]))]}
        for m in ("soleus_r", "gasmed_r", "gaslat_r", "vaslat_r", "tibant_r")
        for i in [muscles.index(m)]}
    # cross-check: C1 from MocoInverse's default (cold) guess, if it was run
    cold = None
    cf = os.path.join(CDIR, "c1_cold_info.json")
    if os.path.exists(cf) and os.path.exists(os.path.join(CDIR, "c1_outputs.sto")):
        ci = json.load(open(cf))
        _, la, A = read_sto(os.path.join(CDIR, "c1_outputs.sto"))
        _, lb, B = read_sto(os.path.join(CDIR, "c1_cold_outputs.sto"))
        cols = [l for l in la if l.endswith("_r|tendon_force")]
        d = A[:, [la.index(c) for c in cols]] - B[:, [lb.index(c) for c in cols]]
        cold = {"success": ci["success"], "num_iterations": ci["num_iterations"],
                "runtime_s": ci["runtime_s"], "objective": ci["objective"],
                "right_tendon_force_diff_vs_warm_N": {
                    "rms": float(np.sqrt(np.mean(d ** 2))),
                    "max_abs": float(np.abs(d).max())}}
    np.savez_compressed(NPZ, **{**old, **new})
    with open(JSN) as f:
        side = json.load(f)
    side["tendon_compliance"] = {
        "script": "scripts/opensim/compliance.py",
        "keys": {
            "c{1,2,3}_force": "[T,M] N, tendon force (MocoInverse output)",
            "c{1,2,3}_activation": "[T,M]",
            "c{1,2,3}_fiber_length_norm": "[T,M] l_M / l_opt",
            "c{1,2,3}_fiber_velocity_norm": "[T,M] v_M / (v_max l_opt), + = lengthening",
            "c{1,2,3}_tendon_length": "[T,M] m",
            "c{1,2,3}_tendon_strain": "[T,M] (l_T - l_slack) / l_slack",
            "c{1,2,3}_moment": "[T,3] N m, sum force x the rigid moment_arm (same geometry)",
            "compliant_mask_c{1,2,3}": "[M] bool, right-leg muscles with a compliant tendon",
        },
        "ladder": {"c1": "compliant soleus/gasmed/gaslat, both legs",
                   "c2": f"compliant where l_slack / l_opt > {RATIO_THRESHOLD}",
                   "c3": "all muscles compliant + passive fibre force on"},
        "ratio_threshold_justification": (
            "Zajac (1989) shows the tendon-to-fibre length ratio l_slack/l_opt "
            "sets how much tendon stretch changes fibre length and velocity: "
            "tendon stretch at F_iso is eps0 l_slack = 0.049 l_slack, so at a "
            "ratio of 1 a fully loaded tendon shifts the fibre by ~5% of l_opt "
            "(small vs the active f-L width), while the triceps surae (ratio "
            "5.4-6.6) shift by 27-32% of l_opt. Below ~1 a rigid tendon is a "
            "good approximation (also the rule of thumb in Millard 2013 / "
            "De Groote 2016: compliance matters for long-tendon muscles)."),
        "model_parameters_all_80": [
            {k: v for k, v in x.items()} for x in ratio_table()],
        "tendon_stiffness_note": (
            "every muscle has tendon_strain_at_one_norm_force = 0.049 (DGF "
            "default; the Rajagopal-Lai-Uhlrich Millard muscles use the "
            "default tendon curve, so nothing like the 0.10 Achilles strain "
            "some versions use). kT = ln(6)/0.049 = 36.6; normalized "
            "stiffness at F_iso = 0.2 kT 6 = 43.9, i.e. k = 43.9 F_iso / "
            "l_slack N/m (soleus_r ~957 kN/m at F_iso)."),
        "stages": side_c,
        "c1_cold_start_check": cold,
        "rigid_reference": rigid,
        "stances": {"frames": [list(map(int, s)) for s in st],
                    "definition": "right Fz > 20 N, complete stances in the window"},
        "work_definition": (
            "power delivered to the skeleton P = -F v (v + = lengthening); "
            "tendon P = -F dl_T/dt (tendon_velocity output), fibre P = -F "
            "v_M along the tendon; positive/negative parts integrated over "
            "each complete right stance, mean over stances"),
    }
    with open(JSN, "w") as f:
        json.dump(side, f, indent=2, default=float)
    log(f"collected: {sorted(new)}")
    log(json.dumps(summ_side))


# ---------------------------------------------------------------------------
# knee contact for each converged stage
# ---------------------------------------------------------------------------

def stage_contact():
    import knee_contact as K
    import knee_collect as KC
    os.chdir(P.OUT)
    b = np.load(P.BONEID)
    t = b["t"]
    bw = float(b["body_mass"]) * KC.G
    d_ic, _ = K.d_ic_scaled()
    muscles = K.right_muscles(P.PROCESSED)
    tso = read_sto("so/p1_trial13_so_StaticOptimization_force.sto")[0]
    sets, inside = {}, {}
    for stage in STAGES:
        src = source(stage)
        info_f = os.path.join(CDIR, f"{src}_info.json")
        if not os.path.exists(info_f) or not json.load(open(info_f))["success"]:
            continue
        ts, lab, O = read_sto(os.path.join(CDIR, f"{src}_outputs.sto"))
        ins = (tso >= ts[0] - 1e-9) & (tso <= ts[-1] + 1e-9)
        sets[stage] = {m: np.where(ins, np.interp(
            tso, ts, O[:, lab.index(f"/forceset/{m}|tendon_force")]), 0.0)
            for m in muscles}
        inside[stage] = ins
    if not sets:
        log("no converged stage")
        return
    t0 = time.time()
    W, _ = K.knee_wrench(P.PROCESSED, P.IK, tso, sets)
    log(f"knee wrench for {list(sets)} in {time.time() - t0:.0f} s")
    old = dict(np.load(KNPZ, allow_pickle=False))
    st = [tuple(x) for x in old["stance_frames"]]
    new, summ = {}, {}
    for stage in sets:
        sp = K.split(W[stage][0], W[stage][1], d_ic)
        for k in ("total", "medial", "lateral", "lig_lateral"):
            x = np.where(inside[stage], sp[k], np.nan) / bw
            key = (f"{stage}_contact_{k}" if k != "lig_lateral"
                   else f"{stage}_lig_lateral")
            new[key] = KC.on_clock(t, tso, x)
        tot = new[f"{stage}_contact_total"]
        summ[stage] = {
            "peak_total_BW": KC.stance_peaks(tot, st),
            "peak_medial_BW": KC.stance_peaks(new[f"{stage}_contact_medial"], st),
            "peak_lateral_BW": KC.stance_peaks(new[f"{stage}_contact_lateral"], st),
        }
    np.savez(KNPZ, **{**old, **new})
    with open(KJSN) as f:
        side = json.load(f)
    side["tendon_compliance"] = {
        "script": "scripts/opensim/compliance.py contact",
        "keys": "c{1,2,3}_contact_{total,medial,lateral}, c{1,2,3}_lig_lateral (BW)",
        "stages": summ,
        "note": "same Newton-Euler + two-point split as the baseline, with "
                "the compliant MocoInverse tendon forces (only converged "
                "stages)"}
    with open(KJSN, "w") as f:
        json.dump(side, f, indent=2, default=float)
    log(json.dumps(summ, indent=1))


def stage_list():
    rows = ratio_table()
    for x in sorted(rows, key=lambda r: -r["ratio"]):
        if x["muscle"].endswith("_r"):
            print(f"{x['muscle']:14s} ratio {x['ratio']:5.2f}  l_s "
                  f"{x['l_slack_m']:.3f}  l_o {x['l_opt_m']:.3f}  eps0 "
                  f"{x['tendon_strain_at_one_norm_force']:.3f}  k@Fiso "
                  f"{x['stiffness_at_f_iso_N_per_m'] / 1000:.0f} kN/m  "
                  f"stretch@Fiso {x['tendon_stretch_at_f_iso_mm']:.1f} mm")
    for s in STAGES:
        n = compliant_names(s, rows)
        print(s, len(n), [m for m in n if m.endswith("_r")])


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=")
            kw[k] = (v if k in ("tag", "guess", "prev") else
                     int(v) if k in ("max_iter", "eqgoal") else float(v))
    if cmd in STAGES:
        stage_solve(cmd, **kw)
    elif cmd == "analyze":
        os.chdir(P.OUT)
        analyze(sys.argv[2])
    elif cmd == "collect":
        stage_collect()
    elif cmd == "contact":
        stage_contact()
    elif cmd == "list":
        stage_list()
