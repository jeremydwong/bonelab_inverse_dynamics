"""Right tibiofemoral contact forces (SO + MocoInverse) and a synthetic varus sweep.

Run in the conda env `opensim` (like pipeline.py), cwd anywhere:

    PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
    $PY scripts/opensim/knee_contact.py baseline          # SO + dyn on the existing solves
    $PY scripts/opensim/knee_contact.py step 0 5 ... 50   # varus sweep steps (mm), one process each is fine
    $PY scripts/opensim/knee_contact.py moco 25 50        # MocoInverse on sweep steps (~18 min each)
    uv run python scripts/opensim/knee_kam_boneid.py      # boneid KAM cross-check (uv env)
    $PY scripts/opensim/knee_contact.py collect           # -> data/opensim/p1_trial13_knee_contact.{npz,json}

KNEE REACTION (no OpenSim JointReaction analysis; that one realizes forward
dynamics, so its accelerations are whatever the forces produce).  Here the
reaction is a plain Newton-Euler balance of the subtree BELOW the knee
(tibia_r + talus_r + calcn_r + toes_r; subtalar/mtp are welded in the
processed model):

    reaction on tibia from femur = sum(m a_com, I alpha + w x I w)   (IK accelerations)
                                   - gravity - GRF (+free moment) - muscle path forces

taken about the walker-knee child-frame origin (the knee centre on the tibia)
and expressed in the tibia_r body frame (X anterior, Y up the long axis,
Z lateral for the right leg).  Muscle forces enter as tension x unit vector
on every path segment that touches a subtree body (wrap points included), so
muscles wholly inside the subtree cancel.  q, u, udot come from the 12 Hz
low-passed IK (same filter as ID/SO/MocoInverse), quintic interpolating
spline.  Reserve/coordinate actuators are generalized forces along the knee
DOF and carry nothing across the contact (the knee reserve is reported).

CONTACT MODEL: two point contacts on a flat plateau perpendicular to the
tibial Y axis, at z = -/+ d_ic/2 (medial/lateral) through the knee centre.
Ligaments carry nothing.  With compressive force Fc = -F_y and frontal
reaction moment M_x (about tibial X):
    F_med + F_lat = Fc,   (F_med - F_lat) d_ic/2 = -M_x
If a condyle comes out negative it is reported as lift-off: its contact
force is clamped to 0 and the tension a collateral ligament at that condyle
would need (= -raw value) is reported separately; the other condyle keeps
its raw value (moment balance about it is unchanged).

VARUS SWEEP: per d, the right knee marker cluster is moved laterally by d
(perpendicular to the hip-ankle line, lateral = epicondyle axis), thigh
markers rigidly rotated about the hip, shank markers about the ankle centre
(hip, ankle, foot, GRF untouched) -- the AFF notebook's varus_shift on
markers.  The model gets a matching fixed varus at the knee: the walker
knee's child (tibia) offset frame is rotated by theta = d (1/L_thigh +
1/L_shank) about the tibial X (anterior) axis through the knee centre, so the
tibia sits in varus relative to the femur.  Then IK, ID, SO, contact.
"""

from __future__ import annotations

import glob
import json
import os

import sys
import time

import numpy as np
import opensim as osim
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

osim.Logger.setLevelString("error")

OUT = P.OUT
VARUS = os.path.join(OUT, "varus")
GRF_ABS = os.path.join(P.WORK, "p1_trial13_grf.mot")
TRC_ABS = os.path.join(P.WORK, "p1_trial13_markers.trc")
SUBTREE = ("tibia_r", "talus_r", "calcn_r", "toes_r")
KNEE = "walker_knee_r"
D_IC_GENERIC = 0.040          # m (Lerner 2015 generic +-20 mm; Gerus 2013 40 mm)
D_MM = list(range(0, 55, 5))
THIGH_MK = ["RGTR", "RTH1", "RTH2", "RTH3", "RLEP", "RMEP"]
SHANK_MK = ["RSH1", "RSH2", "RSH3", "RLML", "RMML"]


def log(msg):
    print(f"[knee] {msg}", flush=True)


def mat(R):
    return np.array([[R.get(i, j) for j in range(3)] for i in range(3)])


def vec(v):
    return np.array([v.get(0), v.get(1), v.get(2)])


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def read_sto(path):
    tbl = osim.TimeSeriesTable(path)
    return (np.array(tbl.getIndependentColumn()),
            list(tbl.getColumnLabels()), tbl.getMatrix().to_numpy())


# ---------------------------------------------------------------------------
# kinematics + external load on the frame clock
# ---------------------------------------------------------------------------

def coordinate_splines(model, ik_file):
    """{coord: (q, u, udot) callables} from the 12 Hz low-passed IK, radians."""
    tp = osim.TableProcessor(ik_file)
    tp.append(osim.TabOpLowPassFilter(P.LOWPASS_HZ))
    tbl = tp.process()
    tk = np.array(tbl.getIndependentColumn())
    labels = list(tbl.getColumnLabels())
    data = tbl.getMatrix().to_numpy()
    cs = model.getCoordinateSet()
    out = {}
    for j, lab in enumerate(labels):
        if not cs.contains(lab):
            continue
        v = data[:, j].copy()
        if cs.get(lab).getMotionType() == 1:
            v = np.deg2rad(v)
        sp = make_interp_spline(tk, v, k=5)
        out[lab] = sp
    return out


def grf_r(t):
    tg, lab, G = read_sto(GRF_ABS)

    def col(n):
        return np.interp(t, tg, G[:, lab.index(n)])
    f = np.stack([col(f"ground_force_r_v{a}") for a in "xyz"], 1)
    p = np.stack([col(f"ground_force_r_p{a}") for a in "xyz"], 1)
    m = np.stack([col(f"ground_torque_r_{a}") for a in "xyz"], 1)
    return f, p, m


def knee_wrench(model_file, ik_file, t, force_sets):
    """Reaction on the tibia from the femur at the knee centre, tibia frame.

    Returns {"ext": (F[T,3], M[T,3])  -- inertia + gravity + GRF only (the net
    intersegmental load, what ID sees), name: (F, M) for each force set
    (dict muscle -> [T] tension) with that set's muscle forces included}, and
    geometry per frame (hip, knee, ankle centres in ground; tibia/femur axes).
    """
    model = osim.Model(model_file)
    s = model.initSystem()
    mat_sub = model.getMatterSubsystem()
    cs = model.getCoordinateSet()
    sp = coordinate_splines(model, ik_file)
    coords = [cs.get(i) for i in range(cs.getSize())]
    # coordinate -> u index
    uidx = {}
    for c in coords:
        for k in coords:
            k.setSpeedValue(s, 0.0)
        c.setSpeedValue(s, 1.0)
        uidx[c.getName()] = int(np.argmax(np.abs(s.getU().to_numpy())))
    bodies = [model.getBodySet().get(b) for b in SUBTREE]
    mbi = [int(b.getMobilizedBodyIndex()) for b in bodies]
    mass = [b.getMass() for b in bodies]
    com_b = [vec(b.getMassCenter()) for b in bodies]
    inert = []
    for b in bodies:
        I = b.getInertia()
        mo, pr = vec(I.getMoments()), vec(I.getProducts())
        inert.append(np.array([[mo[0], pr[0], pr[1]], [pr[0], mo[1], pr[2]],
                               [pr[1], pr[2], mo[2]]]))
    g = vec(model.getGravity())
    grf_f, grf_p, grf_m = grf_r(t)
    knee_child = model.getJointSet().get(KNEE).getChildFrame()
    ankle_parent = model.getJointSet().get("ankle_r").getParentFrame()
    femur = model.getBodySet().get("femur_r")
    tibia = model.getBodySet().get("tibia_r")
    muscles_all = set()
    for fs in force_sets.values():
        muscles_all |= set(fs)
    mus = {n: model.getMuscles().get(n) for n in muscles_all}
    T = len(t)
    ext_F, ext_M = np.zeros((T, 3)), np.zeros((T, 3))
    setF = {k: np.zeros((T, 3)) for k in force_sets}
    setM = {k: np.zeros((T, 3)) for k in force_sets}
    geo = {k: np.zeros((T, 3)) for k in ("hip", "knee", "ankle", "femur_z",
                                          "tibia_x")}
    A = osim.VectorOfSpatialVec()
    for i, ti in enumerate(t):
        s.setTime(float(ti))
        ud = np.zeros(s.getNU())
        for c in coords:
            f = sp.get(c.getName())
            if f is None:
                continue
            c.setValue(s, float(f(ti)), False)
            c.setSpeedValue(s, float(f(ti, 1)))
            ud[uidx[c.getName()]] = float(f(ti, 2))
        model.realizeVelocity(s)
        mat_sub.calcBodyAccelerationFromUDot(s, osim.Vector(ud.tolist()), A)
        P0 = vec(knee_child.getPositionInGround(s))
        RT = mat(tibia.getTransformInGround(s).R())
        F = np.zeros(3)
        M = np.zeros(3)
        for k, b in enumerate(bodies):
            X = b.getTransformInGround(s)
            R, o = mat(X.R()), vec(X.p())
            w = vec(b.getAngularVelocityInGround(s))
            sv = A.get(mbi[k])
            al, ao = vec(sv.get(0)), vec(sv.get(1))
            r = R @ com_b[k]
            c = o + r
            ac = ao + np.cross(al, r) + np.cross(w, np.cross(w, r))
            IG = R @ inert[k] @ R.T
            # reaction = inertial - applied (gravity)
            F += mass[k] * ac - mass[k] * g
            M += (IG @ al + np.cross(w, IG @ w)
                  + np.cross(c - P0, mass[k] * ac)
                  - np.cross(c - P0, mass[k] * g))
        F -= grf_f[i]
        M -= np.cross(grf_p[i] - P0, grf_f[i]) + grf_m[i]
        ext_F[i], ext_M[i] = RT.T @ F, RT.T @ M
        # muscle path forces on subtree bodies, per unit tension
        unitF, unitM = {}, {}
        for n, mu in mus.items():
            cp = mu.getGeometryPath().getCurrentPath(s)
            pts, onsub = [], []
            for j in range(cp.getSize()):
                pp = cp.get(j)
                pts.append(vec(pp.getLocationInGround(s)))
                onsub.append(pp.getParentFrame().findBaseFrame().getName()
                             in SUBTREE)
            fu, mu_ = np.zeros(3), np.zeros(3)
            for j in range(len(pts) - 1):
                d = pts[j + 1] - pts[j]
                L = np.linalg.norm(d)
                if L < 1e-9:
                    continue
                u = d / L
                if onsub[j]:
                    fu += u
                    mu_ += np.cross(pts[j] - P0, u)
                if onsub[j + 1]:
                    fu -= u
                    mu_ -= np.cross(pts[j + 1] - P0, u)
            unitF[n], unitM[n] = fu, mu_
        for k, fs in force_sets.items():
            Fm = sum(fs[n][i] * unitF[n] for n in fs)
            Mm = sum(fs[n][i] * unitM[n] for n in fs)
            setF[k][i] = RT.T @ (F - Fm)
            setM[k][i] = RT.T @ (M - Mm)
        geo["hip"][i] = vec(femur.getPositionInGround(s))
        geo["knee"][i] = P0
        geo["ankle"][i] = vec(ankle_parent.getPositionInGround(s))
        geo["femur_z"][i] = mat(femur.getTransformInGround(s).R())[:, 2]
        geo["tibia_x"][i] = RT[:, 0]
    out = {"ext": (ext_F, ext_M)}
    out.update({k: (setF[k], setM[k]) for k in force_sets})
    return out, geo


def split(F, M, d_ic):
    """Medial/lateral point contacts, ligaments carry nothing (see module doc)."""
    fc = -F[:, 1]
    diff = -M[:, 0] / (0.5 * d_ic)
    med, lat = 0.5 * (fc + diff), 0.5 * (fc - diff)
    return {"total": fc, "medial_raw": med, "lateral_raw": lat,
            "medial": np.maximum(med, 0.0), "lateral": np.maximum(lat, 0.0),
            "lig_lateral": np.maximum(-lat, 0.0),
            "lig_medial": np.maximum(-med, 0.0),
            "frontal_moment": M[:, 0], "ap_shear": F[:, 0],
            "ml_shear": F[:, 2]}


def so_forces(sto, t, muscles):
    ts, lab, F = read_sto(sto)
    return {m: np.interp(t, ts, F[:, lab.index(m)]) for m in muscles}, ts


def dyn_forces(outputs_sto, t, muscles):
    tm, lab, O = read_sto(outputs_sto)
    return {m: np.interp(t, tm, O[:, lab.index(f"/forceset/{m}|tendon_force")])
            for m in muscles}, tm


def right_muscles(model_file):
    m = osim.Model(model_file)
    return [m.getMuscles().get(i).getName()
            for i in range(m.getMuscles().getSize())
            if m.getMuscles().get(i).getName().endswith("_r")]


def d_ic_scaled():
    summ = P.load_summary()
    f = summ["scale"]["factors_uniform"]["tibia_r"]
    return D_IC_GENERIC * f, f


def frontal_offset(geo):
    """Signed knee-centre distance from the hip-ankle line, lateral +, [T] m."""
    a = unit(geo["hip"] - geo["ankle"])
    v = geo["knee"] - geo["ankle"]
    perp = v - a * np.sum(v * a, 1, keepdims=True)
    lat = geo["femur_z"] - a * np.sum(geo["femur_z"] * a, 1, keepdims=True)
    return np.sum(perp * unit(lat), 1)


# ---------------------------------------------------------------------------
# baseline: existing SO + MocoInverse solves
# ---------------------------------------------------------------------------

def stage_baseline():
    os.chdir(OUT)
    t0 = time.time()
    muscles = right_muscles(P.PROCESSED)
    tso = read_sto("so/p1_trial13_so_StaticOptimization_force.sto")[0]
    so, _ = so_forces("so/p1_trial13_so_StaticOptimization_force.sto", tso,
                      muscles)
    dyn, tm = dyn_forces("p1_trial13_mocoinverse_outputs.sto", tso, muscles)
    inside = (tso >= tm[0] - 1e-9) & (tso <= tm[-1] + 1e-9)
    for m in dyn:
        dyn[m] = np.where(inside, dyn[m], 0.0)
    W, geo = knee_wrench(P.PROCESSED, P.IK, tso, {"so": so, "dyn": dyn})
    np.savez(os.path.join(VARUS, "baseline_wrench.npz"), t=tso,
             dyn_inside=inside,
             **{f"{k}_F": v[0] for k, v in W.items()},
             **{f"{k}_M": v[1] for k, v in W.items()},
             **{f"geo_{k}": v for k, v in geo.items()})
    log(f"baseline wrench done in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------
# varus sweep
# ---------------------------------------------------------------------------

def min_rotation(a, b):
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    vx = np.zeros(a.shape[:-1] + (3, 3))
    vx[..., 0, 1], vx[..., 0, 2] = -v[..., 2], v[..., 1]
    vx[..., 1, 0], vx[..., 1, 2] = v[..., 2], -v[..., 0]
    vx[..., 2, 0], vx[..., 2, 1] = -v[..., 1], v[..., 0]
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))[..., None, None]


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


def shifted_markers(d, out_path):
    """Move the right knee laterally by d (m); returns data lengths + shift."""
    tbl = osim.TimeSeriesTableVec3(TRC_ABS)
    names = list(tbl.getColumnLabels())
    t = np.array(tbl.getIndependentColumn())
    X = {n: P.trc_array(tbl, n) for n in names}
    hip = X["RHJC"]
    ankle = 0.5 * (X["RLML"] + X["RMML"])
    knee = 0.5 * (X["RLEP"] + X["RMEP"])
    a = unit(hip - ankle)
    lat = X["RLEP"] - X["RMEP"]
    lat = unit(lat - a * np.sum(lat * a, 1, keepdims=True))
    knee_new = knee + d * lat
    Rt = min_rotation(unit(knee - hip), unit(knee_new - hip))
    Rs = min_rotation(unit(knee - ankle), unit(knee_new - ankle))
    Y = dict(X)
    for n in THIGH_MK:
        Y[n] = hip + np.einsum("tij,tj->ti", Rt, X[n] - hip)
    for n in SHANK_MK:
        Y[n] = ankle + np.einsum("tij,tj->ti", Rs, X[n] - ankle)
    write_trc(out_path, t, names, np.stack([Y[n] for n in names], 1),
              1.0 / (t[1] - t[0]))
    knee_moved = 0.5 * (Y["RLEP"] + Y["RMEP"])
    L_t = float(np.median(np.linalg.norm(hip - knee, axis=1)))
    L_s = float(np.median(np.linalg.norm(knee - ankle, axis=1)))
    shift = np.sum((knee_moved - knee) * lat, 1)
    return {"L_thigh_m": L_t, "L_shank_m": L_s,
            "marker_knee_lateral_shift_mean_m": float(shift.mean()),
            "marker_knee_lateral_shift_min_m": float(shift.min()),
            "marker_knee_lateral_shift_max_m": float(shift.max())}


def deform_model(src, dst, theta):
    """Rotate the walker knee's tibia offset frame by theta about tibial X
    (through the knee centre): tibia in varus (+theta) relative to the femur."""
    m = osim.Model(src)
    f = osim.PhysicalOffsetFrame.safeDownCast(
        m.updJointSet().get(KNEE).upd_frames(1))
    r = osim.Rotation()
    r.setRotationToBodyFixedXYZ(f.get_orientation())
    R_O = mat(r)
    c, s_ = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    Rn = Rx.T @ R_O
    b = np.arcsin(np.clip(Rn[0, 2], -1, 1))
    a = np.arctan2(-Rn[1, 2], Rn[2, 2])
    g = np.arctan2(-Rn[0, 1], Rn[0, 0])
    f.set_orientation(osim.Vec3(a, b, g))
    m.finalizeConnections()
    m.printToXML(dst)
    chk = osim.Rotation()
    chk.setRotationToBodyFixedXYZ(osim.Vec3(a, b, g))
    assert np.allclose(mat(chk), Rn, atol=1e-9)


def ankle_shift_default(model_file):
    m = osim.Model(model_file)
    s = m.initSystem()
    return vec(m.getJointSet().get("ankle_r").getParentFrame()
               .getPositionInGround(s))


def stage_step(d_mm):
    d = d_mm / 1000.0
    step = os.path.join(VARUS, f"d{d_mm:02d}")
    os.makedirs(step, exist_ok=True)
    os.chdir(step)
    P.save_summary = lambda _d: None
    P.load_summary = lambda: {}
    info = {"d_m": d}
    t0 = time.time()
    info.update(shifted_markers(d, os.path.join(step, "markers.trc")))
    theta = d * (1.0 / info["L_thigh_m"] + 1.0 / info["L_shank_m"])
    info["theta_rad"] = theta
    info["theta_deg"] = float(np.degrees(theta))
    deform_model(os.path.join(OUT, P.SCALED), "scaled_varus.osim", theta)
    a0 = ankle_shift_default(os.path.join(OUT, P.SCALED))
    a1 = ankle_shift_default("scaled_varus.osim")
    info["default_pose_ankle_shift_m"] = (a1 - a0).tolist()  # -Z = medial
    P.TRC = os.path.join(step, "markers.trc")
    P.GRF = GRF_ABS
    P.SCALED = "scaled_varus.osim"
    P.IK = "ik.mot"
    P.ID = "id.sto"
    P.PROCESSED = "processed.osim"
    P.EXT_LOADS = "external_loads.xml"
    w2 = {n: P.WEIGHTS.get(n, 1.0) for n in P.marker_table()}
    t1 = time.time()
    P.run_ik(P.SCALED, P.IK, "setup_ik.xml", w2)
    info["ik"] = P.ik_errors("ik_ik_marker_errors.sto")
    info["ik"]["runtime_s"] = time.time() - t1
    t1 = time.time()
    P.stage_id()
    info["id_runtime_s"] = time.time() - t1
    t1 = time.time()
    P.stage_so()
    info["so_runtime_s"] = time.time() - t1
    t1 = time.time()
    muscles = right_muscles(P.PROCESSED)
    sto = "so/p1_trial13_so_StaticOptimization_force.sto"
    tso = read_sto(sto)[0]
    so, _ = so_forces(sto, tso, muscles)
    W, geo = knee_wrench(P.PROCESSED, P.IK, tso, {"so": so})
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
    log(f"d={d_mm} mm: theta {info['theta_deg']:.2f} deg, IK RMS "
        f"{info['ik']['rms_mean_m']*1000:.1f} mm, {info['runtime_s']:.0f} s")


def stage_moco_step(d_mm):
    step = os.path.join(VARUS, f"d{d_mm:02d}")
    os.chdir(step)
    P.save_summary = lambda _d: None
    P.load_summary = lambda: {}
    P.GRF = GRF_ABS
    P.SCALED = "scaled_varus.osim"
    P.IK = "ik.mot"
    P.EXT_LOADS = "external_loads.xml"
    t0 = time.time()
    P.stage_moco(tag="")
    rt = time.time() - t0
    sol = osim.MocoTrajectory("p1_trial13_mocoinverse_solution.sto")
    muscles = right_muscles("processed.osim")
    tso = read_sto("so/p1_trial13_so_StaticOptimization_force.sto")[0]
    dyn, tm = dyn_forces("p1_trial13_mocoinverse_outputs.sto", tso, muscles)
    W, _ = knee_wrench("processed.osim", "ik.mot", tso, {"dyn": dyn})
    np.savez("wrench_dyn.npz", t=tso, dyn_F=W["dyn"][0], dyn_M=W["dyn"][1],
             window=np.array([tm[0], tm[-1]]))
    with open("info_moco.json", "w") as f:
        json.dump({"runtime_s": rt, "window_s": [float(tm[0]), float(tm[-1])],
                   "num_iterations": int(sol.getNumIterations()) if hasattr(sol, "getNumIterations") else None},
                  f, indent=2)
    log(f"moco d={d_mm}: {rt:.0f} s")


if __name__ == "__main__":
    os.makedirs(VARUS, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "baseline":
        stage_baseline()
    elif cmd == "step":
        for a in sys.argv[2:]:
            stage_step(int(a))
    elif cmd == "moco":
        for a in sys.argv[2:]:
            stage_moco_step(int(a))
    elif cmd == "collect":
        import knee_collect
        knee_collect.main()
