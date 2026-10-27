"""OpenSim pipeline for P1 trial 13: scale -> IK -> ID -> SO -> MocoInverse.

Run in the SEPARATE conda env `opensim` (OpenSim 4.6, osx-arm64), NOT uv:

    PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
    $PY scripts/opensim/pipeline.py scale ik id so moco collect

Inputs come from `uv run python scripts/opensim/export_inputs.py`
(data/opensim/work/*).  Everything is written into data/opensim/ and every
setup XML uses paths relative to that directory, so the setups can be re-run
with the OpenSim GUI / cmd tools from there.

Model: RajagopalLaiUhlrich2023.osim (opensim-models, Rajagopal 2016 + Lai
2017 knee/muscle edits + Uhlrich 2022 passive curves / hip abductors), 80
Millard muscles, arms kept (they have their own CoordinateActuators).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
import opensim as osim

osim.Logger.setLevelString("warn")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "opensim")
WORK = os.path.join(OUT, "work")
GENERIC_URL = ("https://raw.githubusercontent.com/opensim-org/opensim-models/"
               "master/Models/Rajagopal/RajagopalLaiUhlrich2023.osim")
GENERIC = "RajagopalLaiUhlrich2023.osim"
TRC = "work/p1_trial13_markers.trc"
GRF = "work/p1_trial13_grf.mot"
BONEID = os.path.join(WORK, "p1_trial13_boneid.npz")
EXT_LOADS = "p1_trial13_external_loads.xml"
MARKED = "p1_generic_marked.osim"
SCALED0 = "p1_scaled_unplaced.osim"
SCALED = "p1_scaled.osim"
PROCESSED = "p1_scaled_dgf_reserves.osim"
IK1 = "p1_trial13_ik_pass1.mot"
IK = "p1_trial13_ik.mot"
ID = "p1_trial13_id.sto"
LOWPASS_HZ = 12.0                     # same as boneid report.p1_params()
WINDOW = (0.10, 5.20)                 # s; trims IK/filter edges (T ends 5.30)
T_END = 5.3                           # s, IK/ID end time (trial length)
TAG = "p1_trial13"                    # name prefix of ID/SO/external-loads outputs
# (stepfreq.py re-points TRC/GRF/IK/ID/... plus WINDOW, T_END and TAG per
# trial; the defaults above are trial 13 and must stay as they are)
MESH = 0.02                           # s, MocoInverse mesh interval
SUMMARY = os.path.join(WORK, "summary.json")

# data marker -> (body, location in generic-model body frame).  Locations for
# the anatomical markers are copied from the model's own landmark markers
# (RLFC/RMFC epicondyles, RLMAL/RMMAL malleoli, RCAL, RMT5, RACR, RLEL,
# RFAradius, RTB*, RTH*), SACR = mid-PSIS, RHJC/LHJC = femur origin = the
# model's hip centre.  Cluster markers / GTR / arm markers are re-placed from
# the data afterwards (see `place_markers`), so their start values only need
# to be roughly right.
def _r(side, v):
    v = list(v)
    return v if side == "R" else [v[0], v[1], -v[2]]


def marker_table():
    tab = {"RASI": ("pelvis", [0.00953047, 0.0180816, 0.128464]),
           "LASI": ("pelvis", [0.00953047, 0.0180816, -0.128464]),
           "SACR": ("pelvis", [-0.155, 0.035, 0.0])}
    for s in "RL":
        lo = s.lower()
        tab.update({
            f"{s}HJC": (f"femur_{lo}", [0, 0, 0]),
            f"{s}GTR": (f"femur_{lo}", _r(s, [-0.01, -0.03, 0.08])),
            f"{s}TH1": (f"femur_{lo}", _r(s, [0.018, -0.15, 0.064])),
            f"{s}TH2": (f"femur_{lo}", _r(s, [0.08, -0.23, 0.0047])),
            f"{s}TH3": (f"femur_{lo}", _r(s, [0.01, -0.30, 0.06])),
            f"{s}LEP": (f"femur_{lo}", _r(s, [0, -0.404, 0.05])),
            f"{s}MEP": (f"femur_{lo}", _r(s, [0, -0.404, -0.05])),
            f"{s}SH1": (f"tibia_{lo}", _r(s, [-0.0017, -0.1565, 0.0492])),
            f"{s}SH2": (f"tibia_{lo}", _r(s, [0.0370, -0.2301, -0.0039])),
            f"{s}SH3": (f"tibia_{lo}", _r(s, [0.0114, -0.2952, 0.0554])),
            f"{s}LML": (f"tibia_{lo}", _r(s, [-0.005, -0.3888, 0.053])),
            f"{s}MML": (f"tibia_{lo}", _r(s, [0.006, -0.3888, -0.038])),
            f"{s}CAL": (f"calcn_{lo}", _r(s, [-0.025, 0.02, -0.005])),
            f"{s}5TH": (f"calcn_{lo}", _r(s, [0.145, 0.02491, 0.059])),
            f"{s}AC": ("torso", _r(s, [-0.003, 0.425, 0.13])),
            f"{s}EP": (f"humerus_{lo}", _r(s, [0.015, -0.28, 0.04])),
            f"{s}WR": (f"radius_{lo}", _r(s, [0.0005, -0.235, 0.05])),
        })
    return tab


# Markers that DEFINE the anatomy (kept where the generic model puts them);
# everything else is re-placed from the data after a first IK pass.
FIXED = {"RASI", "LASI", "RHJC", "LHJC", "RLEP", "RMEP", "LLEP", "LMEP",
         "RLML", "RMML", "LLML", "LMML"}
# IK weights: joint-defining markers high, clusters 1, arms low.
WEIGHTS = {**{k: 10.0 for k in FIXED}, "SACR": 5.0, "RCAL": 5.0, "R5TH": 5.0,
           "LCAL": 5.0, "L5TH": 5.0, "RAC": 1.0, "LAC": 1.0, "REP": 0.5,
           "LEP": 0.5, "RWR": 0.5, "LWR": 0.5}


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def load_summary():
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            return json.load(f)
    return {}


def save_summary(d):
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=2, default=float)


def trc_array(tbl, name):
    col = tbl.getDependentColumn(name)
    return np.array([[col[i][0], col[i][1], col[i][2]]
                     for i in range(col.size())])


# ---------------------------------------------------------------------------
# 1. scale
# ---------------------------------------------------------------------------

def stage_scale():
    if not os.path.exists(GENERIC):
        log(f"downloading {GENERIC_URL}")
        urllib.request.urlretrieve(GENERIC_URL, GENERIC)
    model = osim.Model(GENERIC)
    ms = model.updMarkerSet()
    ms.clearAndDestroy()
    for name, (body, loc) in marker_table().items():
        mk = osim.Marker(name, model.getBodySet().get(body), osim.Vec3(*loc))
        mk.set_fixed(name in FIXED)
        model.addMarker(mk)
    model.finalizeConnections()
    model.printToXML(MARKED)

    # segment-length measurements: per-frame distances averaged over the
    # walking trial (there is no static trial; averaging marker POSITIONS
    # over a stride would shrink distances on swinging segments, so the
    # ScaleTool's own marker-pair measurement over a time range is not used)
    tbl = osim.TimeSeriesTableVec3(TRC)
    d = {n: trc_array(tbl, n) for n in tbl.getColumnLabels()}
    state = model.initSystem()
    g = {n: np.array(model.getMarkerSet().get(n).getLocationInGround(state)
                     .to_numpy()) for n in marker_table()}

    def mid(src, a, b):
        return 0.5 * (src[a] + src[b])

    def dist(src, a, b):
        a = src[a] if isinstance(a, str) else a
        b = src[b] if isinstance(b, str) else b
        return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)

    def ratio(f):
        return float(np.mean(f(d)) / np.mean(f(g)))

    fac, raw = {}, {}
    fac["pelvis"] = 0.5 * (ratio(lambda s: dist(s, "RASI", "LASI"))
                           + ratio(lambda s: dist(s, "RHJC", "LHJC")))
    for S in "RL":
        s = S.lower()
        fem = ratio(lambda x: dist(x, x[f"{S}HJC"], mid(x, f"{S}LEP", f"{S}MEP")))
        tib = ratio(lambda x: dist(x, mid(x, f"{S}LEP", f"{S}MEP"),
                                   mid(x, f"{S}LML", f"{S}MML")))
        foot = ratio(lambda x: dist(x, f"{S}CAL", f"{S}5TH"))
        hum = ratio(lambda x: dist(x, f"{S}AC", f"{S}EP"))
        fore = ratio(lambda x: dist(x, f"{S}EP", f"{S}WR"))
        # The data's CAL-5TH distance is 1.22-1.26x the model's RCAL-RMT5:
        # the lab's 5TH marker is evidently further forward on the metatarsal
        # than the model's, so foot LENGTH from it would inflate the foot
        # (and the Achilles moment arm) by ~25%.  Feet are scaled with the
        # mean leg factor instead and CAL/5TH are re-placed from the data.
        # Same story for the wrist marker: forearm/hand take the humerus
        # factor.  The raw marker ratios are kept in the summary.
        leg = 0.5 * (fem + tib)
        raw[S] = {"foot_cal_5th": foot, "forearm_ep_wr": fore}
        fac.update({f"femur_{s}": fem, f"patella_{s}": fem,
                    f"tibia_{s}": tib, f"talus_{s}": leg,
                    f"calcn_{s}": leg, f"toes_{s}": leg,
                    f"humerus_{s}": hum, f"ulna_{s}": hum,
                    f"radius_{s}": hum, f"hand_{s}": hum})
    fac["torso"] = ratio(lambda x: dist(x, mid(x, "RAC", "LAC"),
                                        mid(x, "RHJC", "LHJC")))

    mass = float(np.load(BONEID)["body_mass"])
    st = osim.ScaleTool()
    st.setName("p1_scaled")
    st.setSubjectMass(mass)
    st.getGenericModelMaker().setModelFileName(MARKED)
    sc = st.getModelScaler()
    sc.setApply(True)
    order = osim.ArrayStr()
    order.append("manualScale")
    sc.setScalingOrder(order)
    for body, f in fac.items():
        s = osim.Scale()
        s.setName(body)
        s.setSegmentName(body)
        s.setScaleFactors(osim.Vec3(f, f, f))
        s.setApply(True)
        sc.getScaleSet().cloneAndAppend(s)
    sc.setMarkerFileName(TRC)
    tr = osim.ArrayDouble()
    tr.append(0.0)
    tr.append(5.3)
    sc.setTimeRange(tr)
    sc.setPreserveMassDist(True)
    sc.setOutputModelFileName(SCALED0)
    sc.setOutputScaleFileName("p1_scale_factors.xml")
    st.getMarkerPlacer().setApply(False)
    st.printToXML("p1_setup_scale.xml")
    osim.ScaleTool("p1_setup_scale.xml").run()

    scaled = osim.Model(SCALED0)
    s2 = scaled.initSystem()
    summ = load_summary()
    summ["scale"] = {"factors_uniform": fac, "unused_marker_ratios": raw, "body_mass_kg": mass,
                     "model_mass_kg": scaled.getTotalMass(s2),
                     "generic_mass_kg": model.getTotalMass(state),
                     "method": ("manualScale, uniform per body; factor = "
                                "trial-mean per-frame marker distance / "
                                "generic-model distance (pelvis: mean of "
                                "ASIS width and HJC width; femur: HJC to "
                                "mid-epicondyles; tibia: mid-epicondyles to "
                                "mid-malleoli; feet: mean of femur+tibia "
                                "(CAL-5TH ratio 1.2+ rejected, marker not "
                                "at the model's MT5 landmark); torso: "
                                "mid-AC to mid-HJC; arms: AC-EP for the "
                                "whole arm); "
                                "mass distribution preserved")}
    save_summary(summ)
    log(f"scale factors: {json.dumps({k: round(v, 3) for k, v in fac.items()})}")


# ---------------------------------------------------------------------------
# 2. IK (two passes with marker placement in between)
# ---------------------------------------------------------------------------

def run_ik(model_file, out, setup, weights):
    ik = osim.InverseKinematicsTool()
    ik.setName(os.path.splitext(out)[0])
    ik.set_model_file(model_file)
    ik.setMarkerDataFileName(TRC)
    ik.setOutputMotionFileName(out)
    ik.set_results_directory(".")
    ik.set_report_errors(True)
    ik.set_accuracy(1e-5)
    ik.setStartTime(0.0)
    ik.setEndTime(T_END)
    for name, w in weights.items():
        t = osim.IKMarkerTask()
        t.setName(name)
        t.setApply(w > 0)
        t.setWeight(max(w, 0.0))
        ik.getIKTaskSet().cloneAndAppend(t)
    ik.printToXML(setup)
    osim.InverseKinematicsTool(setup).run()


def coords_table_to_states(model, mot):
    """[(state per row)] from an IK .mot (degrees) -- position-level only."""
    tbl = osim.TimeSeriesTable(mot)
    in_deg = tbl.getTableMetaDataAsString("inDegrees").strip() == "yes"
    labels = list(tbl.getColumnLabels())
    data = tbl.getMatrix().to_numpy()
    t = np.array(tbl.getIndependentColumn())
    cs = model.getCoordinateSet()
    cols = []
    for j, lab in enumerate(labels):
        if not cs.contains(lab):
            continue
        c = cs.get(lab)
        v = data[:, j].copy()
        if in_deg and c.getMotionType() == 1:          # Rotational
            v = np.deg2rad(v)
        cols.append((c, v))
    return t, cols


PAIRS = [("RLEP", "RMEP"), ("LLEP", "LMEP"), ("RLML", "RMML"),
         ("LLML", "LMML")]


def place_markers(model_in, ik_in, model_out, names, fix_widths):
    """Move `names` to their trial-mean body-frame position under `ik_in`.

    With `fix_widths`, each fixed anatomical pair (epicondyles, malleoli)
    keeps the MODEL's midpoint and direction (so the model's knee/ankle
    centre is untouched) but takes the DATA's mean marker separation -- skin
    markers sit ~1-2 cm outside the bony landmarks, so the data's widths are
    17-40% larger than the model's bone-surface landmarks.
    """
    model = osim.Model(model_in)
    state = model.initSystem()
    t, cols = coords_table_to_states(model, ik_in)
    tbl = osim.TimeSeriesTableVec3(TRC)
    ground = model.getGround()
    acc = {n: [] for n in names}
    data = {n: trc_array(tbl, n) for n in marker_table()}
    for i in range(len(t)):
        for c, v in cols:
            c.setValue(state, float(v[i]), False)
        model.realizePosition(state)
        for n in names:
            body = model.getMarkerSet().get(n).getParentFrame()
            p = ground.findStationLocationInAnotherFrame(
                state, osim.Vec3(*data[n][i]), body)
            acc[n].append([p.get(0), p.get(1), p.get(2)])
    moved = {}
    ms = model.updMarkerSet()
    for n in names:
        loc = np.mean(acc[n], axis=0)
        old = np.array(ms.get(n).get_location().to_numpy())
        ms.get(n).set_location(osim.Vec3(*loc))
        moved[n] = float(np.linalg.norm(loc - old))
    if fix_widths:
        for a, b in PAIRS:
            pa = np.array(ms.get(a).get_location().to_numpy())
            pb = np.array(ms.get(b).get_location().to_numpy())
            mid, u = 0.5 * (pa + pb), (pa - pb) / np.linalg.norm(pa - pb)
            w = float(np.linalg.norm(data[a] - data[b], axis=1).mean())
            ms.get(a).set_location(osim.Vec3(*(mid + 0.5 * w * u)))
            ms.get(b).set_location(osim.Vec3(*(mid - 0.5 * w * u)))
            moved[a] = moved[b] = float(0.5 * w - 0.5 * np.linalg.norm(pa - pb))
    model.printToXML(model_out)
    log("placed: " + ", ".join(f"{k} {v*1000:.0f}mm" for k, v in moved.items()))
    return moved


def ik_errors(errfile):
    tbl = osim.TimeSeriesTable(errfile)
    rms = tbl.getDependentColumn("marker_error_RMS").to_numpy()
    mx = tbl.getDependentColumn("marker_error_max").to_numpy()
    return {"rms_mean_m": float(rms.mean()), "rms_max_m": float(rms.max()),
            "max_marker_error_mean_m": float(mx.mean()),
            "max_marker_error_max_m": float(mx.max())}


def per_marker_rms(model_file, mot):
    model = osim.Model(model_file)
    state = model.initSystem()
    t, cols = coords_table_to_states(model, mot)
    tbl = osim.TimeSeriesTableVec3(TRC)
    names = list(marker_table())
    data = {n: trc_array(tbl, n) for n in names}
    err = {n: [] for n in names}
    for i in range(len(t)):
        for c, v in cols:
            c.setValue(state, float(v[i]), False)
        model.realizePosition(state)
        for n in names:
            p = model.getMarkerSet().get(n).getLocationInGround(state)
            err[n].append(np.linalg.norm(p.to_numpy() - data[n][i]))
    return {n: float(np.sqrt(np.mean(np.square(e)))) for n, e in err.items()}


def stage_ik():
    """Three IK passes with marker placement in between (no static trial).

    pass 1: scaled model, generic marker locations, ONLY anatomical markers
            (ASIS, SACR, HJC, epicondyles, malleoli, CAL, 5TH, AC/EP/WR).
    place:  non-fixed markers -> trial-mean body-frame position; epicondyle /
            malleolus pairs widened to the data's separation (midpoint kept).
    pass 2: all markers.
    place:  non-fixed markers again, now from cluster-informed kinematics.
    pass 3: all markers -> the IK used everywhere downstream.
    Fixed (never moved): ASIS, HJC, epicondyle/malleolus MIDPOINTS -- these
    define the model's pelvis, hip, knee and ankle.
    """
    anat = FIXED | {"SACR", "RCAL", "R5TH", "LCAL", "L5TH", "RAC", "LAC",
                    "REP", "LEP", "RWR", "LWR"}
    w1 = {n: (WEIGHTS.get(n, 1.0) if n in anat else 0.0) for n in marker_table()}
    w2 = {n: WEIGHTS.get(n, 1.0) for n in marker_table()}
    movable = [n for n in marker_table() if n not in FIXED]
    t0 = time.time()
    run_ik(SCALED0, IK1, "p1_setup_ik_pass1.xml", w1)
    e1 = ik_errors(os.path.splitext(IK1)[0] + "_ik_marker_errors.sto")
    m1 = place_markers(SCALED0, IK1, "p1_scaled_placed1.osim", movable, True)
    run_ik("p1_scaled_placed1.osim", "p1_trial13_ik_pass2.mot",
           "p1_setup_ik_pass2.xml", w2)
    m2 = place_markers("p1_scaled_placed1.osim", "p1_trial13_ik_pass2.mot",
                       SCALED, movable, False)
    run_ik(SCALED, IK, "p1_setup_ik.xml", w2)
    errs = ik_errors(os.path.splitext(IK)[0] + "_ik_marker_errors.sto")
    pm = per_marker_rms(SCALED, IK)
    summ = load_summary()
    summ["ik"] = {**errs, "per_marker_rms_m": pm, "weights": w2,
                  "pass1_errors_generic_markers": e1,
                  "placement_shift_m_pass1": m1,
                  "placement_shift_m_pass2": m2,
                  "runtime_s": time.time() - t0,
                  "procedure": stage_ik.__doc__}
    save_summary(summ)
    log(f"IK pass1 mean RMS {e1['rms_mean_m']*1000:.1f} mm; final mean RMS "
        f"{errs['rms_mean_m']*1000:.1f} mm, worst max "
        f"{errs['max_marker_error_max_m']*1000:.1f} mm")
    log("per-marker RMS mm: " + ", ".join(f"{k} {v*1000:.0f}"
                                          for k, v in pm.items()))


# ---------------------------------------------------------------------------
# 3. ID
# ---------------------------------------------------------------------------

def write_external_loads():
    el = osim.ExternalLoads()
    el.setName(f"{TAG}_grf")
    el.setDataFileName(GRF)
    for s in "rl":
        ef = osim.ExternalForce()
        ef.setName(f"grf_{s}")
        ef.set_applied_to_body(f"calcn_{s}")
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body("ground")
        ef.set_force_identifier(f"ground_force_{s}_v")
        ef.set_point_identifier(f"ground_force_{s}_p")
        ef.set_torque_identifier(f"ground_torque_{s}_")
        el.cloneAndAppend(ef)
    el.printToXML(EXT_LOADS)


def stage_id():
    write_external_loads()
    idt = osim.InverseDynamicsTool()
    idt.setName(f"{TAG}_id")
    idt.setModelFileName(SCALED)
    idt.setCoordinatesFileName(IK)
    idt.setLowpassCutoffFrequency(LOWPASS_HZ)
    idt.setExternalLoadsFileName(EXT_LOADS)
    idt.setOutputGenForceFileName(ID)
    idt.setResultsDir(".")
    idt.setStartTime(0.0)
    idt.setEndTime(T_END)
    ex = osim.ArrayStr()
    ex.append("Muscles")
    idt.setExcludedForces(ex)
    idt.printToXML("p1_setup_id.xml")
    t0 = time.time()
    osim.InverseDynamicsTool("p1_setup_id.xml").run()
    summ = load_summary()
    summ["id"] = {"runtime_s": time.time() - t0,
                  "lowpass_coordinates_hz": LOWPASS_HZ,
                  "excluded_forces": ["Muscles"]}
    save_summary(summ)


# ---------------------------------------------------------------------------
# 4/5. processed model, SO and MocoInverse
# ---------------------------------------------------------------------------

def processed_model():
    """Scaled model -> DeGroote-Fregly muscles (rigid tendon, no passive),
    subtalar/mtp/wrist welded, pelvis residuals, unit reserves elsewhere,
    external loads attached.  The SAME model feeds SO and MocoInverse so the
    only difference between them is static vs. dynamic optimization."""
    base = osim.Model(SCALED)
    fs = base.updForceSet()
    for name in ("wrist_flex_r", "wrist_dev_r", "wrist_flex_l", "wrist_dev_l"):
        fs.remove(fs.getIndex(name))       # their coordinates get welded
    base.initSystem()
    for coord, f in (("pelvis_tilt", 100.0), ("pelvis_list", 100.0),
                     ("pelvis_rotation", 100.0), ("pelvis_tx", 100.0),
                     ("pelvis_ty", 100.0), ("pelvis_tz", 100.0)):
        ca = osim.CoordinateActuator(coord)
        ca.setName(f"residual_{coord}")
        ca.setOptimalForce(f)
        ca.setMinControl(-np.inf)
        ca.setMaxControl(np.inf)
        base.addForce(ca)
    base.finalizeConnections()
    mp = osim.ModelProcessor(base)
    welds = osim.StdVectorString()
    for j in ("subtalar_r", "subtalar_l", "mtp_r", "mtp_l",
              "radius_hand_r", "radius_hand_l"):
        welds.append(j)
    mp.append(osim.ModOpReplaceJointsWithWelds(welds))
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    mp.append(osim.ModOpIgnoreTendonCompliance())
    mp.append(osim.ModOpAddReserves(1.0))
    return mp


def stage_so():
    write_external_loads()
    mp = processed_model()
    m = mp.process()
    m.initSystem()
    m.printToXML(PROCESSED)
    at = osim.AnalyzeTool()
    at.setName(f"{TAG}_so")
    at.setModelFilename(PROCESSED)
    at.setCoordinatesFileName(IK)
    at.setLowpassCutoffFrequency(LOWPASS_HZ)
    at.setExternalLoadsFileName(EXT_LOADS)
    at.setInitialTime(WINDOW[0])
    at.setFinalTime(WINDOW[1])
    at.setResultsDir("so")
    so = osim.StaticOptimization()
    so.setName("StaticOptimization")
    so.setStartTime(WINDOW[0])
    so.setEndTime(WINDOW[1])
    so.setUseModelForceSet(True)
    so.setActivationExponent(2.0)
    so.setUseMusclePhysiology(True)
    so.setConvergenceCriterion(1e-4)
    so.setMaxIterations(200)
    at.getAnalysisSet().cloneAndAppend(so)
    at.printToXML("p1_setup_so.xml")
    t0 = time.time()
    tool = osim.AnalyzeTool("p1_setup_so.xml")
    tool.run()
    ctrl = f"so/{TAG}_so_StaticOptimization_controls.xml"
    if os.path.exists(ctrl):                # 8 MB, same data as _force.sto
        os.remove(ctrl)
    summ = load_summary()
    summ["so"] = {"runtime_s": time.time() - t0, "window_s": list(WINDOW),
                  "activation_exponent": 2, "use_muscle_physiology": True,
                  "convergence_criterion": 1e-4, "max_iterations": 200,
                  "lowpass_coordinates_hz": LOWPASS_HZ}
    save_summary(summ)


def stage_moco(window=WINDOW, tag=""):
    write_external_loads()
    mp = processed_model()
    mp.append(osim.ModOpAddExternalLoads(EXT_LOADS))
    inv = osim.MocoInverse()
    inv.setName("p1_trial13_mocoinverse" + tag)
    inv.setModel(mp)
    tp = osim.TableProcessor(IK)
    tp.append(osim.TabOpLowPassFilter(LOWPASS_HZ))
    tp.append(osim.TabOpUseAbsoluteStateNames())
    inv.setKinematics(tp)
    inv.set_initial_time(window[0])
    inv.set_final_time(window[1])
    inv.set_mesh_interval(MESH)
    inv.set_kinematics_allow_extra_columns(True)
    inv.set_minimize_sum_squared_activations(True)
    inv.set_convergence_tolerance(1e-3)
    inv.set_constraint_tolerance(1e-3)
    inv.set_max_iterations(3000)
    inv.append_output_paths(".*tendon_force")
    inv.append_output_paths(".*activation")
    inv.printToXML(f"p1_setup_mocoinverse{tag}.xml")

    study = inv.initialize()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    settings = {
        "solver": "MocoCasADiSolver (CasADi + IPOPT)",
        "optim_solver": solver.get_optim_solver(),
        "transcription_scheme": solver.get_transcription_scheme(),
        "num_mesh_intervals": solver.get_num_mesh_intervals(),
        "mesh_interval_s": MESH,
        "multibody_dynamics_mode": solver.get_multibody_dynamics_mode(),
        "optim_convergence_tolerance": solver.get_optim_convergence_tolerance(),
        "optim_constraint_tolerance": solver.get_optim_constraint_tolerance(),
        "optim_max_iterations": solver.get_optim_max_iterations(),
        "optim_hessian_approximation": solver.get_optim_hessian_approximation(),
        "window_s": list(window),
        "cost": ("MocoInverse default: sum of squared controls (muscle "
                 "excitations + reserves/residuals, weight 1) + sum of "
                 "squared muscle activations (minimize_sum_squared_"
                 "activations = true)"),
        "muscle_model": ("DeGrooteFregly2016, rigid tendon, passive fibre "
                         "force ignored, active force-length width x1.5, "
                         "fibre damping 0.1, activation dynamics "
                         "tau_act/tau_deact = 0.010/0.040 s (inherited from "
                         "the Rajagopal model), tanh smoothing b = 0.1"),
        "kinematics": f"IK, low-pass {LOWPASS_HZ} Hz, prescribed (PositionMotion)",
    }
    log(f"MocoInverse settings: {settings}")
    t0 = time.time()
    res = inv.solve()
    rt = time.time() - t0
    sol = res.getMocoSolution()
    success = bool(sol.success())
    sol.unseal()
    sol.write(f"p1_trial13_mocoinverse{tag}_solution.sto")
    outs = res.getOutputs()
    osim.STOFileAdapter.write(outs, f"p1_trial13_mocoinverse{tag}_outputs.sto")
    summ = load_summary()
    summ["moco" + tag] = {**settings, "runtime_s": rt, "success": success,
                         "status": sol.getStatus(),
                         "num_iterations": sol.getNumIterations(),
                         "objective": sol.getObjective()}
    save_summary(summ)
    log(f"MocoInverse{tag}: success={success} status={sol.getStatus()} "
        f"iters={sol.getNumIterations()} obj={sol.getObjective():.4g} "
        f"runtime {rt:.0f} s")


# ---------------------------------------------------------------------------
# 6. collect -> npz/json contract
# ---------------------------------------------------------------------------

def read_sto(path):
    tbl = osim.TimeSeriesTable(path)
    return (np.array(tbl.getIndependentColumn()),
            list(tbl.getColumnLabels()), tbl.getMatrix().to_numpy())


def stage_collect():
    import collect
    collect.main()


STAGES = {"scale": stage_scale, "ik": stage_ik, "id": stage_id,
          "so": stage_so, "moco": stage_moco, "collect": stage_collect,
          "moco_stride": lambda: stage_moco((1.0, 2.2), "_stride")}

if __name__ == "__main__":
    os.chdir(OUT)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for st in sys.argv[1:]:
        log(f"=== {st} ===")
        STAGES[st]()
