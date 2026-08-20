"""boneid.io_v3d — read Visual3D-exported ``*_5StridesData.mat`` trials.

Style follows core.py: plain @dataclass records of numpy arrays plus free
functions. No member functions. SI units, time-major arrays [T, ...].

The export format (van der Zee, Mundinger & Kuo, "A biomechanics dataset of
healthy human walking at various speeds, step lengths and step widths",
Sci Data 2022,
https://www.nature.com/articles/s41597-022-01817-1) holds, per subject file, a
1-D array of trial structs with these fields (shapes for p1, trial 0):

    Platform.ForcePlatformCorners   [3,4,2]  mm, lab frame, 2 plates
    Analog.f{1,2}{original,processed}, m{1,2}{...}   [Ta,3] plate-local analog
    TargetData.<MARKER>_pos         [T,5]  x y z residual n_cameras, METERS
    TargetData.<MARKER>_pos_proc    [T,5]  Visual3D-processed (gap filled/filtered)
    Landmark.HH{L,R}                [T,3]  hip joint centres, m
    Time.TIME [7200] @120 Hz, Time.AnalogTime [72000] @1200 Hz  (UNCROPPED)
    Force.{force,cop,freemoment}{1,2}  [Ta,3]  N / m / N m, LAB frame
    Kinetic_Kinematic.<seg><quantity>  [T,3]  Visual3D's own kinematics/kinetics
    Link_Model_Based.<joint>_<quantity>  [T,3] Visual3D joint angles/moments/...

Empirical findings that drive this loader (verified on p1, all 33 trials, see
tests/test_io_v3d.py):

* TIME/AnalogTime cover the whole 60 s recording, but everything else is
  already cropped to a ~5-stride window: T frames of mocap and Ta ~= 10*T - k
  (k in 4..15) samples of analog.  Cross-correlating plate vertical force
  against Visual3D's own ``rFtProxEndForce`` over sub-sample shifts puts the
  crop origins on top of each other to better than 2 analog samples (1.7 ms;
  r = 0.99996 at shift -1.5 samples vs 0.99994 at shift 0 for p1 trial 0).
  So we simply define  t_mocap[i] = i/120  and  t_analog[k] = k/1200,
  both starting at zero at the start of the cropped window, and interpolate.
  The trailing k analog samples that have no mocap partner are clamped.
* Plate 2 (``force2``) sits at lab x < 0 and carries the RIGHT foot in every
  p1 trial; ``plate_for_side`` re-derives this per trial from COP proximity to
  the foot markers rather than trusting it.
* Lab frame: +Z up, +X to the subject's left, +Y posterior (the runner faces
  -Y).  Nothing here hardcodes that; segment frames come from the markers.
* Force/COP/free moment are already in the lab frame and in SI; the plate
  surface is at z = -0.1746 m.  Free moment is purely vertical.
* ``Link_Model_Based.*_moment`` is NORMALISED BY BODY MASS (N m / kg): p1
  trial 0 has max |r_ank_moment| = 1.254 N m/kg while the equivalent
  ``Kinetic_Kinematic.rFtProxEndTorque`` peaks at 105 N m, and the estimated
  body mass is 84.8 kg (105/84.8 = 1.24).  ``*_angle`` is in DEGREES.
* ``Kinetic_Kinematic.*AngVel`` / ``*AngAcc`` are in the LAB frame, rad/s:
  our world-frame shank omega correlates 0.987 with ``rSkAngVel`` on the
  dominant (lab-x) component with amplitude ratio 0.98; the local-frame
  version does not line up component-wise.
* The 33 trial slots are the paper's 33 controlled WALKING conditions
  (speed 0.7-2.0 m/s x step length x step width) -- there is no running
  anywhere.  p1 trial 0: 5 strides in 11.3 s, 57% stance, ~23% double
  support; trial 24: 5 strides in 4.7 s.  5 slots are empty placeholders.
  So "crossover" frames are common and real: they are double support.
* End-to-end check: feeding `build_chain` straight into
  ``core.inverse_dynamics`` reproduces Visual3D's own lab-frame
  ``rFtProxEndTorque`` / ``rSkProxEndTorque`` to 1.6 / 1.7 N m RMS on p1 trial
  0 (peaks 105 / 70 N m), same sign convention -- which is the real proof that
  the crop alignment, units and plate assignment here are right.
* Segment naming in Kinetic_Kinematic is inconsistent: the right shank is
  ``rSk`` for AngVel/AngAcc/CG* but ``rSh`` for ProxEnd*/DistEnd*/SegResidual.
  ``reference_series`` hides that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.io

from .core import (
    ForcePlate,
    GroundWrench,
    SegmentKinematics,
    Skeleton,
    detect_contact,
    lowpass,
)

MOCAP_RATE = 120.0
ANALOG_RATE = 1200.0

#: Markers present in the export (side prefix L/R, plus the unpaired SACR).
SIDED_MARKERS = ("5TH", "AC", "ASI", "CAL", "EP", "GTR", "LEP", "LML",
                 "MEP", "MML", "SH1", "SH2", "SH3", "TH1", "TH2", "TH3", "WR")


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------

@dataclass
class V3dTrial:
    """One Visual3D trial, raw-ish: everything on the cropped 5-stride window.

    Two time bases share an origin: `t` (mocap, `rate` Hz, T samples) and
    `t_analog` (`analog_rate` Hz, Ta samples).  `t_full` is the uncropped
    60 s recording clock kept only for provenance -- no array here is indexed
    by it.
    """

    path: str
    index: int                          # trial index within the file
    rate: float                         # Hz, mocap
    analog_rate: float                  # Hz, force/analog
    t: np.ndarray                       # [T] s
    t_analog: np.ndarray                # [Ta] s
    markers: dict[str, np.ndarray]      # name -> [T,3] m (raw `_pos`)
    markers_proc: dict[str, np.ndarray]  # name -> [T,3] m (`_pos_proc`)
    marker_residual: dict[str, np.ndarray]  # name -> [T] m, V3D fit residual
    landmarks: dict[str, np.ndarray]    # 'HHL'/'HHR' -> [T,3] m
    force: np.ndarray                   # [Ta,P,3] N, lab frame
    cop: np.ndarray                     # [Ta,P,3] m, lab frame (junk when unloaded)
    free_moment: np.ndarray             # [Ta,P,3] N m, lab frame (z only)
    plates: list[ForcePlate]            # [P]
    reference: dict[str, np.ndarray]    # V3D's own outputs, see module docstring
    t_full: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_analog_full: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _struct_fields(s) -> list[str]:
    return list(getattr(s, "_fieldnames", []))


def _xyz(a: np.ndarray) -> np.ndarray:
    """First three columns of a V3D target array as float64 [T,3]."""
    return np.asarray(a, dtype=float)[:, :3]


def _plate_geometry(platform) -> list[ForcePlate]:
    """ForcePlate records from ``ForcePlatformCorners`` [3,4,P] in mm.

    Plate frame: e1 along corner0->corner1, e2 along corner3->corner0, e3 =
    e1 x e2 (up).  ``ForcePlatformOrigin`` is empty in this export, so the
    plate origin is taken as the centroid of the four corners.
    """
    corners = np.asarray(platform.ForcePlatformCorners, dtype=float) / 1000.0
    plates: list[ForcePlate] = []
    for p in range(corners.shape[2]):
        c = corners[:, :, p].T                       # [4,3] m
        e1 = c[1] - c[0]
        e2 = c[0] - c[3]
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 - e1 * (e2 @ e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        plates.append(ForcePlate(corners=c, origin=c.mean(axis=0),
                                 r_lab=np.stack([e1, e2, e3], axis=1)))
    return plates


def _trial_from_struct(s, path: str, index: int) -> V3dTrial:
    td = s.TargetData
    markers: dict[str, np.ndarray] = {}
    markers_proc: dict[str, np.ndarray] = {}
    residual: dict[str, np.ndarray] = {}
    for f in _struct_fields(td):
        a = getattr(td, f)
        if f.endswith("_pos_proc"):
            markers_proc[f[: -len("_pos_proc")]] = _xyz(a)
        elif f.endswith("_pos"):
            name = f[: -len("_pos")]
            markers[name] = _xyz(a)
            residual[name] = np.asarray(a, dtype=float)[:, 3]

    n_t = len(next(iter(markers.values())))
    force = np.stack([np.asarray(s.Force.force1, float),
                      np.asarray(s.Force.force2, float)], axis=1)
    cop = np.stack([np.asarray(s.Force.cop1, float),
                    np.asarray(s.Force.cop2, float)], axis=1)
    fmom = np.stack([np.asarray(s.Force.freemoment1, float),
                     np.asarray(s.Force.freemoment2, float)], axis=1)
    n_a = len(force)

    landmarks = {f: np.asarray(getattr(s.Landmark, f), float)
                 for f in _struct_fields(s.Landmark)}

    reference: dict[str, np.ndarray] = {}
    for group in ("Kinetic_Kinematic", "Link_Model_Based"):
        g = getattr(s, group, None)
        if g is None:
            continue
        for f in _struct_fields(g):
            reference[f] = np.asarray(getattr(g, f), dtype=float)

    return V3dTrial(
        path=path, index=index, rate=MOCAP_RATE, analog_rate=ANALOG_RATE,
        t=np.arange(n_t) / MOCAP_RATE,
        t_analog=np.arange(n_a) / ANALOG_RATE,
        markers=markers, markers_proc=markers_proc, marker_residual=residual,
        landmarks=landmarks, force=force, cop=cop, free_moment=fmom,
        plates=_plate_geometry(s.Platform), reference=reference,
        t_full=np.asarray(s.Time.TIME, float),
        t_analog_full=np.asarray(s.Time.AnalogTime, float))


def _is_populated(s) -> bool:
    """Some trial slots are empty placeholders (p1 slots 25-29 of 33): the
    struct exists but every field is a 0-length ndarray."""
    return hasattr(getattr(s, "TargetData", None), "_fieldnames")


def load_v3d_trials(path: str | os.PathLike, skip_empty: bool = True):
    """Read the trials in a ``p<N>_5StridesData.mat`` file.

    The file always holds 33 trial slots, but some are empty placeholders for
    conditions the subject did not complete (p1 has 28 populated slots; 25-29
    are empty).  With `skip_empty` (the default) only populated trials come
    back, each keeping its file slot in `V3dTrial.index`; otherwise the list is
    33 long with None in the empty slots.
    """
    path = os.fspath(path)
    raw = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    data = np.atleast_1d(raw["data"])
    out = [_trial_from_struct(s, path, i) if _is_populated(s) else None
           for i, s in enumerate(data)]
    return [t for t in out if t is not None] if skip_empty else out


def load_v3d_trial(path: str | os.PathLike, index: int = 0) -> V3dTrial:
    """Read a single trial by FILE SLOT index (raises if that slot is empty)."""
    trial = load_v3d_trials(path, skip_empty=False)[index]
    if trial is None:
        raise ValueError(f"trial slot {index} of {path} is empty")
    return trial


def reference_series(trial: V3dTrial, name: str) -> np.ndarray:
    """Visual3D reference array by name, tolerating the rSk/rSh naming split.

    ``reference_series(tr, 'rSkProxEndPos')`` finds ``rShProxEndPos``.  There
    is only one pelvis, exported as ``rPv``, so ``lPv*`` maps onto it too.
    """
    ref = trial.reference
    if name in ref:
        return ref[name]
    for a, b in (("rSk", "rSh"), ("rSh", "rSk"), ("lSk", "lSh"),
                 ("lSh", "lSk"), ("lPv", "rPv")):
        alt = name.replace(a, b, 1)
        if name.startswith(a) and alt in ref:
            return ref[alt]
    raise KeyError(f"no Visual3D reference series {name!r}")


# ---------------------------------------------------------------------------
# Plates, contact, body mass
# ---------------------------------------------------------------------------

def analog_to_mocap(trial: V3dTrial, x: np.ndarray,
                    antialias_hz: float = 50.0) -> np.ndarray:
    """Resample an analog-rate array [Ta,...] onto the mocap time base [T,...].

    Low-passes at `antialias_hz` (below the 60 Hz mocap Nyquist) before linear
    interpolation; set 0 to skip.  Samples beyond the (slightly short) analog
    span are clamped to the last analog value -- at most 15 samples / 12 ms.
    """
    x = np.asarray(x, dtype=float)
    if antialias_hz > 0:
        x = lowpass(x, trial.analog_rate, antialias_hz)
    flat = x.reshape(len(x), -1)
    out = np.empty((len(trial.t), flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(trial.t, trial.t_analog, flat[:, c])
    return out.reshape((len(trial.t),) + x.shape[1:])


def foot_centroid(trial: V3dTrial, side: str) -> np.ndarray:
    """[T,3] mean of the heel, 5th-metatarsal and mid-malleoli markers."""
    s = side.upper()[0]
    m = trial.markers
    ankle = 0.5 * (m[s + "LML"] + m[s + "MML"])
    return (m[s + "CAL"] + m[s + "5TH"] + ankle) / 3.0


def plate_for_side(trial: V3dTrial, side: str,
                   threshold_n: float = 20.0) -> int:
    """Index of the plate carrying `side`'s foot, by COP-to-foot proximity.

    For each plate, take the frames where it is loaded above `threshold_n` and
    measure the mean horizontal distance from its COP to the foot marker
    centroid; the smaller wins.  For p1 this always returns 1 for 'r'
    (mean distance ~0.09 m, vs ~0.5 m for the other plate).
    """
    cent = foot_centroid(trial, side)
    best, best_d = 0, np.inf
    for p in range(trial.force.shape[1]):
        fz = analog_to_mocap(trial, trial.force[:, p, 2])
        cop = analog_to_mocap(trial, trial.cop[:, p], antialias_hz=0.0)
        on = fz > threshold_n
        if on.sum() < 5:
            continue
        d = float(np.mean(np.linalg.norm(cop[on, :2] - cent[on, :2], axis=1)))
        if d < best_d:
            best, best_d = p, d
    return best


def contact_mask(trial: V3dTrial, plate: int,
                 threshold_n: float = 20.0) -> np.ndarray:
    """[T] bool: plate `plate` loaded above threshold, on the mocap time base."""
    fz = analog_to_mocap(trial, trial.force[:, plate, 2])
    mask, _ = detect_contact(fz, threshold_n, min_gap=5)
    return mask


def crossover_flags(trial: V3dTrial, side: str, threshold_n: float = 20.0,
                    other_threshold_n: float | None = None) -> np.ndarray:
    """[T] bool: this foot is in stance AND the other plate is also loaded.

    Treadmill running with one belt per foot gives a clean single-plate stance;
    a frame flagged here means the contralateral foot (or a stray part of this
    one) is on the other belt, so the ground wrench cannot be attributed to
    this limb alone.  `other_threshold_n` defaults to `threshold_n`
    (AnalysisParams.contact_threshold_n is the natural source for both).
    """
    if other_threshold_n is None:
        other_threshold_n = threshold_n
    p = plate_for_side(trial, side, threshold_n)
    mine = contact_mask(trial, p, threshold_n)
    other = np.zeros_like(mine)
    for q in range(trial.force.shape[1]):
        if q != p:
            other |= contact_mask(trial, q, other_threshold_n)
    return mine & other


def estimate_body_mass(trial: V3dTrial, gravity: float = 9.81,
                       threshold_n: float = 20.0) -> float:
    """Body mass [kg] from the mean total vertical GRF over whole strides.

    Averages the summed vertical force of both plates between the first and
    last right-foot contact onset in the cropped window, so the average spans
    an integer number of strides (falls back to the whole window if fewer than
    two contacts are found).  On p1 this gives 84.7-85.0 kg across all 33
    trials (spread 0.4%).
    """
    fz_total = trial.force[:, :, 2].sum(axis=1)
    p = plate_for_side(trial, "r", threshold_n)
    _, events = detect_contact(trial.force[:, p, 2], threshold_n, min_gap=120)
    if len(events) >= 2:
        fz_total = fz_total[events[0][0]:events[-1][0]]
    return float(fz_total.mean() / gravity)


# ---------------------------------------------------------------------------
# de Leva (1996) segment inertial parameters
# ---------------------------------------------------------------------------
#
# de Leva P. (1996) "Adjustments to Zatsiorsky-Seluyanov's segment inertia
# parameters", J Biomech 29(9):1223-1230, Table 4.  Mass as a fraction of body
# mass; CM position as a fraction of segment length measured FROM THE
# PROXIMAL endpoint of de Leva's segment definition; radii of gyration as
# fractions of segment length about the sagittal (AP), transverse (ML) and
# longitudinal axes through the CM.
#
# Segment endpoints in de Leva:
#   foot   = heel (calcaneus) -> toe tip (acropodion)
#   shank  = femoral condyles (knee JC) -> malleoli (ankle JC)
#   thigh  = hip JC -> femoral condyles (knee JC)
#   pelvis = omphalion -> mid-hip JC        (de Leva's LOWER trunk, LPT)
#   mid_trunk  = xiphion -> omphalion       (MPT)
#   upper_trunk = suprasternale -> xiphion  (UPT)
#   head   = vertex -> cervicale (C7)       (head AND neck)
#   upper_arm = shoulder JC -> elbow JC
#   forearm   = elbow JC -> stylion (wrist)
#   hand      = stylion -> 3rd dactylion
#
# (sagittal, transverse and longitudinal radii differ by <5% for the leg
# segments, so the axis assignment below is not a sensitive choice.)
#
# The 12-segment whole-body model uses every row below exactly once per side,
# and the mass fractions sum to 1.000 (male) / 0.9999 (female):
#   2*(foot+shank+thigh) + pelvis(LPT) + [UPT+MPT+head] + 2*(upper_arm+forearm+hand)

DELEVA_MALE = {
    #                mass    com     rg_sagittal rg_transverse rg_longitudinal
    "foot":         (0.0137, 0.4415, 0.257, 0.245, 0.124),
    "shank":        (0.0433, 0.4459, 0.255, 0.249, 0.103),
    "thigh":        (0.1416, 0.4095, 0.329, 0.329, 0.149),
    "pelvis":       (0.1117, 0.6115, 0.615, 0.551, 0.587),
    "mid_trunk":    (0.1633, 0.4502, 0.482, 0.383, 0.468),
    "upper_trunk":  (0.1596, 0.2999, 0.716, 0.454, 0.659),
    "head":         (0.0694, 0.5976, 0.362, 0.376, 0.312),
    "upper_arm":    (0.0271, 0.5772, 0.285, 0.269, 0.158),
    "forearm":      (0.0162, 0.4574, 0.276, 0.265, 0.121),
    "hand":         (0.0061, 0.7900, 0.628, 0.513, 0.401),
}

DELEVA_FEMALE = {
    "foot":         (0.0129, 0.4014, 0.299, 0.279, 0.139),
    "shank":        (0.0481, 0.4416, 0.271, 0.267, 0.093),
    "thigh":        (0.1478, 0.3612, 0.369, 0.364, 0.162),
    "pelvis":       (0.1247, 0.4920, 0.433, 0.402, 0.444),
    "mid_trunk":    (0.1465, 0.4512, 0.433, 0.354, 0.415),
    "upper_trunk":  (0.1545, 0.2077, 0.746, 0.502, 0.718),
    "head":         (0.0668, 0.5894, 0.330, 0.359, 0.318),
    "upper_arm":    (0.0255, 0.5754, 0.278, 0.260, 0.148),
    "forearm":      (0.0138, 0.4559, 0.261, 0.257, 0.094),
    "hand":         (0.0056, 0.7474, 0.531, 0.454, 0.335),
}

#: de Leva (1996) Table 2 SAMPLE-MEAN segment lengths [m] for the two segments
#: whose endpoints this markerset cannot see (no head markers, no hand markers)
#: plus the two trunk sub-segments (xiphion is not marked either).  These are
#: population means, NOT subject-scaled -- see `deleva_upper_body` for exactly
#: how each is used and how much it can matter.
DELEVA_MEAN_LENGTH = {
    "male":   {"head": 0.2033, "hand": 0.0862,
               "upper_trunk": 0.2421, "mid_trunk": 0.2155},
    "female": {"head": 0.2002, "hand": 0.0780,
               "upper_trunk": 0.2280, "mid_trunk": 0.2053},
}


def deleva_skeleton(body_mass: float, foot_length: float, shank_length: float,
                    thigh_length: float, pelvis_length: float,
                    foot_heel_local: np.ndarray | None = None,
                    sex: str = "male") -> Skeleton:
    """Skeleton for the chain [foot, shank, thigh, pelvis] from de Leva (1996).

    SEGMENT FRAME CONVENTION (shared with `build_chain`, right-handed):

        shank / thigh / pelvis   x = anterior, y = subject's LEFT, z = the
                                 long axis pointing DISTAL -> PROXIMAL ("up"
                                 in anatomical standing posture).
        foot                     z = the long axis pointing HEEL -> TOE,
                                 y = subject's LEFT, x = z x y = plantar
                                 ("down" in standing posture).

    In every case local z is the segment's LONGITUDINAL axis (the one de Leva's
    longitudinal radius of gyration refers to) and local y is the medio-lateral
    axis, so sagittal-plane rotation is about y for all four segments.  The
    inertia tensor is diagonal in this frame:

        I = m * diag((rg_sagittal*L)^2, (rg_transverse*L)^2, (rg_long*L)^2)

    i.e. the sagittal radius of gyration is taken about the sagittal (AP) axis
    x and the transverse one about the transverse (ML) axis y, the usual
    reading of Zatsiorsky's axis names.  The two differ by under 4% for every
    leg segment (0.255 vs 0.249 for the shank), so swapping them changes a
    moment of inertia by under 5% -- this is not a sensitive choice.

    `com_local` is measured from the segment ORIGIN, which core.py fixes at the
    PROXIMAL JOINT.  For shank/thigh/pelvis the origin coincides with de Leva's
    proximal endpoint, so com_local = [0, 0, -com_fraction * L].  The foot is
    the exception: its origin is the ankle joint centre but de Leva measures
    from the heel, so pass `foot_heel_local` = the heel marker position in the
    foot frame relative to the ankle (build_chain computes it); the foot COM is
    then foot_heel_local + [0, 0, com_fraction * foot_length].

    `length` records the segment lengths used for the inertia scaling.
    """
    table = DELEVA_MALE if sex.lower().startswith("m") else DELEVA_FEMALE
    names = ["foot", "shank", "thigh", "pelvis"]
    lengths = np.array([foot_length, shank_length, thigh_length, pelvis_length],
                       dtype=float)
    if foot_heel_local is None:
        foot_heel_local = np.zeros(3)
    foot_heel_local = np.asarray(foot_heel_local, dtype=float)

    mass = np.empty(4)
    com = np.zeros((4, 3))
    inertia = np.zeros((4, 3, 3))
    for i, name in enumerate(names):
        m_frac, c_frac, rg_sag, rg_tra, rg_lon = table[name]
        length = lengths[i]
        mass[i] = m_frac * body_mass
        if name == "foot":
            com[i] = foot_heel_local + np.array([0.0, 0.0, c_frac * length])
        else:
            com[i] = np.array([0.0, 0.0, -c_frac * length])
        inertia[i] = np.diag(mass[i] * (np.array([rg_sag, rg_tra, rg_lon])
                                        * length) ** 2)

    return Skeleton(segment_names=names,
                    joint_names=["ankle", "knee", "hip"],
                    mass=mass, com_local=com, inertia_local=inertia,
                    length=lengths)


# ---------------------------------------------------------------------------
# Segment frames and the chain
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def orthonormal_frame(long_axis: np.ndarray, left_axis: np.ndarray) -> np.ndarray:
    """world_from_segment [T,3,3] from a long axis (-> local z) and an
    approximate subject-left direction (-> local y).

    `left_axis` is orthogonalised against `long_axis`; local x completes the
    right-handed triad (x = y x z).  Columns of the result are the local axes
    expressed in the lab frame.
    """
    z = _unit(np.asarray(long_axis, float))
    y = np.asarray(left_axis, float)
    y = _unit(y - z * np.sum(y * z, axis=-1, keepdims=True))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def joint_centers(trial: V3dTrial, side: str) -> dict[str, np.ndarray]:
    """Per-frame joint centres and segment endpoints [T,3] for one leg.

    ankle   = mid(LML, MML)     (matches V3D rFtProxEndPos to 3.1 mm RMS)
    knee    = mid(LEP, MEP)     (matches V3D rShProxEndPos to 3.4 mm RMS)
    hip     = the HHR/HHL landmark, which Visual3D already exports per frame
              (it is rigid in the RASI/LASI/SACR cluster to 0.6 mm, so no
              re-fit is needed; matches V3D rThProxEndPos to 20 mm RMS)
    mid_hip = mid(HHL, HHR)
    l5s1    = mid(mid-ASIS, SACR) -- our pelvis proximal end, a mid-pelvis
              point at ASIS height standing in for the L5/S1 joint
    toe     = the 5th-metatarsal marker; heel = the calcaneus marker.
    """
    s = side.upper()[0]
    m = trial.markers
    hip_name = "HHR" if s == "R" else "HHL"
    return {
        "ankle": 0.5 * (m[s + "LML"] + m[s + "MML"]),
        "knee": 0.5 * (m[s + "LEP"] + m[s + "MEP"]),
        "hip": trial.landmarks[hip_name],
        "mid_hip": 0.5 * (trial.landmarks["HHL"] + trial.landmarks["HHR"]),
        "l5s1": 0.5 * (0.5 * (m["RASI"] + m["LASI"]) + m["SACR"]),
        "heel": m[s + "CAL"],
        "toe": m[s + "5TH"],
    }


def segment_frames(trial: V3dTrial, side: str) -> np.ndarray:
    """world_from_segment [T,4,3,3] for [foot, shank, thigh, pelvis].

    Medio-lateral axes come from the anatomical marker pairs: malleoli for the
    foot and shank, femoral epicondyles for the thigh, ASIS markers for the
    pelvis.  For the RIGHT leg the medial marker lies to the subject's left of
    the lateral one, so left_axis = medial - lateral; the left leg flips it.
    """
    s = side.upper()[0]
    m = trial.markers
    jc = joint_centers(trial, side)
    sign = 1.0 if s == "R" else -1.0
    ml_ankle = sign * (m[s + "MML"] - m[s + "LML"])
    ml_knee = sign * (m[s + "MEP"] - m[s + "LEP"])
    ml_pelvis = m["LASI"] - m["RASI"]

    foot = orthonormal_frame(jc["toe"] - jc["heel"], ml_ankle)
    shank = orthonormal_frame(jc["knee"] - jc["ankle"], ml_ankle)
    thigh = orthonormal_frame(jc["hip"] - jc["knee"], ml_knee)
    pelvis = orthonormal_frame(jc["l5s1"] - jc["mid_hip"], ml_pelvis)
    return np.stack([foot, shank, thigh, pelvis], axis=1)


def ground_wrench(trial: V3dTrial, plate: int,
                  point: np.ndarray | None = None,
                  threshold_n: float = 20.0,
                  antialias_hz: float = 50.0) -> GroundWrench:
    """Plate load as a wrench about a fixed lab `point`, on the mocap clock.

    The wrench is formed at the analog rate (where the COP is defined),
    ``moment = (cop - point) x force + free_moment``, zeroed wherever the
    vertical force is below `threshold_n` -- COP is garbage there (values of
    thousands of metres) and must never reach the interpolator unmasked.  The
    force and the moment are then anti-alias filtered and resampled onto the
    trial's mocap time base, so the result shares its time base with
    `build_chain`'s SegmentKinematics.

    `point` defaults to the lab origin [0,0,0] (the treadmill origin; the plate
    surface is 0.1746 m below it).
    """
    if point is None:
        point = np.zeros(3)
    point = np.asarray(point, dtype=float)
    f = trial.force[:, plate].copy()
    cop = trial.cop[:, plate]
    free = trial.free_moment[:, plate]
    on = f[:, 2] > threshold_n
    m = np.cross(cop - point, f) + free
    f[~on] = 0.0
    m[~on] = 0.0
    return GroundWrench(t=trial.t.copy(), rate=trial.rate,
                        force=analog_to_mocap(trial, f, antialias_hz),
                        moment=analog_to_mocap(trial, m, antialias_hz),
                        point=point)


def build_chain(trial: V3dTrial, skeleton: Skeleton | None = None,
                side: str = "r", point: np.ndarray | None = None,
                threshold_n: float = 20.0, sex: str = "male"):
    """(Skeleton, SegmentKinematics, GroundWrench) for [foot, shank, thigh, pelvis].

    Everything lands on the trial's mocap time base (the full cropped window).

    Segment origins (`prox_pos`) are the proximal joints, as core.py requires:
        foot   prox = ankle,   dist = 5th metatarsal ("toe")
        shank  prox = knee,    dist = ankle
        thigh  prox = hip,     dist = knee
        pelvis prox = L5S1     (mid of mid-ASIS and SACR), dist = mid-hip
    Rotations follow the convention documented in `deleva_skeleton`.

    If `skeleton` is None one is built with `deleva_skeleton` from
    `estimate_body_mass` and the trial-mean segment lengths, including the
    heel-in-foot-frame offset needed to place the foot COM.
    """
    jc = joint_centers(trial, side)
    r_world = segment_frames(trial, side)

    prox = np.stack([jc["ankle"], jc["knee"], jc["hip"], jc["l5s1"]], axis=1)
    dist = np.stack([jc["toe"], jc["ankle"], jc["knee"], jc["mid_hip"]], axis=1)

    if skeleton is None:
        heel_local = np.einsum("tji,tj->ti", r_world[:, 0],
                               jc["heel"] - jc["ankle"]).mean(axis=0)
        lengths = np.linalg.norm(dist - prox, axis=2).mean(axis=0)
        foot_length = float(np.linalg.norm(jc["toe"] - jc["heel"],
                                           axis=1).mean())
        skeleton = deleva_skeleton(
            body_mass=estimate_body_mass(trial, threshold_n=threshold_n),
            foot_length=foot_length, shank_length=float(lengths[1]),
            thigh_length=float(lengths[2]), pelvis_length=float(lengths[3]),
            foot_heel_local=heel_local, sex=sex)

    kin = SegmentKinematics(t=trial.t.copy(), rate=trial.rate,
                            r_world=r_world, prox_pos=prox, dist_pos=dist)
    ground = ground_wrench(trial, plate_for_side(trial, side, threshold_n),
                           point=point, threshold_n=threshold_n)
    return skeleton, kin, ground


# ---------------------------------------------------------------------------
# Upper body: torso (+head lumped) and two arms (+hands lumped)
# ---------------------------------------------------------------------------

def _lump(bodies) -> tuple[float, float, np.ndarray]:
    """Rigidly combine bodies given along a common axis.

    `bodies` is a sequence of (mass, z_com, inertia_diag[3]) where `z_com` is
    the body's COM coordinate along the shared local z axis (the other two
    coordinates are assumed zero -- everything here is axisymmetric-ish and
    stacked along one long axis) and `inertia_diag` is its principal inertia
    about ITS OWN COM in the shared frame.

    Returns (total mass, combined z_com, combined inertia_diag about the
    combined COM).  Parallel axis is applied to the two transverse axes (x, y);
    the longitudinal (z) inertias just add, since the shift is along z.
    """
    m_tot = sum(b[0] for b in bodies)
    z_tot = sum(b[0] * b[1] for b in bodies) / m_tot
    out = np.zeros(3)
    for m, z, i_diag in bodies:
        d2 = (z - z_tot) ** 2
        out += np.asarray(i_diag, float) + m * np.array([d2, d2, 0.0])
    return m_tot, z_tot, out


def _principal(table, name: str, length: float, body_mass: float) -> np.ndarray:
    """m * (rg*L)^2 for the three axes of a de Leva row."""
    m_frac, _, rg_s, rg_t, rg_l = table[name]
    return m_frac * body_mass * (np.array([rg_s, rg_t, rg_l]) * length) ** 2


def deleva_upper_body(body_mass: float, torso_length: float,
                      upper_arm_length: tuple[float, float],
                      forearm_length: tuple[float, float],
                      sex: str = "male") -> Skeleton:
    """Skeleton for the 5-segment upper body, from de Leva (1996).

    Segment order (chosen so each arm is a CONTIGUOUS distal->proximal
    sub-chain that `core.slice_chain` can cut out and `core.inverse_dynamics`
    can run directly):

        0  r_forearm_hand   1  r_upper_arm
        2  l_forearm_hand   3  l_upper_arm
        4  torso

    `joint_names` is therefore ["r_elbow", "r_shoulder", "l_elbow",
    "l_shoulder"]; L5S1 is not in this list because it joins segment 4 to the
    PELVIS, which lives in the leg chains.

    FRAME CONVENTION (right-handed, as in `deleva_skeleton`): local z is the
    segment's long axis, local y is the subject's LEFT.  For the arms z points
    DISTAL -> PROXIMAL (wrist->elbow, elbow->shoulder) exactly like shank and
    thigh, and the segment origin is the proximal joint, so `com_local` is
    NEGATIVE along z.  The TORSO IS THE EXCEPTION: its origin is L5S1, at its
    caudal end, and its z points L5S1 -> mid-acromion, so its `com_local` is
    POSITIVE along z.  (The torso's "proximal end" in the graph sense is
    ambiguous -- it has three joints -- so the origin is simply pinned at L5S1,
    the joint it shares with the pelvis.)

    TORSO = de Leva UPPER TRUNK + MID TRUNK + HEAD, lumped rigidly.

      * Mass fraction 0.1596 + 0.1633 + 0.0694 = 0.3923 of body mass (male);
        0.1545 + 0.1465 + 0.0668 = 0.3678 (female).
      * `torso_length` L is the measured L5S1 -> mid-acromion distance.  Three
        geometric assumptions, all approximations of unmarked landmarks:
          (a) L5S1 (our mid-ASIS/SACR point) stands in for de Leva's OMPHALION,
              the shared UPT/MPT/LPT boundary -- already assumed by the pelvis
              segment in `deleva_skeleton`.
          (b) MID-ACROMION stands in for de Leva's SUPRASTERNALE (top of the
              upper trunk) and for CERVICALE/C7 (bottom of the head segment).
              The two real landmarks are within ~3 cm of each other and of the
              acromion midpoint; this is the largest single geometric
              approximation in the torso model.
          (c) XIPHION, the UPT/MPT boundary, is unmarked, so L is split between
              the two sub-segments in de Leva's Table 2 mean-length RATIO
              (male 0.2421 : 0.2155 = 0.529 : 0.471; female 0.2280 : 0.2053 =
              0.526 : 0.474).  Only the ratio is used, never the absolute
              means, and it is close enough to 50:50 that the combined COM
              moves by under 5 mm for any plausible alternative split.
      * The HEAD (de Leva's head segment INCLUDES the neck) is placed along the
        upward extension of the torso long axis, its caudal endpoint
        (cervicale) at mid-acromion, with de Leva's SAMPLE-MEAN head length
        (0.2033 m male / 0.2002 m female) -- there are no head markers, so it
        cannot be subject-scaled.  Its COM is at 59.76% of head length from the
        vertex, i.e. 0.4024 * 0.2033 = 0.0818 m ABOVE mid-acromion (male).
      * Inertia: each part's de Leva principal inertia about its own COM
        (m (rg L_part)^2 per axis), then parallel-axis onto the lumped COM.

    ARMS.  Joint centres ARE THE MARKERS: shoulder = the acromion marker (AC),
    elbow = the epicondyle marker (EP), wrist = the wrist marker (WR).  This
    markerset has no medial/lateral pair anywhere on the arm, so there is no
    way to bisect to a joint centre as the knee and ankle do; the marker sits
    ~2-3 cm lateral/superficial to the true centre.  That offset is a real
    error in the arm moment arms, and it is why arm kinetics from this
    markerset are indicative, not definitive.  Arm masses are ~2.7% (upper) and
    ~2.2% (forearm+hand) of body mass, so the effect on everything proximal is
    small.

    FOREARM+HAND is lumped rigidly: forearm mass 0.0162 with COM at 45.74% of
    forearm length from the elbow, plus hand mass 0.0061 with COM at 79.00% of
    de Leva's SAMPLE-MEAN hand length (0.0862 m male / 0.0780 m female) BEYOND
    the wrist along the forearm axis; inertias combined by parallel axis.  The
    hand is 0.61% of body mass and its COM placement is uncertain by a few cm,
    which moves the whole-body COM by under 2 mm.

    AXIAL INERTIA IS APPROXIMATE for the arms.  Upper arm and forearm are close
    to axisymmetric and this markerset cannot observe their axial rotation
    (single long axis, ML direction borrowed from the torso -- see
    `build_upper_body`), so the x/y (sagittal/transverse) inertia split and the
    axial angular velocity are both nominal.  de Leva's sagittal and transverse
    radii differ by ~6% for the upper arm and ~4% for the forearm, so the
    penalty for getting the split wrong is small; the axial term is genuinely
    unobserved.
    """
    table = DELEVA_MALE if sex.lower().startswith("m") else DELEVA_FEMALE
    means = DELEVA_MEAN_LENGTH["male" if sex.lower().startswith("m")
                               else "female"]

    # --- torso = mid trunk + upper trunk + head, stacked along +z from L5S1
    frac_u = means["upper_trunk"] / (means["upper_trunk"] + means["mid_trunk"])
    len_m = (1.0 - frac_u) * torso_length      # L5S1 -> xiphion
    len_u = frac_u * torso_length              # xiphion -> mid-acromion
    len_h = means["head"]
    # de Leva COM fractions run from each segment's PROXIMAL (cranial for the
    # trunk) endpoint, so convert each to a height above L5S1.
    z_mid = len_m * (1.0 - table["mid_trunk"][1])
    z_up = len_m + len_u * (1.0 - table["upper_trunk"][1])
    z_head = torso_length + len_h * (1.0 - table["head"][1])
    m_torso, z_torso, i_torso = _lump([
        (table["mid_trunk"][0] * body_mass, z_mid,
         _principal(table, "mid_trunk", len_m, body_mass)),
        (table["upper_trunk"][0] * body_mass, z_up,
         _principal(table, "upper_trunk", len_u, body_mass)),
        (table["head"][0] * body_mass, z_head,
         _principal(table, "head", len_h, body_mass)),
    ])

    names = ["r_forearm_hand", "r_upper_arm", "l_forearm_hand", "l_upper_arm",
             "torso"]
    mass = np.zeros(5)
    com = np.zeros((5, 3))
    inertia = np.zeros((5, 3, 3))
    lengths = np.zeros(5)

    for k in (0, 1):                                  # 0 = right, 1 = left
        l_fa = float(forearm_length[k])
        l_ua = float(upper_arm_length[k])
        l_hd = means["hand"]
        # forearm + hand, measured DOWN the arm from the elbow, then negated
        m_fh, z_fh, i_fh = _lump([
            (table["forearm"][0] * body_mass, table["forearm"][1] * l_fa,
             _principal(table, "forearm", l_fa, body_mass)),
            (table["hand"][0] * body_mass, l_fa + table["hand"][1] * l_hd,
             _principal(table, "hand", l_hd, body_mass)),
        ])
        i_fw = 2 * k                                  # forearm slot
        mass[i_fw] = m_fh
        com[i_fw] = np.array([0.0, 0.0, -z_fh])
        inertia[i_fw] = np.diag(i_fh)
        lengths[i_fw] = l_fa

        i_uw = 2 * k + 1                              # upper arm slot
        mass[i_uw] = table["upper_arm"][0] * body_mass
        com[i_uw] = np.array([0.0, 0.0, -table["upper_arm"][1] * l_ua])
        inertia[i_uw] = np.diag(_principal(table, "upper_arm", l_ua, body_mass))
        lengths[i_uw] = l_ua

    mass[4] = m_torso
    com[4] = np.array([0.0, 0.0, z_torso])            # POSITIVE: origin at L5S1
    inertia[4] = np.diag(i_torso)
    lengths[4] = torso_length

    return Skeleton(segment_names=names,
                    joint_names=["r_elbow", "r_shoulder",
                                 "l_elbow", "l_shoulder"],
                    mass=mass, com_local=com, inertia_local=inertia,
                    length=lengths)


def upper_body_points(trial: V3dTrial) -> dict[str, np.ndarray]:
    """Per-frame upper-body endpoints [T,3].

    l5s1         = the same mid(mid-ASIS, SACR) point `joint_centers` uses, so
                   torso and pelvis share the L5S1 joint EXACTLY.
    mid_acromion = mid(RAC, LAC), the torso's cranial end.
    r/l_shoulder = the RAC / LAC marker itself (see `deleva_upper_body`).
    r/l_elbow    = the REP / LEP marker; r/l_wrist = RWR / LWR.
    """
    m = trial.markers
    return {
        "l5s1": 0.5 * (0.5 * (m["RASI"] + m["LASI"]) + m["SACR"]),
        "mid_acromion": 0.5 * (m["RAC"] + m["LAC"]),
        "r_shoulder": m["RAC"], "l_shoulder": m["LAC"],
        "r_elbow": m["REP"], "l_elbow": m["LEP"],
        "r_wrist": m["RWR"], "l_wrist": m["LWR"],
    }


def build_upper_body(trial: V3dTrial, skeleton: Skeleton | None = None,
                     threshold_n: float = 20.0, sex: str = "male"):
    """(Skeleton, SegmentKinematics) for the 5-segment upper body.

    Segment order matches `deleva_upper_body`:
        [r_forearm_hand, r_upper_arm, l_forearm_hand, l_upper_arm, torso]

    Origins (`prox_pos`) and distal ends:
        forearm+hand  prox = elbow (EP),      dist = wrist (WR)
        upper arm     prox = shoulder (AC),   dist = elbow (EP)
        torso         prox = L5S1,            dist = mid-acromion

    MEDIO-LATERAL AXIS.  The torso's ML axis is LAC - RAC (subject's left),
    which is a genuine anatomical pair.  The arms have no ML pair at all, so
    each arm frame is completed with the TORSO's ML direction orthogonalised
    against that arm's long axis.  This is a consistent right-handed
    completion, not an anatomical one: it fixes the arm's axial rotation to
    "follows the shoulders", which is close to true for the swinging arms of
    walking and gets less true the more the humerus rotates internally.  Since
    the arm segments are near-axisymmetric (de Leva sagittal vs transverse
    radii differ by 6% / 4%) the resulting inertia error is small; the frame
    degenerates only if an arm points along the shoulder axis (fully abducted),
    which does not happen in these walking trials.

    If `skeleton` is None one is built with `deleva_upper_body` from
    `estimate_body_mass` and the trial-mean marker-to-marker lengths.
    """
    p = upper_body_points(trial)
    ml_torso = trial.markers["LAC"] - trial.markers["RAC"]

    torso = orthonormal_frame(p["mid_acromion"] - p["l5s1"], ml_torso)
    frames = []
    prox = []
    dist = []
    for s in ("r", "l"):
        frames.append(orthonormal_frame(p[s + "_elbow"] - p[s + "_wrist"],
                                        ml_torso))
        prox.append(p[s + "_elbow"])
        dist.append(p[s + "_wrist"])
        frames.append(orthonormal_frame(p[s + "_shoulder"] - p[s + "_elbow"],
                                        ml_torso))
        prox.append(p[s + "_shoulder"])
        dist.append(p[s + "_elbow"])
    frames.append(torso)
    prox.append(p["l5s1"])
    dist.append(p["mid_acromion"])

    r_world = np.stack(frames, axis=1)                  # [T,5,3,3]
    prox_pos = np.stack(prox, axis=1)                   # [T,5,3]
    dist_pos = np.stack(dist, axis=1)

    if skeleton is None:
        seg_len = np.linalg.norm(dist_pos - prox_pos, axis=2).mean(axis=0)
        skeleton = deleva_upper_body(
            body_mass=estimate_body_mass(trial, threshold_n=threshold_n),
            torso_length=float(seg_len[4]),
            upper_arm_length=(float(seg_len[1]), float(seg_len[3])),
            forearm_length=(float(seg_len[0]), float(seg_len[2])),
            sex=sex)

    kin = SegmentKinematics(t=trial.t.copy(), rate=trial.rate,
                            r_world=r_world, prox_pos=prox_pos,
                            dist_pos=dist_pos)
    return skeleton, kin


def trim_v3d_mat(src: str | os.PathLike, dst: str | os.PathLike,
                 slots=(13,)) -> str:
    """Write a slimmed copy of a ``*_5StridesData.mat`` keeping only `slots`.

    Used to produce the committed sample ``data/p1_trial13_sample.mat``
    (2.5 MB vs 84 MB) — small enough for the repo, and it reproduces the full
    pipeline's numbers bit-for-bit (torque RMS vs Visual3D, residual, energy).

    Round-trips through scipy's record representation, so every field the
    loader reads survives. NOTE: trial ``.index`` reflects position in the
    trimmed file (slot 13 becomes index 0), since the export itself carries
    no slot numbering.
    """
    raw = scipy.io.loadmat(os.fspath(src), struct_as_record=True,
                           squeeze_me=False)
    data = raw["data"]
    sel = data[:, list(slots)] if data.ndim == 2 else data[list(slots)]
    scipy.io.savemat(os.fspath(dst), {"data": sel}, do_compression=True)
    return os.fspath(dst)
