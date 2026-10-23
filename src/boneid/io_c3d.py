"""boneid.io_c3d — read raw c3d trials and calibrate from a STATIC trial.

The Asics-study c3d exports (``S<n>_<shoe>_{Static,PI,RP}.c3d``) differ from
the Visual3D ``*_5StridesData.mat`` format in one important way: they include a
STATIC calibration trial.  The static trial carries anatomical markers that
are REMOVED before the dynamic trials (AnkleMed, KneeMed — plus the extra
ShankAntDist/ThighAntDist cluster markers), so the joint centres they define
cannot be read off the dynamic marker set directly.  Instead:

1.  ``calibrate_static`` measures, in the static pose, the joint centres and
    anatomical segment frames from the full marker set, records each tracking
    CLUSTER's marker positions, and takes body mass from the static vertical
    force.
2.  ``build_chain`` fits each cluster's static shape to its dynamic
    trajectory frame-by-frame (Soderkvist & Wedin SVD fit,
    ``core.fit_rigid_transform``) and carries the static joint centres and
    frames through that pose — the classic virtual-point / CAST approach, the
    same SVD fit the recovered ``VirtualPoint_v2.m`` performs (its ``RM`` is
    the transpose of our ``r_world``; ``boneid.legacy`` ports it exactly).
    The per-frame fit residual is kept as the data-quality signal.

Conventions conserved from the legacy pipeline (the genuine approach):

* hip centre = 0.70*RGT + 0.30*LGT (only the RIGHT leg is instrumented);
* knee = mid(KneeMed, KneeLat), ankle = mid(AnkleMed, AnkleLat), both static;
* clusters: foot [HeelLat HeelMed DorsalFoot], shank [ShankAntProx
  ShankPostDist ShankPostProx], thigh [ThighAntProx ThighPostDist
  ThighPostProx], pelvis [RASIS LASIS RPSIS LPSIS RGT LGT].

Everything else follows io_v3d: SI units, time-major arrays, segment frames
with local z along the long axis and local y to the subject's left, de Leva
segment parameters, and the ground load as a WRENCH about a fixed lab point.

Empirical findings that drive this loader (S1, all three shoe conditions):

* Lab frame: +z up, +x subject-left, +y posterior (subject faces -y) — the
  same convention as the treadmill dataset in io_v3d.  Markers in mm at
  300 Hz; analog at 3000 Hz (two type-2 plates, 6 channels each).
* ezc3d's ``extract_forceplat_data`` returns force [N], COP [mm] and the free
  moment Tz [N mm] already rotated into the LAB frame, reaction-on-subject
  (static standing: Fz ~ +490 N, COP inside the loaded plate, under the
  subject).  COP is NaN wherever the plate is unloaded and must be masked
  before any interpolation.
* BOTH feet land on the same belt (plate 1) in every dynamic trial here, so
  plate loading alone cannot attribute a contact to a limb.
  ``instrumented_foot_wrench`` splits the record into contact bouts and keeps
  a bout only when the plate COP tracks the instrumented foot's markers
  (mean horizontal distance under ``attribute_max_m``); alternating steps of
  the uninstrumented left foot land ~0.3 m away and are dropped.
* Peak vertical force ~3 body weights with no double support: running.
* The named markers have no gaps in these files (only unnamed ``*n``
  reconstructions do, and they are discarded), but the two thigh markers
  SWAP LABELS in several trials (frame-to-frame jumps of 20-30 m/s where
  real motion never exceeds ~5 m/s) — sometimes briefly, in S1_AM_PI
  permanently from t=4.5 s on.  ``load_c3d`` first UNDOES swaps by
  trajectory continuity (``_unswap_labels``), then invalidates any leftover
  excursions by speed threshold (``_despike``) and bridges them by linear
  interpolation; the cluster fit residual still shows the scars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .core import (
    ForcePlate,
    GroundWrench,
    SegmentKinematics,
    Skeleton,
    detect_contact,
)
from .io_v3d import (analog_to_mocap, append_deleva_torso, deleva_skeleton,
                     orthonormal_frame)

#: Tracking clusters (all present in BOTH static and dynamic trials), keyed by
#: the chain's segment order [foot, shank, thigh, pelvis].
CLUSTERS = {
    "foot": ("HeelLat", "HeelMed", "DorsalFoot"),
    "shank": ("ShankAntProx", "ShankPostDist", "ShankPostProx"),
    "thigh": ("ThighAntProx", "ThighPostDist", "ThighPostProx"),
    "pelvis": ("RASIS", "LASIS", "RPSIS", "LPSIS", "RGT", "LGT"),
}

SEGMENTS = ("foot", "shank", "thigh", "pelvis")

#: OPTIONAL torso tracking.  The Asics study put no markers above the pelvis,
#: so its chain stops there and the L5S1 residual carries the whole upper
#: body.  When a static trial DOES carry these (the idealized subject in
#: `simulate.simulate_squat_to_stand_c3d` does), the chain gains a fifth
#: segment: the de Leva torso (trunk + head) from L5S1 to mid-acromion,
#: tracked by `TORSO_CLUSTER`, its frame from the acromions (RAC, LAC).
TORSO_CLUSTER = ("C7", "CLAV", "STRN", "T10")
TORSO_LANDMARKS = ("RAC", "LAC")

#: Which calibrated points ride on which cluster during dynamic trials.  The
#: joint shared by two segments is reconstructed from ONE cluster (the legacy
#: choice): ankle from the foot, knee from the shank, hip from the thigh.
CLUSTER_POINTS = {
    "foot": ("ankle", "toe", "heel"),
    "shank": ("knee",),
    "thigh": ("hip",),
    "pelvis": ("l5s1", "mid_hip", "hip_l"),
}


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------

@dataclass
class C3dTrial:
    """One c3d recording in SI units, lab frame, reaction-on-subject."""

    path: str
    rate: float                         # Hz, mocap
    analog_rate: float                  # Hz, force
    t: np.ndarray                       # [T] s
    t_analog: np.ndarray                # [Ta] s
    markers: dict[str, np.ndarray]      # name -> [T,3] m
    force: np.ndarray                   # [Ta,P,3] N, lab frame
    cop: np.ndarray                     # [Ta,P,3] m, lab frame (NaN unloaded)
    free_moment: np.ndarray             # [Ta,P,3] N m, lab frame
    plates: list[ForcePlate]            # [P]


@dataclass
class StaticCalibration:
    """Everything a dynamic trial needs from the static trial.

    ``cluster_neutral[seg]`` is the cluster's marker positions in the static
    pose (lab frame, [M,3]); ``points[name]`` are the calibrated joint
    centres/endpoints in the same pose; ``r_static`` are the anatomical
    segment frames there.  ``build_chain`` moves all of it with the cluster.
    """

    path: str
    side: str                           # 'r' — the instrumented leg
    sex: str
    body_mass: float                    # kg, from the static vertical force
    cluster_neutral: dict[str, np.ndarray]   # seg -> [M,3] m, static pose
    points: dict[str, np.ndarray]       # name -> [3] m, static pose
    r_static: np.ndarray                # [4,3,3] world_from_segment, static
    lengths: dict[str, float]           # foot/shank/thigh/pelvis [m]
    foot_heel_local: np.ndarray         # [3] heel in the foot frame re ankle
    #: which markers form each cluster / which points ride on which cluster;
    #: None = this module's Asics defaults (CLUSTERS / CLUSTER_POINTS).
    cluster_markers: dict | None = None
    point_carriers: dict | None = None
    foot_outline_local: np.ndarray | None = None  # drawing landmarks [M,3]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Bridge NaN runs per column by linear interpolation (ends clamped)."""
    x = np.array(x, dtype=float)
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.any() and not bad.all():
            idx = np.arange(len(x))
            x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x


def _unswap_labels(markers: dict[str, np.ndarray], rate: float,
                   max_speed: float = 10.0, tol: float = 0.06):
    """Undo marker LABEL SWAPS by trajectory continuity.

    Two markers that exchange labels each show a frame-to-frame jump (well
    above any real speed) at the same instant, and each lands on the other's
    pre-jump position.  When that signature matches, the trajectory TAILS are
    swapped back from the jump onward, restoring both markers' identities for
    the rest of the trial — a permanent swap cannot be fixed by
    interpolation, only by unswapping.  `_despike` afterwards cleans the one
    or two midway frames the tracker interpolated during the exchange.

    Returns the list of applied swaps [(frame, name_a, name_b)].
    """
    events: list[tuple[int, str]] = []
    for n, x in markers.items():
        v = np.linalg.norm(np.diff(x, axis=0), axis=1) * rate
        events.extend((int(f), n) for f in np.flatnonzero(v > max_speed))
    events.sort()

    swaps = []
    done: list[tuple[int, str]] = []           # boundaries already explained
    for f, a in events:
        if (f, a) in done:
            continue
        xa = markers[a]
        lo, hi = f - 1, f + 3                  # covers 1-2 frame exchanges
        if lo < 1 or hi >= len(xa):
            continue
        # The partner's own jump may hide below `max_speed` (trackers
        # interpolate the exchange over two frames), so test EVERY other
        # marker.  The segment keeps moving during the exchange (a running
        # thigh covers several cm in 3 frames), so compare against a
        # constant-velocity extrapolation, and swap only when the crossed
        # assignment clearly beats keeping the labels as they are.
        pa = xa[lo] + (hi - lo) * (xa[lo] - xa[lo - 1])
        best = None
        for b, xb in markers.items():
            if b == a:
                continue
            pb = xb[lo] + (hi - lo) * (xb[lo] - xb[lo - 1])
            d_cross = (np.linalg.norm(xa[hi] - pb)
                       + np.linalg.norm(xb[hi] - pa))
            d_keep = (np.linalg.norm(xa[hi] - pa)
                      + np.linalg.norm(xb[hi] - pb))
            if (d_cross < 2 * tol and d_cross < 0.5 * d_keep
                    and (best is None or d_cross < best[0])):
                best = (d_cross, b)
        if best is not None:
            b = best[1]
            xb = markers[b]
            cut = f + 1
            xa[cut:], xb[cut:] = xb[cut:].copy(), xa[cut:].copy()
            swaps.append((cut, a, b))
            done.extend((g, m) for g, m in events
                        if abs(g - f) <= 3 and m in (a, b))
    return swaps


def _despike(x: np.ndarray, rate: float, max_speed: float = 10.0,
             max_span_s: float = 0.2) -> np.ndarray:
    """NaN out label-swap excursions, leaving `_fill_gaps` to bridge them.

    A frame-to-frame speed above `max_speed` marks a glitch boundary; real
    motion in these running trials never exceeds ~5 m/s (99.9th percentile)
    while a momentary label swap between two thigh markers ~0.1 m apart reads
    as 20-30 m/s at 300 Hz.  Two boundaries within `max_span_s` of each other
    bracket a swapped/displaced span, which is invalidated whole; an isolated
    boundary invalidates just its two neighbouring frames.
    """
    v = np.linalg.norm(np.diff(x, axis=0), axis=1) * rate
    bounds = list(np.flatnonzero(v > max_speed))
    max_span = int(round(max_span_s * rate))
    k = 0
    while k < len(bounds):
        if k + 1 < len(bounds) and bounds[k + 1] - bounds[k] <= max_span:
            x[bounds[k] + 1:bounds[k + 1] + 1] = np.nan
            k += 2
        else:
            x[bounds[k]:bounds[k] + 2] = np.nan
            k += 1
    return x


def _plates_from_corners(corners: np.ndarray) -> list[ForcePlate]:
    """ForcePlate records from per-plate corner arrays [P][3,4] in mm."""
    plates: list[ForcePlate] = []
    for cs in corners:
        c = np.asarray(cs, dtype=float).T / 1000.0       # [4,3] m
        e1 = c[1] - c[0]
        e2 = c[0] - c[3]
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 - e1 * (e2 @ e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        plates.append(ForcePlate(corners=c, origin=c.mean(axis=0),
                                 r_lab=np.stack([e1, e2, e3], axis=1)))
    return plates


#: lab_from_file rotation for recordings whose vertical is +Y (Visual3D's
#: default lab): maps (x, y, z)_file -> (x, -z, y), i.e. keeps the file's x,
#: sends its up-axis y to our z (up), and its z to -y — right-handed.  For
#: the RBDS treadmill this lands on x anterior, y subject-left, z up.
Y_UP_AXES = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])


def load_c3d(path: str | os.PathLike,
             axes: np.ndarray | None = None, repair: bool = True) -> C3dTrial:
    """Read one c3d file: markers in m, plate loads in the lab frame in SI.

    Unnamed markers (``*n`` — reconstruction leftovers) are dropped.  Force,
    COP and free moment come from ezc3d's force-platform extraction (lab
    frame, reaction on the subject); COP stays NaN where the plate is
    unloaded, exactly as delivered — mask before use.

    `axes` (optional [3,3] lab_from_file rotation, e.g. `Y_UP_AXES`) is
    applied to every point and force-plate vector, for recordings whose lab
    frame is not z-up.  core.py's gravity points down our z; rotate here
    rather than teaching the rest of the pipeline about other conventions.

    `repair=False` skips the label-swap repair, despiking and gap bridging —
    the raw markers exactly as the legacy MATLAB read them (`boneid.legacy`).
    """
    import ezc3d

    path = os.fspath(path)
    c = ezc3d.c3d(path, extract_forceplat_data=True)
    par = c["parameters"]
    rate = float(np.atleast_1d(par["POINT"]["RATE"]["value"])[0])
    analog_rate = float(np.atleast_1d(par["ANALOG"]["RATE"]["value"])[0])
    unit = str(np.atleast_1d(par["POINT"]["UNITS"]["value"])[0]).strip()
    if unit != "mm":
        raise ValueError(f"{path}: expected marker units mm, got {unit!r}")
    r = np.eye(3) if axes is None else np.asarray(axes, dtype=float)

    labels = [str(s).strip() for s in par["POINT"]["LABELS"]["value"]]
    pts = c["data"]["points"]                            # [4,M,T]
    markers = {name: (pts[:3, i].T / 1000.0) @ r.T
               for i, name in enumerate(labels) if not name.startswith("*")}
    if repair:
        _unswap_labels(markers, rate)
        markers = {name: _fill_gaps(_despike(x, rate))
                   for name, x in markers.items()}

    plats = c["data"]["platform"]
    force = np.stack([np.asarray(p["force"], float).T @ r.T for p in plats],
                     axis=1)
    cop = np.stack([np.asarray(p["center_of_pressure"], float).T @ r.T
                    for p in plats], axis=1) / 1000.0
    free = np.stack([np.asarray(p["Tz"], float).T @ r.T for p in plats],
                    axis=1) / 1000.0
    n_t = next(iter(markers.values())).shape[0]

    return C3dTrial(
        path=path, rate=rate, analog_rate=analog_rate,
        t=np.arange(n_t) / rate,
        t_analog=np.arange(force.shape[0]) / analog_rate,
        markers=markers, force=force, cop=cop, free_moment=free,
        plates=_plates_from_corners([r @ np.asarray(p["corners"], float)
                                     for p in plats]))


#: Corners [4,3] (m, about the plate centre) of the one plate `write_c3d`
#: writes: 1.0 m along lab x by 0.5 m along lab y, surface in the plane of
#: the centre.  With this corner order ezc3d's type-2 plate frame is
#: diag(1, -1, -1) of the lab, which is what `write_c3d` inverts.
WRITE_PLATE_CORNERS = np.array([[0.5, -0.25, 0.0], [-0.5, -0.25, 0.0],
                                [-0.5, 0.25, 0.0], [0.5, 0.25, 0.0]])


def write_c3d(path: str | os.PathLike, markers: dict[str, np.ndarray],
              rate: float, analog_rate: float, force: np.ndarray,
              moment: np.ndarray, plate_center: np.ndarray | None = None):
    """Write markers and ONE force plate as a c3d that `load_c3d` reads back.

    `markers` name -> [T,3] m (lab frame); `force` [Ta,3] N and `moment`
    [Ta,3] N m are the load ON THE SUBJECT, the moment taken about
    `plate_center` (default the lab origin), which is also the centre of the
    plate surface.  ``analog_rate`` must be a whole multiple of ``rate``.
    The plate is a type-2 (six-channel) platform with corners
    `WRITE_PLATE_CORNERS`; ezc3d then derives COP and the free moment from
    the channels exactly as it does for a lab recording.  This exists so a
    simulated body goes through the SAME loader as real data.
    """
    import ezc3d
    from ezc3d import ezc3d as _swig

    center = np.zeros(3) if plate_center is None else np.asarray(plate_center, float)
    names = list(markers)
    n_t = len(markers[names[0]])
    ratio = int(round(analog_rate / rate))
    if len(force) != ratio * n_t:
        raise ValueError("need analog_rate/rate analog samples per frame")

    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [float(rate)]
    c["parameters"]["POINT"]["LABELS"]["value"] = tuple(names)
    c["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    pts = np.ones((4, len(names), n_t))
    pts[:3] = np.stack([markers[n] * 1000.0 for n in names], axis=1).T
    c["data"]["points"] = pts

    c["parameters"]["ANALOG"]["RATE"]["value"] = [float(analog_rate)]
    c["parameters"]["ANALOG"]["LABELS"]["value"] = ("Fx1", "Fy1", "Fz1",
                                                    "Mx1", "My1", "Mz1")
    c["parameters"]["ANALOG"]["UNITS"]["value"] = ("N", "N", "N",
                                                   "Nmm", "Nmm", "Nmm")
    flip = np.array([1.0, -1.0, -1.0])        # lab -> plate channel axes
    c["data"]["analogs"] = np.concatenate(
        [force * flip, 1000.0 * moment * flip], axis=1).T[None]

    def add_int(name, value):
        # ezc3d's add_parameter stores int lists as FLOAT, which its own
        # force-plate reader then rejects: set the INT type explicitly.
        value = np.asarray(value, dtype=int)
        c.add_parameter("FORCE_PLATFORM", name, value.astype(float))
        c["parameters"]["FORCE_PLATFORM"][name]["value"] = value
        c["parameters"]["FORCE_PLATFORM"][name]["type"] = _swig.INT

    add_int("USED", [1])
    add_int("TYPE", [2])
    add_int("ZERO", [1, 0])
    add_int("CHANNEL", [[1], [2], [3], [4], [5], [6]])
    corners = (WRITE_PLATE_CORNERS + center) * 1000.0
    c.add_parameter("FORCE_PLATFORM", "CORNERS", corners.T[:, :, None])
    c.add_parameter("FORCE_PLATFORM", "ORIGIN", np.zeros((3, 1)))
    c.write(os.fspath(path))


# ---------------------------------------------------------------------------
# Static calibration
# ---------------------------------------------------------------------------

def static_points(marker_mean: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Joint centres and segment endpoints in the static pose.

    ankle = mid(AnkleMed, AnkleLat); knee = mid(KneeMed, KneeLat);
    hip = 0.70*RGT + 0.30*LGT (the legacy regression toward the instrumented
    right side; ``hip_l`` mirrors it for drawing the pelvis);
    heel = mid(HeelMed, HeelLat); toe = mid(MTP1Head, MTP5Head) (the
    metatarsal centre, the foot's distal endpoint);
    l5s1 = mid(mid-ASIS, mid-PSIS); mid_hip = mid(RGT, LGT).
    """
    m = marker_mean
    mid_asis = 0.5 * (m["RASIS"] + m["LASIS"])
    mid_psis = 0.5 * (m["RPSIS"] + m["LPSIS"])
    return {
        "ankle": 0.5 * (m["AnkleMed"] + m["AnkleLat"]),
        "knee": 0.5 * (m["KneeMed"] + m["KneeLat"]),
        "hip": 0.70 * m["RGT"] + 0.30 * m["LGT"],
        "hip_l": 0.70 * m["LGT"] + 0.30 * m["RGT"],
        "mid_hip": 0.5 * (m["RGT"] + m["LGT"]),
        "l5s1": 0.5 * (mid_asis + mid_psis),
        "heel": 0.5 * (m["HeelMed"] + m["HeelLat"]),
        "toe": 0.5 * (m["MTP1Head"] + m["MTP5Head"]),
        "toebox": m["ToeboxAnt"],   # shoe tip; drawing only, never dynamics
        **({"mid_acromion": 0.5 * (m["RAC"] + m["LAC"])}
           if all(n in m for n in TORSO_LANDMARKS) else {}),
    }


def has_torso(markers) -> bool:
    """True when the torso cluster and acromions are all marked."""
    return all(n in markers for n in TORSO_CLUSTER + TORSO_LANDMARKS)


def chain_segments(cal: StaticCalibration) -> tuple[str, ...]:
    """The chain this calibration supports: SEGMENTS, plus the torso when
    the static trial marked it."""
    return SEGMENTS + (("torso",) if "torso" in cal.cluster_neutral else ())


def calibrate_static(static: C3dTrial, sex: str = "male",
                     gravity: float = 9.81,
                     body_mass: float | None = None) -> StaticCalibration:
    """Measure the subject in the static pose.

    Markers are averaged over the whole static recording; joint centres and
    anatomical frames follow `static_points` and the frame convention of
    ``io_v3d.deleva_skeleton`` (local z along the long axis pointing
    proximal — heel->toe for the foot — local y to the subject's LEFT; the
    right leg's medial markers lie to the subject's left of the lateral
    ones).  Body mass is the summed mean vertical force over both plates.

    ``lengths['foot']`` uses the anterior toebox marker (heel -> ToeboxAnt),
    the closest thing this shoe-mounted markerset has to de Leva's heel->toe
    tip length; the chain's working distal endpoint stays the metatarsal
    centre.

    `body_mass` (kg), when given, replaces the plate estimate: the mass the
    de Leva fractions are taken of, e.g. from a scale.  The plate only knows
    what is standing on it.

    If the static trial marks the torso (`has_torso`), it is calibrated too:
    its frame from L5S1 -> mid-acromion and the acromions, its cluster
    shape, and ``lengths['torso']``.
    """
    mean = {k: v.mean(axis=0) for k, v in static.markers.items()}
    pts = static_points(mean)
    torso = has_torso(mean)

    ml_ankle = mean["AnkleMed"] - mean["AnkleLat"]
    ml_knee = mean["KneeMed"] - mean["KneeLat"]
    ml_pelvis = mean["LASIS"] - mean["RASIS"]
    r_static = np.stack([
        orthonormal_frame(pts["toe"] - pts["heel"], ml_ankle),
        orthonormal_frame(pts["knee"] - pts["ankle"], ml_ankle),
        orthonormal_frame(pts["hip"] - pts["knee"], ml_knee),
        orthonormal_frame(pts["l5s1"] - pts["mid_hip"], ml_pelvis),
    ] + ([orthonormal_frame(pts["mid_acromion"] - pts["l5s1"],
                            mean["LAC"] - mean["RAC"])] if torso else []))

    lengths = {
        "foot": float(np.linalg.norm(mean["ToeboxAnt"] - pts["heel"])),
        "shank": float(np.linalg.norm(pts["knee"] - pts["ankle"])),
        "thigh": float(np.linalg.norm(pts["hip"] - pts["knee"])),
        "pelvis": float(np.linalg.norm(pts["l5s1"] - pts["mid_hip"])),
        # forefoot width, used only to size the drawn foot wedge
        "forefoot_width": float(np.linalg.norm(mean["MTP1Head"]
                                               - mean["MTP5Head"])),
    }
    if torso:
        lengths["torso"] = float(np.linalg.norm(pts["mid_acromion"]
                                                - pts["l5s1"]))
    heel_local = r_static[0].T @ (pts["heel"] - pts["ankle"])

    if body_mass is None:
        body_mass = float(static.force[:, :, 2].sum(axis=1).mean() / gravity)
    clusters = {**CLUSTERS, **({"torso": TORSO_CLUSTER} if torso else {})}
    neutral = {seg: np.stack([mean[n] for n in clusters[seg]])
               for seg in clusters}
    return StaticCalibration(
        path=static.path, side="r", sex=sex, body_mass=float(body_mass),
        cluster_neutral=neutral, points=pts, r_static=r_static,
        lengths=lengths, foot_heel_local=heel_local,
        cluster_markers=clusters if torso else None,
        point_carriers=({**CLUSTER_POINTS, "torso": ("mid_acromion",)}
                        if torso else None),
        foot_outline_local=np.stack([
            r_static[0].T @ (mean[n] - pts["ankle"])
            for n in ("HeelLat", "HeelMed", "MTP1Head", "MTP5Head",
                      "ToeboxAnt", "AnkleMed", "AnkleLat")]))


# ---------------------------------------------------------------------------
# Virtual points: carry the calibration through the dynamic trial
# ---------------------------------------------------------------------------

def cluster_poses(trial: C3dTrial, cal: StaticCalibration):
    """Fit each static cluster to its dynamic trajectory.

    Returns ``{seg: (r [T,3,3], d [T,3], res [T])}`` with
    ``marker(t) ~= r @ marker_static + d``; ``res`` is the RMS fit residual
    per frame [m] — the rigidity/data-quality signal every run reports.
    """
    from .core import fit_rigid_transform

    clusters = cal.cluster_markers or CLUSTERS
    out = {}
    for seg in chain_segments(cal):
        moving = np.stack([trial.markers[n] for n in clusters[seg]], axis=1)
        out[seg] = fit_rigid_transform(cal.cluster_neutral[seg], moving)
    return out


def virtual_points(poses, cal: StaticCalibration) -> dict[str, np.ndarray]:
    """[T,3] trajectory of every calibrated point, carried by its cluster."""
    out = {}
    carriers = cal.point_carriers or CLUSTER_POINTS
    for seg, names in carriers.items():
        if seg not in poses:
            continue
        r, d, _ = poses[seg]
        for name in names:
            if name in cal.points:
                out[name] = np.einsum("tij,j->ti", r, cal.points[name]) + d
    return out


# ---------------------------------------------------------------------------
# Ground wrench with per-bout foot attribution
# ---------------------------------------------------------------------------

def foot_centroid(trial: C3dTrial) -> np.ndarray:
    """[T,3] mean of the instrumented foot's markers (attribution anchor)."""
    names = ("HeelLat", "HeelMed", "DorsalFoot", "MTP1Head", "MTP5Head")
    return np.mean([trial.markers[n] for n in names], axis=0)


def instrumented_foot_wrench(trial: C3dTrial,
                             point: np.ndarray | None = None,
                             threshold_n: float = 50.0,
                             attribute_max_m: float = 0.25,
                             antialias_hz: float = 50.0,
                             min_gap_s: float = 0.015,
                             min_peak_n: float = 250.0):
    """Ground wrench of the INSTRUMENTED foot only, plus the bout table.

    Both feet strike the same belt in these trials, so each contact bout (a
    connected run of vertical force above `threshold_n`, gaps shorter than
    `min_gap_s` bridged) is attributed by geometry: the bout belongs to the
    instrumented foot when the plate COP's mean horizontal distance to the
    foot-marker centroid is under `attribute_max_m` (measured ~0.05-0.10 m
    for own steps vs ~0.3 m for the other foot's).  Unattributed bouts are
    zeroed, so during the other foot's stance — when the instrumented leg
    genuinely carries no external load — the wrench is exactly zero.

    Bouts peaking below `min_peak_n` are belt vibration, not steps, and are
    ignored outright: after toe-off the unloaded belt rings at 50-100 N,
    while every genuine running stance here peaks above 1.2 kN.  `min_gap_s`
    must stay SHORTER than that ringing (~20-30 ms flickers) or consecutive
    left/right stances get bridged into one unattributable bout.

    The wrench is formed at the analog rate about the fixed lab `point`
    (default: the lab origin), ``moment = (cop - point) x force + Tz``, then
    anti-alias filtered and resampled to the mocap clock like io_v3d.

    Returns ``(GroundWrench, bouts)`` where each bout is a dict with keys
    ``plate, t_on, t_off, mine, cop_dist_m`` — report it; every step the
    pipeline drops should be visible, not silent.
    """
    if point is None:
        point = np.zeros(3)
    point = np.asarray(point, dtype=float)
    cent = foot_centroid(trial)
    cent_a = np.stack([np.interp(trial.t_analog, trial.t, cent[:, c])
                       for c in range(2)], axis=1)       # [Ta,2]

    n_a, n_p = trial.force.shape[:2]
    f_out = np.zeros((n_a, 3))
    m_out = np.zeros((n_a, 3))
    bouts = []
    min_gap = int(round(min_gap_s * trial.analog_rate))
    for p in range(n_p):
        f = trial.force[:, p]
        cop = trial.cop[:, p]
        free = trial.free_moment[:, p]
        _, events = detect_contact(f[:, 2], threshold_n, min_gap=min_gap)
        for s, e in events:
            sl = slice(s, e + 1)
            peak = f[sl, 2].max()
            if peak < min_peak_n:
                continue                       # belt vibration, not a step
            # judge COP distance only where the plate carries real load
            # (>50% of the bout's peak): at the stance edges the COP is
            # noisy and smears the left/right separation.
            ok = np.isfinite(cop[sl, 0]) & (f[sl, 2] > 0.5 * peak)
            d = float(np.mean(np.linalg.norm(
                cop[sl][ok][:, :2] - cent_a[sl][ok], axis=1)))
            mine = d < attribute_max_m
            bouts.append({"plate": p, "t_on": s / trial.analog_rate,
                          "t_off": e / trial.analog_rate, "mine": mine,
                          "cop_dist_m": d})
            if mine:
                # ezc3d leaves COP and Tz NaN on below-threshold samples,
                # which a bridged bout can contain at its edges: zero them.
                c = np.where(np.isfinite(cop[sl]), cop[sl], 0.0)
                fr = np.where(np.isfinite(free[sl]), free[sl], 0.0)
                f_out[sl] += f[sl]
                m_out[sl] += np.cross(c - point, f[sl]) + fr
    ground = GroundWrench(t=trial.t.copy(), rate=trial.rate,
                          force=analog_to_mocap(trial, f_out, antialias_hz),
                          moment=analog_to_mocap(trial, m_out, antialias_hz),
                          point=point)
    return ground, sorted(bouts, key=lambda b: b["t_on"])


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def build_chain(trial: C3dTrial, cal: StaticCalibration,
                point: np.ndarray | None = None,
                threshold_n: float = 50.0):
    """(Skeleton, SegmentKinematics, GroundWrench, quality) for
    [foot, shank, thigh, pelvis] (+ torso, when the static trial marked it),
    everything on the trial's mocap clock.

    Joint centres and segment frames are the STATIC calibration carried by
    the per-frame cluster fits: ``r_world(t) = r_cluster(t) @ r_static`` and
    ``p(t) = r_cluster(t) @ p_static + d(t)``.  Segment origins are the
    proximal joints as core.py requires (foot: ankle->toe, shank:
    knee->ankle, thigh: hip->knee, pelvis: l5s1->mid-hip); the Skeleton comes
    from `deleva_skeleton` with the STATIC lengths and the static body mass —
    the calibration trial, not the dynamic one, defines the subject.  A
    marked torso (mid-acromion -> l5s1) is appended with
    `append_deleva_torso`, and L5S1 becomes a real joint.

    ``quality`` is a dict: ``cluster_residual`` [T,S] m (the SVD fit
    residual per segment) and ``bouts`` (the contact-attribution table from
    `instrumented_foot_wrench`).
    """
    segs = chain_segments(cal)
    torso = "torso" in segs
    poses = cluster_poses(trial, cal)
    pts = virtual_points(poses, cal)

    r_world = np.stack([np.einsum("tij,jk->tik", poses[seg][0],
                                  cal.r_static[i])
                        for i, seg in enumerate(segs)], axis=1)
    prox = np.stack([pts["ankle"], pts["knee"], pts["hip"], pts["l5s1"]]
                    + ([pts["mid_acromion"]] if torso else []), axis=1)
    dist = np.stack([pts["toe"], pts["ankle"], pts["knee"], pts["mid_hip"]]
                    + ([pts["l5s1"]] if torso else []), axis=1)

    skeleton = deleva_skeleton(
        body_mass=cal.body_mass, foot_length=cal.lengths["foot"],
        shank_length=cal.lengths["shank"], thigh_length=cal.lengths["thigh"],
        pelvis_length=cal.lengths["pelvis"],
        foot_heel_local=cal.foot_heel_local, sex=cal.sex)
    if torso:
        skeleton = append_deleva_torso(skeleton, cal.body_mass,
                                       cal.lengths["torso"], sex=cal.sex)
    kin = SegmentKinematics(t=trial.t.copy(), rate=trial.rate,
                            r_world=r_world, prox_pos=prox, dist_pos=dist)
    ground, bouts = instrumented_foot_wrench(trial, point=point,
                                             threshold_n=threshold_n)
    quality = {
        "cluster_residual": np.stack([poses[seg][2] for seg in segs],
                                     axis=1),
        "bouts": bouts,
        "points": pts,
    }
    return skeleton, kin, ground, quality


# ---------------------------------------------------------------------------
# RBDS: the Fukuchi, Fukuchi & Duarte (2017) public running dataset
# ---------------------------------------------------------------------------
#
# "A public dataset of running biomechanics and the effects of running speed
# on lower extremity kinematics and kinetics", PeerJ 5:e3298; raw data at
# https://doi.org/10.6084/m9.figshare.4543435 (CC-BY): 28 regular runners on
# a single-belt instrumented treadmill at 2.5/3.5/4.5 m/s, 150 Hz mocap /
# 300 Hz analog, one force platform under the belt.  File lab frame is
# Y-UP (x anterior, z subject-right): load with ``axes=Y_UP_AXES``.
#
# The marker protocol is the same CAST pattern as the Asics study, BILATERAL:
# the static trial adds anatomical markers (Ankle/Ankle.Medial,
# Knee/Knee.Medial, GTR, HF, TT, MT2) that are removed for running, while
# the tracking clusters (heel triad, 4-marker shank/thigh shells, pelvis
# ASIS/PSIS/Iliac.Crest) survive.  Both feet strike the ONE belt, so contact
# bouts are attributed to whichever foot's markers the COP tracks.
#
# The static c3d carries NO force data, so body mass comes from the caller
# (RBDSinfo.txt) or `estimate_body_mass_grf` on a running trial.

RBDS_PELVIS_CLUSTER = ("R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS",
                       "R.Iliac.Crest", "L.Iliac.Crest")


def rbds_clusters(side: str) -> dict[str, tuple]:
    p = "R." if side.lower().startswith("r") else "L."
    return {
        "foot": (p + "Heel.Bottom", p + "Heel.Lateral", p + "Heel.Top"),
        "shank": (p + "Shank.Top.Lateral", p + "Shank.Top.Medial",
                  p + "Shank.Bottom.Lateral", p + "Shank.Bottom.Medial"),
        "thigh": (p + "Thigh.Top.Lateral", p + "Thigh.Top.Medial",
                  p + "Thigh.Bottom.Lateral", p + "Thigh.Bottom.Medial"),
        "pelvis": RBDS_PELVIS_CLUSTER,
    }


def estimate_body_mass_grf(trial: C3dTrial, gravity: float = 9.81) -> float:
    """Body mass [kg] as the whole-record mean total vertical force / g.

    Valid when every foot contact lands on the recorded plate(s) — true for
    the RBDS single-belt treadmill — since over many strides the mean ground
    reaction supports exactly body weight."""
    return float(trial.force[:, :, 2].sum(axis=1).mean() / gravity)


def _harrington_hip(mean: dict[str, np.ndarray], side: str) -> np.ndarray:
    """Static hip joint centre, Harrington et al. (2007) all-subjects
    regression (the same constants as report.hip_center, static pose)."""
    mid_asis = 0.5 * (mean["R.ASIS"] + mean["L.ASIS"])
    mid_psis = 0.5 * (mean["R.PSIS"] + mean["L.PSIS"])
    e_left = mean["L.ASIS"] - mean["R.ASIS"]
    pw = float(np.linalg.norm(e_left))
    e_left = e_left / pw
    ant = mid_asis - mid_psis
    ant = ant - e_left * (ant @ e_left)
    pd = float(np.linalg.norm(ant))
    e_ant = ant / pd
    e_up = np.cross(e_ant, e_left)
    e_lat = e_left * (-1.0 if side.lower().startswith("r") else 1.0)
    d_ant = (-0.24 * (pd * 1000.0) - 9.9) / 1000.0
    d_up = (-0.30 * (pw * 1000.0) - 10.9) / 1000.0
    d_lat = (0.33 * (pw * 1000.0) + 7.3) / 1000.0
    return mid_asis + d_ant * e_ant + d_up * e_up + d_lat * e_lat


def _bell_hip(mean: dict[str, np.ndarray], side: str) -> np.ndarray:
    """Static hip centre, Bell, Pedersen & Brand (1990) — from mid-ASIS,
    posterior 0.19·PW, distal 0.30·PW, lateral 0.36·PW.  This is EXACTLY the
    landmark the RBDS Visual3D model file defines (RBDSmodelV3D.mdh:
    MCS_ML=0.36·ASIS_Distance, MCS_AP=−0.19, MCS_AXIAL=−0.30), so it is the
    dataset's own hip convention."""
    mid_asis = 0.5 * (mean["R.ASIS"] + mean["L.ASIS"])
    mid_psis = 0.5 * (mean["R.PSIS"] + mean["L.PSIS"])
    e_left = mean["L.ASIS"] - mean["R.ASIS"]
    pw = float(np.linalg.norm(e_left))
    e_left = e_left / pw
    ant = mid_asis - mid_psis
    ant = ant - e_left * (ant @ e_left)
    e_ant = ant / np.linalg.norm(ant)
    e_up = np.cross(e_ant, e_left)
    e_lat = e_left * (-1.0 if side.lower().startswith("r") else 1.0)
    return mid_asis - 0.19 * pw * e_ant - 0.30 * pw * e_up + 0.36 * pw * e_lat

RBDS_HIP_METHODS = {"harrington": _harrington_hip, "bell": _bell_hip}


def rbds_static_points(mean: dict[str, np.ndarray], side: str,
                       hip_method: str = "harrington"
                       ) -> dict[str, np.ndarray]:
    """Joint centres/endpoints for one side in the RBDS static pose.

    ankle = mid(Ankle, Ankle.Medial); knee = mid(Knee, Knee.Medial);
    hip = Harrington (2007) regression from the pelvis markers (GTR exists
    too, but Harrington is the better-validated convention this markerset
    supports and closer to what Visual3D models use); heel = Heel.Bottom;
    toe = mid(MT1, MT5) — the metatarsal centre, so ``lengths['foot']`` is a
    METATARSAL length, a little short of de Leva's heel-to-toe-tip (the foot
    is 1.4% of body mass; the inertia error this causes is negligible);
    l5s1 = mid(mid-ASIS, mid-PSIS); mid_hip = mid of the two regressed hips.
    """
    p = "R." if side.lower().startswith("r") else "L."
    hjc = RBDS_HIP_METHODS[hip_method]
    hip = hjc(mean, side)
    hip_other = hjc(mean, "l" if p == "R." else "r")
    mid_asis = 0.5 * (mean["R.ASIS"] + mean["L.ASIS"])
    mid_psis = 0.5 * (mean["R.PSIS"] + mean["L.PSIS"])
    return {
        "ankle": 0.5 * (mean[p + "Ankle"] + mean[p + "Ankle.Medial"]),
        "knee": 0.5 * (mean[p + "Knee"] + mean[p + "Knee.Medial"]),
        "hip": hip,
        "hip_l": hip_other,       # the contralateral hip (viz pelvis band)
        "mid_hip": 0.5 * (hip + hip_other),
        "l5s1": 0.5 * (mid_asis + mid_psis),
        "heel": mean[p + "Heel.Bottom"],
        "toe": 0.5 * (mean[p + "MT1"] + mean[p + "MT5"]),
    }


def calibrate_rbds_static(static: C3dTrial, body_mass: float,
                          side: str = "r", sex: str = "male",
                          hip_method: str = "harrington") -> StaticCalibration:
    """Static calibration for one RBDS leg (call once per side).

    Frame convention as everywhere in this repo (local z distal->proximal,
    local y to the subject's LEFT; the medial marker of a RIGHT-side pair
    lies to the subject's left of the lateral one, flipped on the left).
    `body_mass` must be supplied — the RBDS static c3d has no force data.
    `hip_method` is 'harrington' (our default) or 'bell' (the dataset's own
    Visual3D landmark, see `_bell_hip`).
    """
    mean = {k: np.nanmean(v, axis=0) for k, v in static.markers.items()}
    pts = rbds_static_points(mean, side, hip_method)
    p = "R." if side.lower().startswith("r") else "L."
    sign = 1.0 if p == "R." else -1.0
    ml_ankle = sign * (mean[p + "Ankle.Medial"] - mean[p + "Ankle"])
    ml_knee = sign * (mean[p + "Knee.Medial"] - mean[p + "Knee"])
    ml_pelvis = mean["L.ASIS"] - mean["R.ASIS"]
    r_static = np.stack([
        orthonormal_frame(pts["toe"] - pts["heel"], ml_ankle),
        orthonormal_frame(pts["knee"] - pts["ankle"], ml_ankle),
        orthonormal_frame(pts["hip"] - pts["knee"], ml_knee),
        orthonormal_frame(pts["l5s1"] - pts["mid_hip"], ml_pelvis),
    ])
    lengths = {
        "foot": float(np.linalg.norm(pts["toe"] - pts["heel"])),
        "shank": float(np.linalg.norm(pts["knee"] - pts["ankle"])),
        "thigh": float(np.linalg.norm(pts["hip"] - pts["knee"])),
        "pelvis": float(np.linalg.norm(pts["l5s1"] - pts["mid_hip"])),
        "forefoot_width": float(np.linalg.norm(mean[p + "MT1"]
                                               - mean[p + "MT5"])),
    }
    clusters = rbds_clusters(side)
    neutral = {seg: np.stack([mean[n] for n in clusters[seg]])
               for seg in SEGMENTS}
    return StaticCalibration(
        path=static.path, side=side.lower()[0], sex=sex, body_mass=body_mass,
        cluster_neutral=neutral, points=pts, r_static=r_static,
        lengths=lengths,
        foot_heel_local=r_static[0].T @ (pts["heel"] - pts["ankle"]),
        cluster_markers=clusters, point_carriers=CLUSTER_POINTS,
        foot_outline_local=np.stack([
            r_static[0].T @ (mean[p + n] - pts["ankle"])
            for n in ("Heel.Bottom", "MT1", "MT5", "Ankle.Medial", "Ankle")]))


def rbds_foot_centroid(trial: C3dTrial, side: str) -> np.ndarray:
    """Mean of whichever foot markers the trial carries (a few subjects'
    running trials drop an MT marker; the heel triad is always present, and
    the attribution that uses this is relative between the two feet)."""
    p = "R." if side.lower().startswith("r") else "L."
    names = (p + "Heel.Bottom", p + "Heel.Lateral", p + "Heel.Top",
             p + "MT1", p + "MT5")
    present = [trial.markers[n] for n in names if n in trial.markers]
    return np.mean(present, axis=0)


def rbds_foot_wrench(trial: C3dTrial, side: str,
                     point: np.ndarray | None = None,
                     threshold_n: float = 50.0,
                     antialias_hz: float = 50.0,
                     min_gap_s: float = 0.015,
                     min_peak_n: float = 250.0):
    """One foot's share of the single-belt wrench, by NEARER-FOOT COP.

    Like `instrumented_foot_wrench`, but with both feet marked the
    attribution is relative: a bout belongs to `side` when the plate COP's
    mean horizontal distance to that side's foot markers is smaller than to
    the other side's.  Returns ``(GroundWrench, bouts)``.
    """
    if point is None:
        point = np.zeros(3)
    point = np.asarray(point, dtype=float)
    other = "l" if side.lower().startswith("r") else "r"
    cents = {}
    for s in (side, other):
        c = rbds_foot_centroid(trial, s)
        cents[s] = np.stack([np.interp(trial.t_analog, trial.t, c[:, k])
                             for k in range(2)], axis=1)

    n_a, n_p = trial.force.shape[:2]
    f_out = np.zeros((n_a, 3))
    m_out = np.zeros((n_a, 3))
    bouts = []
    min_gap = int(round(min_gap_s * trial.analog_rate))
    for pl in range(n_p):
        f = trial.force[:, pl]
        cop = trial.cop[:, pl]
        free = trial.free_moment[:, pl]
        _, events = detect_contact(f[:, 2], threshold_n, min_gap=min_gap)
        for s0, e0 in events:
            sl = slice(s0, e0 + 1)
            peak = f[sl, 2].max()
            if peak < min_peak_n:
                continue
            ok = np.isfinite(cop[sl, 0]) & (f[sl, 2] > 0.5 * peak)
            d = {s: float(np.mean(np.linalg.norm(
                     cop[sl][ok][:, :2] - cents[s][sl][ok], axis=1)))
                 for s in (side, other)}
            mine = d[side] < d[other]
            bouts.append({"plate": pl, "t_on": s0 / trial.analog_rate,
                          "t_off": e0 / trial.analog_rate, "mine": mine,
                          "cop_dist_m": d[side],
                          "cop_dist_other_m": d[other]})
            if mine:
                c = np.where(np.isfinite(cop[sl]), cop[sl], 0.0)
                fr = np.where(np.isfinite(free[sl]), free[sl], 0.0)
                f_out[sl] += f[sl]
                m_out[sl] += np.cross(c - point, f[sl]) + fr
    ground = GroundWrench(t=trial.t.copy(), rate=trial.rate,
                          force=analog_to_mocap(trial, f_out, antialias_hz),
                          moment=analog_to_mocap(trial, m_out, antialias_hz),
                          point=point)
    return ground, sorted(bouts, key=lambda b: b["t_on"])


#: Dempster (1955) leg-segment parameters as tabulated by Winter
#: (Biomechanics and Motor Control of Human Movement, Table 4.1):
#: (mass fraction of body mass, COM fraction of segment length from the
#: PROXIMAL end, transverse radius of gyration / segment length about the
#: COM).  These are the mass fractions Visual3D's default hybrid segments
#: use (its inertias then come from uniform-density geometric solids, which
#: this table's radii stand in for).  Winter's foot segment runs lateral
#: malleolus -> MT2 head, i.e. our ankle -> metatarsal centre line.
DEMPSTER_LEG = {
    "foot": (0.0145, 0.50, 0.475),
    "shank": (0.0465, 0.433, 0.302),
    "thigh": (0.100, 0.433, 0.323),
}


def dempster_leg_skeleton(cal: StaticCalibration) -> Skeleton:
    """A [foot, shank, thigh, pelvis] Skeleton with Dempster/Winter leg
    parameters (`DEMPSTER_LEG`) in place of de Leva — the "their segment
    model" ablation arm.  The pelvis keeps its de Leva row: it sits ABOVE
    the hip in the bottom-up recursion, so it cannot affect the scored
    ankle/knee/hip kinetics.  Axial radius of gyration is set to half the
    transverse one (Winter tabulates only the transverse; the axial term is
    irrelevant to sagittal dynamics).  The foot COM lies halfway along the
    ankle->metatarsal line per Winter's segment definition."""
    base = deleva_skeleton(
        body_mass=cal.body_mass, foot_length=cal.lengths["foot"],
        shank_length=cal.lengths["shank"], thigh_length=cal.lengths["thigh"],
        pelvis_length=cal.lengths["pelvis"],
        foot_heel_local=cal.foot_heel_local, sex=cal.sex)
    toe_local = cal.r_static[0].T @ (cal.points["toe"] - cal.points["ankle"])
    for i, seg in enumerate(("foot", "shank", "thigh")):
        m_frac, c_frac, rg_t = DEMPSTER_LEG[seg]
        length = float(base.length[i])
        base.mass[i] = m_frac * cal.body_mass
        if seg == "foot":
            base.com_local[i] = c_frac * toe_local
        else:
            base.com_local[i] = np.array([0.0, 0.0, -c_frac * length])
        rg = np.array([rg_t, rg_t, 0.5 * rg_t]) * length
        base.inertia_local[i] = np.diag(base.mass[i] * rg ** 2)
    return base


def build_rbds_chain(trial: C3dTrial, cal: StaticCalibration,
                     point: np.ndarray | None = None,
                     threshold_n: float = 50.0,
                     segments: str = "deleva"):
    """(Skeleton, SegmentKinematics, GroundWrench, quality) for one RBDS leg
    — the bilateral analogue of `build_chain`, same contract.  `segments`
    picks the inertial table: 'deleva' (ours) or 'dempster'
    (`dempster_leg_skeleton`, approximating the dataset's V3D defaults)."""
    poses = cluster_poses(trial, cal)
    pts = virtual_points(poses, cal)
    r_world = np.stack([np.einsum("tij,jk->tik", poses[seg][0],
                                  cal.r_static[i])
                        for i, seg in enumerate(SEGMENTS)], axis=1)
    prox = np.stack([pts["ankle"], pts["knee"], pts["hip"], pts["l5s1"]],
                    axis=1)
    dist = np.stack([pts["toe"], pts["ankle"], pts["knee"], pts["mid_hip"]],
                    axis=1)
    if segments == "dempster":
        skeleton = dempster_leg_skeleton(cal)
    else:
        skeleton = deleva_skeleton(
            body_mass=cal.body_mass, foot_length=cal.lengths["foot"],
            shank_length=cal.lengths["shank"],
            thigh_length=cal.lengths["thigh"],
            pelvis_length=cal.lengths["pelvis"],
            foot_heel_local=cal.foot_heel_local, sex=cal.sex)
    kin = SegmentKinematics(t=trial.t.copy(), rate=trial.rate,
                            r_world=r_world, prox_pos=prox, dist_pos=dist)
    ground, bouts = rbds_foot_wrench(trial, cal.side, point=point,
                                     threshold_n=threshold_n)
    quality = {
        "cluster_residual": np.stack([poses[seg][2] for seg in SEGMENTS],
                                     axis=1),
        "bouts": bouts, "points": pts,
    }
    return skeleton, kin, ground, quality
