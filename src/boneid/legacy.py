"""boneid.legacy — a faithful Python port of the legacy MATLAB pipeline.

`matlab-inverse-dynamics/Asics_Moments4JCF_v3.m` and every routine it calls
are now present in the repository (the last three — `VirtualPoint_v2.m`,
`getViconForces_Motek.m`, `ContactEvents_Motek.m` — were recovered on
2026-09-11).  This module re-implements that script's INVERSE-DYNAMICS PATH
line by line, so that the legacy numbers can be produced from the same c3d
files the corrected pipeline reads and the two compared on equal terms.

Faithful means faithful: the legacy conventions and mistakes are reproduced
on purpose, each marked ``# LEGACY:`` at the line where it happens —

* the SIMM frame (x anterior, y up, z subject-right) via `NewCS`;
* Vaughan segment masses/COMs and Vaughan/Zatsiorsky inertias, including
  the shank tensor that carries the FOOT's transverse moment (`Ify`) and the
  thigh `Ity` regression fed the foot length `A(6)`;
* the rearfoot/forefoot split at the MTP axis by which side the COP falls on;
* one-sided "global" inertia  I_G = TM · I  (`multiprod(R_FootTM, Ifoot)`);
* `inv3d`'s moment balance  I·α  without the gyroscopic term;
* `RoomSegAng`'s angular velocity from  TMdot · TM  (no transpose) —
  −θ̇ cos 2θ for a planar rotation θ from the static pose;
* the vertical free moment stored in the z (medio-lateral) slot of `Md`
  (`getViconForces_Motek.m`, ``Md=[zeros zeros FM]``);
* NO low-pass filtering of markers or poses before differentiation; the joint
  moments are filtered at 12 Hz per stance AFTER the recursion;
* contact events from the 5 Hz-smoothed vertical force, right/left told
  apart by the alternating medio-lateral COP, boundaries tightened to 20 N.

`physics="corrected"` swaps ONLY the three equations (similarity-transformed
inertia, true angular velocity/acceleration, gyroscopic term kept) and
`free_moment_axis="y"` puts the free moment on the vertical axis; everything
else — anthropometrics, foot model, filtering, events — stays legacy, so the
gap between the legacy port and `boneid.core` can be attributed piece by piece.

The port is checked against the routines EXECUTED in MATLAB R2024b via
`harness/run_legacy_trial.m` (`run_matlab_legacy`; committed reference in
`data/asics/matlab_reference/`): events identical, every moment and every
exported per-stance curve within ~1e-11 N m.  It is also checked, where the
data allows, against ezc3d's own quantities (with the 18 Hz force
filter switched off, the ported COP algebra reproduces ezc3d's centre of
pressure to well under a millimetre; with it on, as the MATLAB ran, the two
differ by about a centimetre on a moving foot — the filter, not the algebra).
Filtering uses SciPy's `filtfilt` configured with MATLAB's padding rule, so
the port and MATLAB R2024b agree to round-off (see `run_matlab_legacy` and
`tests/test_legacy.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------------------
# Constants from the head of Asics_Moments4JCF_v3.m
# ---------------------------------------------------------------------------

RM_MOCAP = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
RM_FP = RM_MOCAP.copy()
G_SIMM = np.array([0.0, -9.81, 0.0])
AZ0 = -0.0000001                       # getViconForces_Motek's az0 (the main
                                       # script's -0.000001 is never passed in)
OFFSET_FP1 = np.array([0.872614, 1.04162, -0.0477233])   # lab frame, m
OFFSET_FP2 = np.array([0.371453, 1.040980, -0.0464284])
SHOE_HEIGHT = {"AM": 0.03, "AG": 0.02, "AH": 0.04}         # shcond order
EVENT_FORCE_THRESHOLD_N = 20.0
DEFAULT_TRIALS = 18

#: A(1..15) defaults used when "Subject Anthropometrics.xlsx" is absent
#: (it is not in this repository).  1-based names kept for traceability.
DEFAULT_ANTHRO = {1: 53.5, 2: 0.48, 3: 0.46, 4: 0.43, 5: 0.33, 6: 0.25,
                  7: 0.1, 8: 0.08, 9: 0.11, 10: 0.75, 11: 0.3, 12: 0.8,
                  13: 0.3, 14: 0.18, 15: 1.8}

#: The hardcoded bad-step table, subject 1 only (the rest of the table is in
#: the script; this repo only holds S1's data).  Keys: (shoe, speed).
BAD_STEPS_S1 = {("AM", "RP"): [8], ("AM", "PI"): [1] + list(range(7, 19)),
                ("AG", "RP"): [5, 6, 8], ("AG", "PI"): [2, 3, 4, 5, 8, 16],
                ("AH", "RP"): [5, 8], ("AH", "PI"): [5]}


# ---------------------------------------------------------------------------
# Subroutine ports (one function per .m file)
# ---------------------------------------------------------------------------

def new_cs(rm: np.ndarray, data: np.ndarray, origin=None) -> np.ndarray:
    """`NewCS(origin, RM, olddata)`: row vectors [.., 3] mapped by
    ``[x y z 1] * inv([RM origin'; 0 0 0 1])`` — i.e. new = RM @ (old - origin)
    for an orthonormal RM."""
    origin = np.zeros(3) if origin is None else np.asarray(origin, float)
    return (np.asarray(data, float) - origin) @ rm.T


def simm_from_lab(x: np.ndarray) -> np.ndarray:
    """Lab (+x left, +y posterior, +z up) -> SIMM (x anterior, y up, z right)."""
    return new_cs(RM_MOCAP, x)


def lab_from_simm(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, float) @ RM_MOCAP


def _filtfilt_matlab(b, a, x, axis=0):
    """SciPy `filtfilt` configured like MATLAB's: odd extension of length
    3*(max(len(a),len(b))-1) with steady-state initial conditions (SciPy's
    default pad length is 3*max(len(a),len(b)), which changes the first and
    last samples of short stance windows by several N m)."""
    return filtfilt(b, a, x, axis=axis, padtype="odd",
                    padlen=3 * (max(len(a), len(b)) - 1))


def filterdata(x: np.ndarray, sf: float, lp: float, order: int = 4) -> np.ndarray:
    """`filterdata(data, sf, 0, lp, order, 0)`: butter(order/2) + filtfilt,
    whole columns, no zero-skipping."""
    b, a = butter(order // 2, 2.0 * lp / sf)
    return _filtfilt_matlab(b, a, np.asarray(x, float))


def filterdata1(x: np.ndarray, sf: float, lp: float, order: int = 4) -> np.ndarray:
    """`filterdata1`: like `filterdata`, but each column is filtered only
    between its first and last NON-ZERO sample; everything outside that
    window — and any column too short (< 3*order) — is zero."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return filterdata1(x[:, None], sf, lp, order)[:, 0]
    b, a = butter(order // 2, 2.0 * lp / sf)
    out = np.zeros_like(x)
    rows = x.shape[0]
    for p in range(x.shape[1]):
        start, end = 0, rows - 1
        while x[start, p] == 0 and start < rows - 1:
            start += 1
        while x[end, p] == 0 and end > 0:
            end -= 1
        if end - start > 3 * order:
            out[start:end + 1, p] = _filtfilt_matlab(b, a, x[start:end + 1, p])
    return out


def first_central(a: np.ndarray, sf: float) -> np.ndarray:
    """`FirstCentral`: central difference, one-sided at both ends."""
    a = np.asarray(a, float)
    d = np.empty_like(a)
    d[1:-1] = (a[2:] - a[:-2]) * sf / 2.0
    d[0] = (a[1] - a[0]) * sf
    d[-1] = (a[-1] - a[-2]) * sf
    return d


def soder(neut: np.ndarray, mov: np.ndarray):
    """`SODER([Neut; Mov])` vectorised over frames.

    neut: [M,3] neutral marker positions; mov: [T,M,3].  Returns
    (R [T,3,3], d [T,3], res [T]) with mov ≈ R neut + d and the SODER
    residual sqrt(sum(diff²) / (3M − 6)).  Frames containing NaN return NaN.
    """
    neut = np.asarray(neut, float)
    mov = np.asarray(mov, float)
    a0 = neut - neut.mean(axis=0)
    bmean = mov.mean(axis=1, keepdims=True)
    b0 = mov - bmean
    c = np.einsum("tmi,mj->tij", b0, a0)
    ok = np.isfinite(c).all(axis=(1, 2))
    r = np.full((len(mov), 3, 3), np.nan)
    d = np.full((len(mov), 3), np.nan)
    res = np.full(len(mov), np.nan)
    if ok.any():
        p, _, qt = np.linalg.svd(c[ok])
        det = np.linalg.det(np.einsum("tij,tjk->tik", p, qt))
        fix = np.repeat(np.eye(3)[None], int(ok.sum()), axis=0)
        fix[:, 2, 2] = det
        r_ok = np.einsum("tij,tjk,tkl->til", p, fix, qt)
        d_ok = bmean[ok, 0] - np.einsum("tij,j->ti", r_ok, neut.mean(axis=0))
        pred = np.einsum("tij,mj->tmi", r_ok, neut) + d_ok[:, None]
        dof = 3 * len(neut) - 6
        r[ok], d[ok] = r_ok, d_ok
        res[ok] = np.sqrt(((mov[ok] - pred) ** 2).sum(axis=(1, 2)) / dof)
    return r, d, res


def virtual_point_v2(neut: np.ndarray, pneut: np.ndarray, mov: np.ndarray):
    """`VirtualPoint_v2(Neut, pneut, Mov)`: (NewPoint [T,3], RM [T,3,3], Tres [T]).

    NewPoint = R pneut + d; RM is the TRANSPOSE of the SODER rotation — the
    "TM" every downstream legacy routine consumes.
    """
    r, d, res = soder(neut, mov)
    new_point = np.einsum("tij,j->ti", r, np.asarray(pneut, float)) + d
    return new_point, np.swapaxes(r, 1, 2), res


def room_seg_ang(tm: np.ndarray, sf: float, thresh: float = 200.0):
    """`RoomSegAng(TM, sf)` -> (AV [T,3], AA [T,3]).

    LEGACY: S = TMdot @ TM (no transpose), components S[2,1], S[0,2], S[1,0]
    with the alternate skew entry substituted for a whole component when its
    peak exceeds `thresh`; AA = FirstCentral(AV).

    With TM = Rᵀ this is (d/dt Rᵀ) Rᵀ.  For a rotation θ(t) about one fixed
    axis, measured from the static pose, the extracted component is
    −θ̇ cos 2θ instead of θ̇ — right size but wrong sign at θ = 0, zero at 45°, right sign
    (wrong size) beyond.
    Unlike the inertia and gyroscopic mistakes, planar motion does not hide
    this one; what hides it is that segments stay within a few tens of
    degrees of their standing pose and the I·α term is small next to the
    force lever arms.
    """
    tm = np.asarray(tm, float)
    tmdot = first_central(tm.reshape(len(tm), 9), sf).reshape(len(tm), 3, 3)
    s = np.einsum("tij,tjk->tik", tmdot, tm)
    av = np.stack([s[:, 2, 1], s[:, 0, 2], s[:, 1, 0]], axis=1)
    alt = np.stack([-s[:, 1, 2], -s[:, 2, 0], -s[:, 0, 1]], axis=1)
    for c in range(3):
        if np.nanmax(np.abs(av[:, c])) > thresh:
            av[:, c] = alt[:, c]
    aa = first_central(av, sf) if len(tm) > 3 else np.zeros((1, 3))
    return av, aa


def true_seg_ang(tm: np.ndarray, sf: float):
    """The corrected counterpart of `room_seg_ang`: world angular velocity
    from Ṙ Rᵀ with R = TMᵀ, and α = dω/dt."""
    r = np.swapaxes(np.asarray(tm, float), 1, 2)
    rdot = first_central(r.reshape(len(r), 9), sf).reshape(len(r), 3, 3)
    w_hat = np.einsum("tij,tkj->tik", rdot, r)
    av = np.stack([0.5 * (w_hat[:, 2, 1] - w_hat[:, 1, 2]),
                   0.5 * (w_hat[:, 0, 2] - w_hat[:, 2, 0]),
                   0.5 * (w_hat[:, 1, 0] - w_hat[:, 0, 1])], axis=1)
    return av, first_central(av, sf)


def inv3d(cm, jp, jd, rd, md, a_cm, a_ang, m, i_g, g=G_SIMM, omega=None):
    """`inv3d(CM, Jp, Jd, Rd, Md, aCM, aANG, m, I, g)` -> (Rp, Mp).

    Rp = m aCM − Rd − m g;  Mp = I·aANG − Md − (Jd−CM)×Rd − (Jp−CM)×Rp.
    LEGACY: no ω×(Iω).  Pass `omega` to add that term (the corrected physics).
    """
    rp = m * a_cm - rd - m * g
    rot = np.einsum("tij,tj->ti", i_g, a_ang)
    if omega is not None:
        rot = rot + np.cross(omega, np.einsum("tij,tj->ti", i_g, omega))
    mp = rot - md - np.cross(jd - cm, rd) - np.cross(jp - cm, rp)
    return rp, mp


def g2l(tm: np.ndarray, glob: np.ndarray) -> np.ndarray:
    """`G2L(LCS, Global)` as actually evaluated on a T×3×3 TM: MATLAB's
    `LCS(i,1:3)` folds the trailing dims, so the assembled matrix is TMᵀ and
    `inv(TMᵀ) * Global = TM · Global` — a proper global→local transform for
    this TM convention (TM = Rᵀ)."""
    return np.einsum("tij,tj->ti", tm, glob)


def rtocarda_zxy(r: np.ndarray) -> np.ndarray:
    """`RTOCARDA(R, Z, X, Y)` first solution, radians, stored in (x, y, z)
    slots as `CardanSegAngles` returns them.  i=Z, j=X, k=Y, sig=+1."""
    r = np.asarray(r, float)
    a = np.empty(r.shape[:-2] + (3,))
    a[..., 2] = np.arctan2(-r[..., 0, 1], r[..., 1, 1])          # a(i)=a(z)
    a[..., 0] = np.arcsin(np.clip(r[..., 2, 1], -1, 1))         # a(j)=a(x)
    a[..., 1] = np.arctan2(-r[..., 2, 0], r[..., 2, 2])          # a(k)=a(y)
    return a


def cardan_seg_angles(tm: np.ndarray) -> np.ndarray:
    """`CardanSegAngles(TM, 'zxy')` in degrees: RTOCARDA(inv(TM)) × 57.3."""
    return rtocarda_zxy(np.swapaxes(tm, 1, 2)) * 57.3


def cardan_joint_angles(tm_prox: np.ndarray, tm_dist: np.ndarray) -> np.ndarray:
    """`CardanJointAngles(prox, dist, 'zxy')`: RTOCARDA(prox · inv(dist)) × 57.3."""
    rel = np.einsum("tij,tkj->tik", tm_prox, tm_dist)     # prox @ dist^T
    return rtocarda_zxy(rel) * 57.3


def get_vicon_forces_motek(analog: np.ndarray, sf: float, offsets_lab,
                           free_moment_axis: str = "z", lowpass_hz: float = 18.0):
    """`getViconForces_Motek(AnalogSignals, sf, RMfp, origin, FP_offsets, 0)`.

    analog: [T,12] = [F1 M1 F2 M2] with moments in N mm (ezc3d platform
    force/moment, about the plate centre).  Returns (Rd, Md, COP, plate) in
    SIMM, plate 1-based as MATLAB reports it.  LEGACY: the free moment FM
    (about the vertical axis) is returned in Md's z slot; `free_moment_axis=
    "y"` puts it where it belongs.
    """
    analog = np.asarray(analog, float)
    # `lowpass_hz` is the routine's hard-coded 18 Hz; 0 disables it (to check
    # the COP algebra against ezc3d).
    lp = (lambda x: filterdata(x, sf, lowpass_hz)) if lowpass_hz > 0 else (lambda x: x)
    rd = {}
    md = {}
    for k in (1, 2):
        c = 6 * (k - 1)
        rd[k] = new_cs(RM_FP, lp(analog[:, c:c + 3]))
        md[k] = new_cs(RM_FP, lp(analog[:, c + 3:c + 6] / 1000.0))
    plate = 1 if rd[1][:, 1].mean() > rd[2][:, 1].mean() else 2
    rd_p, md_p = rd[plate], md[plate]
    offset = new_cs(RM_FP, np.asarray(offsets_lab, float)[plate - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cop_ml_local = (-md_p[:, 0] - rd_p[:, 2] * AZ0) / rd_p[:, 1]
        cop_ap_local = (md_p[:, 2] - rd_p[:, 0] * AZ0) / rd_p[:, 1]
    cop = np.stack([cop_ap_local + offset[0], np.zeros(len(rd_p)),
                    cop_ml_local + offset[2]], axis=1)
    fm = md_p[:, 1] - rd_p[:, 0] * cop_ml_local + rd_p[:, 2] * cop_ap_local
    md_out = np.zeros_like(md_p)
    md_out[:, 2 if free_moment_axis == "z" else 1] = fm       # LEGACY: z slot
    return rd_p, md_out, cop, plate


def contact_events_motek(f_vert: np.ndarray, cop_ml: np.ndarray, sf: float):
    """`ContactEvents_Motek(Fvert, COPml, sf)` -> (HS, TO, ind_off), 0-based.

    Peaks of the 5 Hz-smoothed force above 100 N (first and last dropped);
    HS/TO by walking out from each peak on the 20 Hz-smoothed force to 5 N;
    right/left separated by the alternating medio-lateral COP at the peaks
    (SIMM z: the right foot has the larger value).  LEGACY: MATLAB's
    `HS(1:2:end-1)=[]` leaves the final element in place when the count is
    odd, reproduced here.
    """
    f_vert = np.asarray(f_vert, float)
    smooth_h = filterdata(f_vert, sf, 5.0)
    locs, _ = find_peaks(smooth_h, height=100.0)
    locs = locs[1:-1]
    smooth_l = filterdata(f_vert, sf, 20.0)
    hs = np.empty(len(locs), int)
    to = np.empty(len(locs), int)
    n = len(smooth_l)
    for i, k0 in enumerate(locs):
        k = k0
        while k > 0 and smooth_l[k] > 5:
            k -= 1
        hs[i] = k
        k = k0
        while k < n - 1 and smooth_l[k] > 5:
            k += 1
        to[i] = k
    cop_loc = np.asarray(cop_ml, float)[locs]
    m1 = np.nanmean(cop_loc[0:len(cop_loc) - 1:2])          # MATLAB 1:2:end-1
    m2 = np.nanmean(cop_loc[1::2])                          # MATLAB 2:2:end
    if m1 < m2:                                             # m2 is the right foot
        drop = np.arange(0, len(hs) - 1, 2)                 # 1:2:end-1 (1-based)
    else:
        drop = np.arange(1, len(hs), 2)                     # 2:2:end
    keep = np.setdiff1d(np.arange(len(hs)), drop)
    hs, to = hs[keep], to[keep]
    ind = _off_ground(hs, to, len(f_vert))
    return hs, to, ind


def _off_ground(hs, to, n):
    parts = [np.arange(0, hs[0] + 1), np.arange(to[-1], n)]
    for i in range(len(to) - 1):
        parts.append(np.arange(to[i], hs[i + 1] + 1))
    return np.unique(np.concatenate(parts))


def tighten_events(hs, to, f_vert, threshold=EVENT_FORCE_THRESHOLD_N):
    """The main script's post-processing: pull each HS/TO in to the first and
    last sample above `threshold`, and rebuild the off-ground index set."""
    hs, to = hs.copy(), to.copy()
    for ev in range(len(hs)):
        idx = np.arange(hs[ev], to[ev] + 1)
        above = np.flatnonzero(f_vert[idx] > threshold)
        if above.size:
            hs[ev], to[ev] = idx[above[0]], idx[above[-1]]
    ind_off = _off_ground(hs, to, len(f_vert)) if len(hs) else np.arange(len(f_vert))
    return hs, to, ind_off


def pchip_resample(percent_old, y, percent_new):
    """`interp1(percent_old, y, percent_new, 'pchip')` along axis 0."""
    return PchipInterpolator(percent_old, np.asarray(y, float), axis=0)(percent_new)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

@dataclass
class LegacyResult:
    """Everything the legacy inverse-dynamics path produces, SIMM frame
    (x anterior, y up, z subject-right) unless a field says otherwise."""

    rate: float
    t: np.ndarray                               # [T] s
    anthro: dict                                # A(1..15) as used
    mass: dict                                  # segment masses (kg)
    inertia_local: dict                         # segment 3x3 tensors
    plate: int                                  # 1-based, as MATLAB reports
    rd: np.ndarray                              # [T,3] ground force (zeroed off-ground)
    md: np.ndarray                              # [T,3] "Md" as inv3d received it
    cop: np.ndarray                             # [T,3]
    free_moment: np.ndarray                     # [T] FM before it was slotted
    hs: np.ndarray                              # [n] 0-based heel strikes (right)
    to: np.ndarray                              # [n] 0-based toe offs (right)
    joints: dict                                # name -> [T,3] joint centre
    tm: dict                                    # segment -> [T,3,3] (= Rᵀ)
    cluster_res: dict                           # segment -> [T] SODER residual
    global_force: dict                          # joint -> [T,3]
    global_moment: dict                         # joint -> [T,3]
    local_force: dict                           # joint -> [T,3]
    local_moment: dict                          # joint -> [T,3]
    joint_angle_deg: dict                       # joint -> [T,3] Cardan zxy
    joint_power: dict                           # joint -> [T] (12 Hz filtered)
    cop_mtp_dist: np.ndarray                    # [T] which foot model was used
    steps: list = field(default_factory=list)   # per-stance dicts (see run)
    physics: str = "legacy"
    notes: list = field(default_factory=list)


def _markers_simm(trial, mean: bool):
    """Marker dict in SIMM metres; the static trial is averaged over frames
    (`nanmean`), the dynamic one kept per frame — exactly the script."""
    out = {}
    for name, x in trial.markers.items():
        out[name] = simm_from_lab(np.nanmean(x, axis=0) if mean else x)
    return out


def legacy_anthropometrics(ms: dict, shoe: str, overrides: dict | None = None):
    """A(1..15): defaults, the caller's overrides (e.g. a measured mass), and
    the four entries the script measures from the static markers."""
    a = dict(DEFAULT_ANTHRO)
    a.update(overrides or {})
    hip = 0.70 * ms["RGT"] + 0.30 * ms["LGT"]
    knee = 0.5 * (ms["KneeLat"] + ms["KneeMed"])
    ankle = 0.5 * (ms["AnkleLat"] + ms["AnkleMed"])
    a[2] = float(np.linalg.norm(hip - knee))
    a[4] = float(np.linalg.norm(knee - ankle))
    a[7] = float(ankle[1] - SHOE_HEIGHT[shoe] - 0.0045)
    a[8] = float(ms["AnkleLat"][2] - ms["AnkleMed"][2] - 0.009)
    return a


def raw_analog(trial) -> np.ndarray:
    """[Ta,12] = [F1 M1 F2 M2] as `ezc3dRead`'s platform output gives it:
    force in N, moment in N mm about the plate centre, lab orientation.
    Read straight from the c3d so nothing of the corrected loader leaks in."""
    import ezc3d
    c = ezc3d.c3d(trial.path, extract_forceplat_data=True)
    cols = []
    for p in c["data"]["platform"][:2]:
        cols.append(np.asarray(p["force"], float).T)
        cols.append(np.asarray(p["moment"], float).T)
    return np.concatenate(cols, axis=1)


def run_legacy_pipeline(static, trial, shoe: str, speed: str = "RP",
                        anthro_overrides: dict | None = None,
                        physics: str = "legacy", free_moment_axis: str = "z",
                        marker_lowpass_hz: float = 0.0,
                        n_steps: int = DEFAULT_TRIALS) -> LegacyResult:
    """The inverse-dynamics path of `Asics_Moments4JCF_v3.m` on one
    (static, dynamic) pair of `io_c3d.C3dTrial`s.

    NOTE: `io_c3d.load_c3d` repairs thigh label swaps and despikes markers;
    the MATLAB did neither.  Pass trials loaded with ``repair=False`` for a
    strictly raw run (on S1_AM_RP the swaps are brief and the difference is
    confined to a few frames).

    `marker_lowpass_hz` > 0 low-passes the dynamic markers before anything
    is fitted or differentiated — NOT something the MATLAB did (it
    differentiated raw marker-derived positions twice at 300 Hz); it is the
    rung of the comparison ladder that isolates that choice.
    """
    if physics not in ("legacy", "corrected"):
        raise ValueError("physics must be 'legacy' or 'corrected'")
    notes = []
    sf = float(trial.rate)
    t = trial.t.copy()
    n_t = len(t)

    # ---- static: markers, joint centres, anthropometrics --------------------
    ms = _markers_simm(static, mean=True)
    A = legacy_anthropometrics(ms, shoe, anthro_overrides)
    n_l5s1 = 0.75 * (0.5 * ms["LPSIS"] + 0.5 * ms["RPSIS"]) + 0.25 * (0.5 * ms["LASIS"] + 0.5 * ms["RASIS"])
    nr_hip = 0.70 * ms["RGT"] + 0.30 * ms["LGT"]
    nr_knee = 0.5 * (ms["KneeLat"] + ms["KneeMed"])
    nr_ankle = 0.5 * (ms["AnkleLat"] + ms["AnkleMed"])
    nr_met = 0.5 * (ms["MTP1Head"] + ms["MTP5Head"])
    clusters_static = {
        "pelvis": np.stack([ms[n] for n in ("RASIS", "LASIS", "RPSIS", "LPSIS", "RGT", "LGT")]),
        "thigh": np.stack([ms[n] for n in ("ThighAntProx", "ThighPostDist", "ThighPostProx")]),
        "leg": np.stack([ms[n] for n in ("ShankAntProx", "ShankPostDist", "ShankPostProx")]),
        "foot": np.stack([ms[n] for n in ("HeelLat", "HeelMed", "DorsalFoot")]),
        "forefoot": np.stack([ms[n] for n in ("MTP1Head", "MTP5Head", "ToeboxAnt")]),
    }
    clusters_static["rearfoot"] = clusters_static["foot"]

    # Vaughan masses (legacy coefficients as typed)
    m_foot = 0.0083 * A[1] + 254.5 * A[6] * A[7] * A[8] - 0.065
    m_leg = 0.0223 * A[1] + 31.33 * A[4] * A[5] ** 2 - 0.016
    m_thigh = 0.1032 * A[1] + 12.76 * A[2] * A[3] ** 2 - 1.023
    foot_x = ms["ToeboxAnt"][0] - ms["HeelMed"][0]
    m_rear = (nr_met[0] - ms["HeelMed"][0]) / foot_x * m_foot
    m_fore = (ms["ToeboxAnt"][0] - nr_met[0]) / foot_x * m_foot

    half_w = (ms["MTP5Head"][2] - ms["MTP1Head"][2]) * 0.5
    i_rf = np.diag([(1 / 12) * m_rear * (3 * half_w ** 2 + (nr_met[0] - ms["HeelMed"][0]) ** 2),
                    (1 / 12) * m_rear * (3 * (nr_ankle[1] * 0.5) ** 2 + (nr_met[0] - ms["HeelMed"][0]) ** 2),
                    (1 / 4) * m_rear * (half_w ** 2 + 3 * (nr_ankle[1] * 0.5) ** 2)])
    i_ff = np.diag([(1 / 12) * m_fore * (3 * half_w ** 2 + (ms["ToeboxAnt"][0] - nr_met[0]) ** 2),
                    (1 / 12) * m_fore * (3 * (ms["ToeboxAnt"][1] * 0.5) ** 2 + (ms["ToeboxAnt"][0] - nr_met[0]) ** 2),
                    (1 / 4) * m_fore * (half_w ** 2 + (ms["ToeboxAnt"][1] * 0.5) ** 2)])
    # Vaughan inertias, as typed — including the two copy-paste slips
    ifx = 0.00021 * A[1] * (4 * A[9] ** 2 + 3 * A[6] ** 2) + 0.00067
    ifz = 0.00023 * A[1] * (4 * A[7] ** 2 + 3 * A[6] ** 2) + 0.00022
    ify = 0.00141 * A[1] * (A[7] ** 2 + A[9] ** 2) - 0.00008
    ilx = 0.00387 * A[1] * (A[4] ** 2 + 0.076 * A[5] ** 2) + 0.00138
    ilz = 0.00347 * A[1] * (A[4] ** 2 + 0.076 * A[5] ** 2) + 0.00511
    itx = 0.00726 * A[1] * (A[2] ** 2 + 0.076 * A[3] ** 2) + 0.01186
    itz = 0.00762 * A[1] * (A[2] ** 2 + 0.076 * A[3] ** 2) + 0.01153
    ity = 0.00151 * A[1] * A[6] ** 2 + 0.00305          # LEGACY: foot length A(6)
    i_local = {"foot": np.diag([ifx, ify, ifz]),
               "leg": np.diag([ilx, ify, ilz]),           # LEGACY: foot's Ify
               "thigh": np.diag([itx, ity, itz]),
               "rearfoot": i_rf, "forefoot": i_ff}
    r_cm_foot_frac = 0.44 + 0.0195 / A[6]

    # ---- dynamic: markers, forces, events --------------------------------
    md_dyn = _markers_simm(trial, mean=False)
    if marker_lowpass_hz > 0:
        md_dyn = {k: filterdata(v, sf, marker_lowpass_hz) for k, v in md_dyn.items()}
        notes.append(f"markers low-passed at {marker_lowpass_hz:g} Hz (NOT legacy behaviour)")
    analog = raw_analog(trial)
    atime = np.arange(len(analog)) / float(trial.analog_rate)
    analog_v = PchipInterpolator(atime, analog, axis=0, extrapolate=True)(t)
    rd, md, cop, plate = get_vicon_forces_motek(
        analog_v, sf, np.stack([OFFSET_FP1, OFFSET_FP2]), free_moment_axis)
    fm = md[:, 2 if free_moment_axis == "z" else 1].copy()
    hs, to, _ = contact_events_motek(rd[:, 1], cop[:, 2], sf)
    hs, to, ind_off = tighten_events(hs, to, rd[:, 1])
    rd[ind_off] = 0.0
    md[ind_off] = 0.0
    cop[ind_off] = np.stack([md_dyn["DorsalFoot"][ind_off, 0], np.zeros(len(ind_off)),
                             md_dyn["DorsalFoot"][ind_off, 2]], axis=1)

    # ---- virtual points and segment TMs ----------------------------------
    dyn = {"pelvis": np.stack([md_dyn[n] for n in ("RASIS", "LASIS", "RPSIS", "LPSIS", "RGT", "LGT")], axis=1),
           "thigh": np.stack([md_dyn[n] for n in ("ThighAntProx", "ThighPostDist", "ThighPostProx")], axis=1),
           "leg": np.stack([md_dyn[n] for n in ("ShankAntProx", "ShankPostDist", "ShankPostProx")], axis=1),
           "foot": np.stack([md_dyn[n] for n in ("HeelLat", "HeelMed", "DorsalFoot")], axis=1),
           "forefoot": np.stack([md_dyn[n] for n in ("MTP1Head", "MTP5Head", "ToeboxAnt")], axis=1)}
    dyn["rearfoot"] = dyn["foot"]
    j_mtp, tm_ff, res_ff = virtual_point_v2(clusters_static["forefoot"], nr_met, dyn["forefoot"])
    _, tm_rf, _ = virtual_point_v2(clusters_static["rearfoot"], nr_ankle, dyn["rearfoot"])
    j_ankle, tm_foot, res_foot = virtual_point_v2(clusters_static["foot"], nr_ankle, dyn["foot"])
    j_knee, tm_leg, res_leg = virtual_point_v2(clusters_static["leg"], nr_knee, dyn["leg"])
    j_hip, tm_thigh, res_thigh = virtual_point_v2(clusters_static["thigh"], nr_hip, dyn["thigh"])
    j_l5s1, tm_pelvis, res_pelvis = virtual_point_v2(clusters_static["pelvis"], n_l5s1, dyn["pelvis"])
    tm = {"foot": tm_foot, "leg": tm_leg, "thigh": tm_thigh, "pelvis": tm_pelvis,
          "rearfoot": tm_rf, "forefoot": tm_ff}
    cluster_res = {"foot": res_foot, "leg": res_leg, "thigh": res_thigh,
                   "pelvis": res_pelvis, "forefoot": res_ff}

    # ---- "global" inertia -------------------------------------------------
    def global_inertia(seg):
        if physics == "legacy":
            return np.einsum("tij,jk->tik", tm[seg], i_local[seg])   # LEGACY: TM·I
        r = np.swapaxes(tm[seg], 1, 2)
        return np.einsum("tij,jk,tlk->til", r, i_local[seg], r)      # R I Rᵀ
    i_g = {seg: global_inertia(seg) for seg in ("foot", "leg", "thigh", "rearfoot", "forefoot")}

    # ---- COMs, accelerations, angular kinematics --------------------------
    cm = {"rearfoot": md_dyn["HeelMed"] + 0.5 * (j_mtp - md_dyn["HeelMed"]),
          "forefoot": j_mtp + 0.5 * (md_dyn["ToeboxAnt"] - j_mtp),
          "foot": md_dyn["HeelLat"] - r_cm_foot_frac * (md_dyn["HeelLat"] - md_dyn["ToeboxAnt"]),
          "leg": j_knee - 0.42 * (j_knee - j_ankle),
          "thigh": j_hip - 0.39 * (j_hip - j_knee)}
    a_cm = {k: first_central(first_central(v, sf), sf) for k, v in cm.items()}  # LEGACY: unfiltered
    ang = {}
    for seg in ("foot", "leg", "thigh", "rearfoot", "forefoot"):
        ang[seg] = (room_seg_ang if physics == "legacy" else true_seg_ang)(tm[seg], sf)
    omega = (lambda seg: None) if physics == "legacy" else (lambda seg: ang[seg][0])

    # ---- foot split at the MTP axis, ankle / knee / hip --------------------
    cop_filt = filterdata1(cop, sf, 12.0)
    m5, m1 = md_dyn["MTP5Head"], md_dyn["MTP1Head"]
    cop_mtp_dist = (((m5[:, 0] - m1[:, 0]) * cop_filt[:, 2] - (m5[:, 2] - m1[:, 2]) * cop_filt[:, 0]
                     + m5[:, 2] * m1[:, 0] - m5[:, 0] * m1[:, 2])
                    / np.sqrt((m5[:, 0] - m1[:, 0]) ** 2 + (m5[:, 2] - m1[:, 2]) ** 2))
    whole = cop_mtp_dist > 0
    f_ankle_whole, m_ankle_whole = inv3d(cm["foot"], j_ankle, cop, rd, md, a_cm["foot"], ang["foot"][1],
                                         m_foot, i_g["foot"], omega=omega("foot"))
    f_mtp, m_mtp = inv3d(cm["forefoot"], j_mtp, cop, rd, md, a_cm["forefoot"], ang["forefoot"][1],
                         m_fore, i_g["forefoot"], omega=omega("forefoot"))
    f_ankle_split, m_ankle_split = inv3d(cm["rearfoot"], j_ankle, j_mtp, -f_mtp, -m_mtp, a_cm["rearfoot"],
                                         ang["rearfoot"][1], m_rear, i_g["rearfoot"], omega=omega("rearfoot"))
    f_ankle = np.where(whole[:, None], f_ankle_whole, f_ankle_split)
    m_ankle = np.where(whole[:, None], m_ankle_whole, m_ankle_split)
    f_mtp = np.where(whole[:, None], 0.0, f_mtp)
    m_mtp = np.where(whole[:, None], 0.0, m_mtp)
    f_knee, m_knee = inv3d(cm["leg"], j_knee, j_ankle, -f_ankle, -m_ankle, a_cm["leg"], ang["leg"][1],
                           m_leg, i_g["leg"], omega=omega("leg"))
    f_hip, m_hip = inv3d(cm["thigh"], j_hip, j_knee, -f_knee, -m_knee, a_cm["thigh"], ang["thigh"][1],
                         m_thigh, i_g["thigh"], omega=omega("thigh"))

    global_force = {"mtp": f_mtp, "ankle": f_ankle, "knee": f_knee, "hip": f_hip}
    global_moment = {"mtp": m_mtp, "ankle": m_ankle, "knee": m_knee, "hip": m_hip}
    prox_tm = {"mtp": tm["rearfoot"], "ankle": tm["leg"], "knee": tm["thigh"], "hip": tm["pelvis"]}
    local_moment = {j: g2l(prox_tm[j], global_moment[j]) for j in global_moment}
    local_force = {j: g2l(prox_tm[j], global_force[j]) for j in global_force}
    local_moment["mtp"] = np.nan_to_num(local_moment["mtp"])
    local_moment["hip"][np.isinf(local_moment["hip"])] = 0.0

    # ---- angles, velocities, powers ---------------------------------------
    joint_angle = {"mtp": cardan_joint_angles(tm["rearfoot"], tm["forefoot"]),
                   "ankle": cardan_joint_angles(tm["leg"], tm["foot"]),
                   "knee": cardan_joint_angles(tm["thigh"], tm["leg"]),
                   "hip": cardan_joint_angles(tm["pelvis"], tm["thigh"])}
    joint_power = {}
    for j, ang_deg in joint_angle.items():
        v_rad = filterdata1(first_central(np.deg2rad(ang_deg), sf), sf, 12.0)
        joint_power[j] = filterdata1(local_moment[j][:, 2] * v_rad[:, 2], sf, 12.0)

    # ---- per-stance products ---------------------------------------------
    bad = set(BAD_STEPS_S1.get((shoe, speed), []))
    steps = []
    pct_new = np.arange(101)
    for i in range(min(n_steps, len(hs) - 1)):
        sl = slice(hs[i], to[i] + 1)
        n_s = to[i] - hs[i] + 1
        pct_old = np.arange(n_s) / sf
        pct_old = pct_old / pct_old[-1] * 100.0
        step = {"index": i + 1, "bad": (i + 1) in bad,
                "hs": int(hs[i]), "to": int(to[i]),
                "contact_time_s": (to[i] - hs[i]) / sf,
                "flight_time_s": (hs[i + 1] - to[i] - n_s) / 2.0 / sf,
                "stride_time_s": (hs[i + 1] - hs[i]) / sf,
                "local_moment": {}, "global_moment": {}, "local_force": {},
                "joint_angle_deg": {}, "power": {}, "peak_local_moment": {}}
        for j in ("mtp", "ankle", "knee", "hip"):
            lm = filterdata(local_moment[j][sl], sf, 12.0)           # LEGACY: per stance, post hoc
            gm = filterdata(global_moment[j][sl], sf, 12.0)
            lf = filterdata(local_force[j][sl], sf, 12.0)
            step["local_moment"][j] = pchip_resample(pct_old, lm, pct_new)
            step["global_moment"][j] = pchip_resample(pct_old, gm, pct_new)
            step["local_force"][j] = pchip_resample(pct_old, lf, pct_new)
            step["joint_angle_deg"][j] = pchip_resample(pct_old, joint_angle[j][sl], pct_new)
            step["power"][j] = pchip_resample(pct_old, joint_power[j][sl], pct_new)
        step["peak_local_moment"] = {
            "mtp": step["local_moment"]["mtp"].min(axis=0),
            "ankle": step["local_moment"]["ankle"].min(axis=0),
            "knee": step["local_moment"]["knee"].max(axis=0),
            "hip": step["local_moment"]["hip"].min(axis=0)}
        step["grf"] = pchip_resample(pct_old, rd[sl], pct_new)
        steps.append(step)

    notes.append(f"plate {plate} (1-based) selected by mean vertical force")
    notes.append(f"{len(hs)} right-foot contacts after right/left separation; "
                 f"{len(steps)} analysed (first {n_steps}, one stride needed after each)")
    notes.append("free moment stored in Md's " + ("z (ML) slot — the legacy quirk" if free_moment_axis == "z"
                                                  else "y (vertical) slot — corrected"))
    return LegacyResult(
        rate=sf, t=t, anthro=A,
        mass={"foot": m_foot, "leg": m_leg, "thigh": m_thigh, "rearfoot": m_rear, "forefoot": m_fore},
        inertia_local=i_local, plate=plate, rd=rd, md=md, cop=cop, free_moment=fm,
        hs=hs, to=to,
        joints={"mtp": j_mtp, "ankle": j_ankle, "knee": j_knee, "hip": j_hip, "l5s1": j_l5s1},
        tm=tm, cluster_res=cluster_res,
        global_force=global_force, global_moment=global_moment,
        local_force=local_force, local_moment=local_moment,
        joint_angle_deg=joint_angle, joint_power=joint_power,
        cop_mtp_dist=cop_mtp_dist, steps=steps, physics=physics, notes=notes)


# ---------------------------------------------------------------------------
# Running the REAL MATLAB (R2024b is installed on the study machine)
# ---------------------------------------------------------------------------

MATLAB_BIN = "/Applications/MATLAB_R2024b.app/bin/matlab"
HARNESS_DIR = "matlab-inverse-dynamics/harness"
MATLAB_REFERENCE = "data/asics/matlab_reference"


def export_for_matlab(c3d_path, dst) -> str:
    """Write one c3d's contents to a .mat shaped like `ezc3dRead`'s outputs
    (no Mac build of the ezc3d MEX exists): `points` [3,M,T] in mm, `labels`
    (cellstr), `point_rate`, `analog_rate`, and per plate `force` [3,Ta] N and
    `moment` [3,Ta] N mm about the plate centre, lab orientation — exactly
    the fields `Asics_Moments4JCF_v3.m` reads from `dataStatic`/`dataAnalog`.
    Unnamed ``*n`` points are dropped (the script stops at the first one)."""
    import ezc3d
    from scipy.io import savemat
    c = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    par = c["parameters"]
    labels = [str(x).strip() for x in par["POINT"]["LABELS"]["value"]]
    keep = [i for i, n in enumerate(labels) if not n.startswith("*")]
    pts = c["data"]["points"][:3][:, keep]                     # [3,M,T] mm
    plats = c["data"]["platform"][:2]
    out = {"points": np.ascontiguousarray(pts),
           "labels": np.array([labels[i] for i in keep], dtype=object),
           "point_rate": float(np.atleast_1d(par["POINT"]["RATE"]["value"])[0]),
           "analog_rate": float(np.atleast_1d(par["ANALOG"]["RATE"]["value"])[0]),
           "force1": np.asarray(plats[0]["force"], float), "moment1": np.asarray(plats[0]["moment"], float),
           "force2": np.asarray(plats[1]["force"], float), "moment2": np.asarray(plats[1]["moment"], float)}
    savemat(str(dst), out, do_compression=True)
    return str(dst)


def run_matlab_legacy(static_c3d, dynamic_c3d, shoe: str, speed: str, out_mat,
                      body_mass: float | None = None, workdir=None,
                      matlab_bin: str = MATLAB_BIN, repo=None) -> dict:
    """Execute the ORIGINAL MATLAB routines on one trial via
    `harness/run_legacy_trial.m` and return the saved outputs as a dict.
    Raises if MATLAB is not installed."""
    import os, shutil, subprocess, tempfile
    from pathlib import Path
    from scipy.io import loadmat
    if not shutil.which(matlab_bin) and not os.path.exists(matlab_bin):
        raise FileNotFoundError(f"MATLAB not found at {matlab_bin}")
    repo = Path(repo or Path(__file__).resolve().parents[2])
    work = Path(workdir or tempfile.mkdtemp(prefix="legacy_matlab_"))
    s_mat = export_for_matlab(static_c3d, work / "static.mat")
    d_mat = export_for_matlab(dynamic_c3d, work / "dynamic.mat")
    shoe_index = {"AM": 1, "AG": 2, "AH": 3}[shoe]
    a1 = "[]" if body_mass is None else repr(float(body_mass))
    paths = [repo / HARNESS_DIR, repo / "matlab-inverse-dynamics" / "originals",
             repo / "matlab-inverse-dynamics" / "Inverse Dynamics Subroutines"]
    cmd = (";".join(f"addpath('{p}')" for p in paths)
           + f";run_legacy_trial('{s_mat}','{d_mat}',{shoe_index},'{speed}','{out_mat}',{a1})")
    subprocess.run([matlab_bin, "-batch", cmd], check=True, cwd=work,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return loadmat(str(out_mat), squeeze_me=True)


#: Fields of the MATLAB output kept in the committed reference file (the TMs
#: and per-frame forces are dropped to keep it small; they are rebuilt on
#: demand by `run_matlab_legacy`).
MATLAB_REFERENCE_FIELDS = (
    "A", "BW", "plate", "HS", "TO", "Rd", "Md", "COP", "mfoot", "mleg", "mthigh",
    "mrearfoot", "mforefoot", "R_Jankle", "R_Jknee", "R_Jhip", "R_Jmtp",
    "GlobalMmtp", "GlobalMankle", "GlobalMknee", "GlobalMhip",
    "LocalMmtp", "LocalMankle", "LocalMknee", "LocalMhip",
    "step_mtp_Mloc", "step_ankle_Mloc", "step_knee_Mloc", "step_hip_Mloc",
    "step_ankle_Mglob", "step_knee_Mglob", "step_hip_Mglob",
    "step_AA", "step_KA", "step_HA", "step_MTPA", "step_A_P", "step_K_P", "step_H_P",
    "step_bad", "step_contact", "PeakMmtp", "PeakMankle", "PeakMknee", "PeakMhip",
    "trialsNaN", "matlab_version", "generated")


def save_matlab_reference(m: dict, dst) -> str:
    """Keep the comparison-relevant subset of a `run_matlab_legacy` result."""
    from scipy.io import savemat
    sub = {k: m[k] for k in MATLAB_REFERENCE_FIELDS if k in m}
    savemat(str(dst), sub, do_compression=True)
    return str(dst)


def load_matlab_reference(path) -> dict:
    from scipy.io import loadmat
    return loadmat(str(path), squeeze_me=True)


def legacy_torque_lab(res: LegacyResult, joint: str) -> np.ndarray:
    """The legacy GLOBAL joint moment rotated into the corrected pipeline's
    lab frame [T,3] so it can be overlaid on `InverseDynamicsResult.joint_torque`
    (same convention: wrench on the distal segment by the proximal, about
    the joint centre)."""
    return lab_from_simm(res.global_moment[joint])


def legacy_stance_torque_lab(res: LegacyResult) -> np.ndarray:
    """[T,3,3] (ankle, knee, hip) global joint moments filtered at 12 Hz per
    stance exactly as the script did before exporting, rotated to the lab
    frame; zero outside the right-foot stances.  This — not the raw
    recursion output — is what the MATLAB wrote out."""
    out = np.zeros((len(res.t), 3, 3))
    for s, e in zip(res.hs, res.to):
        for j, name in enumerate(("ankle", "knee", "hip")):
            out[s:e + 1, j] = lab_from_simm(
                filterdata(res.global_moment[name][s:e + 1], res.rate, 12.0))
    return out


def stance_mean(steps, key: str, joint: str, component: int,
                include_bad: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """(mean [101], sd [101], n) of one stance-normalised curve across steps."""
    rows = [s[key][joint][:, component] for s in steps if include_bad or not s["bad"]]
    a = np.asarray(rows)
    return a.mean(axis=0), a.std(axis=0), len(rows)
