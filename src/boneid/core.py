"""boneid.core — datatypes and math for whole-body inverse dynamics.

Style: plain dataclasses of numpy arrays, free functions that operate on them.
No member functions. All quantities SI, lab frame unless suffixed `_local`.
Time-major arrays: [T, ...]. Rotations are world_from_segment matrices [T, 3, 3].

The model is a serial chain ordered distal -> proximal (e.g. foot, shank,
thigh, torso). Joint j connects segment j to segment j+1, so a chain of S
segments has S-1 joints and one residual wrench at the proximal end of the
top segment.

External loads are represented as a wrench (force + moment) about a FIXED lab
point (e.g. the treadmill origin). This avoids the numerically unstable
center-of-pressure division at low force; COP conversion helpers exist for
comparison and for Visual3D-style inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

GRAVITY = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """A — segment parameters for a serial chain, index 0 = most distal."""

    segment_names: list[str]        # [S]
    joint_names: list[str]          # [S-1], joint j between segment j and j+1
    mass: np.ndarray                # [S] kg
    com_local: np.ndarray           # [S,3] COM in segment frame, origin at the
                                    #        segment's PROXIMAL joint
    inertia_local: np.ndarray       # [S,3,3] about COM, segment frame, kg m^2
    length: np.ndarray              # [S] m (documentation/scaling; math uses
                                    #        the per-frame joint positions)


@dataclass
class ForcePlate:
    """B — force plate geometry in the lab frame."""

    corners: np.ndarray             # [4,3] m
    origin: np.ndarray              # [3] m, plate origin in lab frame
    r_lab: np.ndarray               # [3,3] lab_from_plate rotation


@dataclass
class GroundWrench:
    """C/F — external load as a wrench about a fixed lab point.

    `moment` is the total moment of the contact load about `point`. No COP is
    involved; use `wrench_from_cop` / `cop_from_wrench` to convert.
    """

    t: np.ndarray                   # [T] s
    rate: float                     # Hz
    force: np.ndarray               # [T,3] N, lab frame
    moment: np.ndarray              # [T,3] N m about `point`, lab frame
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [3] m


@dataclass
class SegmentKinematics:
    """D — per-frame pose of every segment in the chain."""

    t: np.ndarray                   # [T] s
    rate: float                     # Hz
    r_world: np.ndarray             # [T,S,3,3] world_from_segment
    prox_pos: np.ndarray            # [T,S,3] proximal end (= joint center)
    dist_pos: np.ndarray            # [T,S,3] distal end


@dataclass
class AnalysisParams:
    """Analysis knobs. E (treadmill speed) rides along here."""

    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    contact_threshold_n: float = 20.0   # vertical force marking ground contact
    lowpass_hz: float = 12.0            # kinematics low-pass cutoff (0 = off)
    force_lowpass_hz: float = 50.0      # force low-pass cutoff (0 = off)
    filter_order: int = 4
    treadmill_speed: float = 0.0        # m/s, belt speed (E)
    flag_crossover: bool = False        # flag steps with load on both plates


@dataclass
class InverseDynamicsResult:
    """OUTPUT — joint kinetics plus the residual wrench at the chain top.

    joint_force[:, j] / joint_torque[:, j] are the force/torque exerted ON
    segment j (distal) BY segment j+1 (proximal) — the classic net joint
    moment acting on the distal segment — expressed in the lab frame, torque
    about the joint center. The residual is the wrench that would have to act
    at the top segment's proximal end for its Newton-Euler balance to close;
    ideally zero.
    """

    t: np.ndarray                   # [T]
    joint_force: np.ndarray         # [T,J,3] N
    joint_torque: np.ndarray        # [T,J,3] N m, lab frame
    joint_torque_local: np.ndarray  # [T,J,3] N m, in proximal-segment frame
    residual_force: np.ndarray      # [T,3] N
    residual_torque: np.ndarray     # [T,3] N m about top proximal end


@dataclass
class EnergyAudit:
    """Energy bookkeeping: d(KE+PE)/dt should equal the sum of joint powers
    plus external (ground) wrench power plus residual wrench power."""

    t: np.ndarray                   # [T]
    kinetic: np.ndarray             # [T] J
    potential: np.ndarray           # [T] J
    joint_power: np.ndarray         # [T,J] W, tau_j . (omega_dist - omega_prox)
                                    #   (tau_j acts ON the distal segment)
    ground_power: np.ndarray        # [T] W
    residual_power: np.ndarray      # [T] W
    de_dt: np.ndarray               # [T] W, d(KE+PE)/dt by central difference
    power_total: np.ndarray         # [T] W, sum of the three power sources
    imbalance: np.ndarray           # [T] W, de_dt - power_total


# ---------------------------------------------------------------------------
# Small math utilities
# ---------------------------------------------------------------------------

def finite_difference(x: np.ndarray, rate: float) -> np.ndarray:
    """First central difference along axis 0; one-sided at the ends."""
    x = np.asarray(x, dtype=float)
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) * (rate / 2.0)
    d[0] = (x[1] - x[0]) * rate
    d[-1] = (x[-1] - x[-2]) * rate
    return d


def lowpass(x: np.ndarray, rate: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0. cutoff<=0 returns x."""
    if cutoff <= 0:
        return np.asarray(x, dtype=float)
    b, a = butter(order // 2, 2.0 * cutoff / rate)
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=0)


def fit_rigid_transform(neutral: np.ndarray, moving: np.ndarray):
    """Soderkvist & Wedin (1993) SVD rigid-body fit, vectorized over time.

    neutral: [M,3] marker positions in a reference frame.
    moving:  [T,M,3] the same markers over time.
    Returns (r [T,3,3], d [T,3], res [T]): moving ~= r @ neutral + d, and the
    RMS fit residual per frame (a rigidity/data-quality signal).
    """
    neutral = np.asarray(neutral, dtype=float)
    moving = np.asarray(moving, dtype=float)
    a0 = neutral - neutral.mean(axis=0)                      # [M,3]
    bmean = moving.mean(axis=1, keepdims=True)               # [T,1,3]
    b0 = moving - bmean                                      # [T,M,3]
    c = np.einsum("tmi,mj->tij", b0, a0)                     # [T,3,3]
    u, _, vt = np.linalg.svd(c)
    det = np.linalg.det(np.einsum("tij,tjk->tik", u, vt))
    fix = np.repeat(np.eye(3)[None], len(det), axis=0)
    fix[:, 2, 2] = det
    r = np.einsum("tij,tjk,tkl->til", u, fix, vt)            # [T,3,3]
    d = bmean[:, 0] - np.einsum("tij,j->ti", r, neutral.mean(axis=0))
    pred = np.einsum("tij,mj->tmi", r, neutral) + d[:, None]
    res = np.sqrt(np.mean(np.sum((moving - pred) ** 2, axis=2), axis=1))
    return r, d, res


def angular_velocity(r: np.ndarray, rate: float) -> np.ndarray:
    """World-frame angular velocity [.. ,3] from rotations [T,...,3,3].

    omega_hat = dR/dt @ R^T; extract and average the two skew halves for
    robustness against finite-difference asymmetry.
    """
    dr = finite_difference(r, rate)
    w_hat = np.einsum("...ij,...kj->...ik", dr, r)  # dR R^T
    w = np.empty(r.shape[:-2] + (3,))
    w[..., 0] = 0.5 * (w_hat[..., 2, 1] - w_hat[..., 1, 2])
    w[..., 1] = 0.5 * (w_hat[..., 0, 2] - w_hat[..., 2, 0])
    w[..., 2] = 0.5 * (w_hat[..., 1, 0] - w_hat[..., 0, 1])
    return w


def inertia_world(r: np.ndarray, inertia_local: np.ndarray) -> np.ndarray:
    """Similarity transform I_world = R I_local R^T (the legacy code's bug:
    it multiplied one-sidedly). Broadcasts over leading axes of r."""
    return np.einsum("...ij,jk,...lk->...il", r, inertia_local, r)


def wrench_from_cop(force: np.ndarray, cop: np.ndarray, free_moment: np.ndarray,
                    point: np.ndarray) -> np.ndarray:
    """Moment about `point` of a load applied at the COP with a free moment."""
    return np.cross(cop - point, force) + free_moment


def cop_from_wrench(force: np.ndarray, moment: np.ndarray, point: np.ndarray,
                    plane_height: float = 0.0, min_fz: float = 10.0):
    """COP on the horizontal plane z=plane_height, plus vertical free moment.

    Standard force-plate algebra; frames with |Fz| < min_fz get NaN COP.
    Returns (cop [T,3], free_moment [T,3])."""
    f = np.asarray(force, float)
    # transfer the moment to the point directly above/below `point` in the plane
    p0 = np.asarray(point, float).copy()
    shift = np.array([0.0, 0.0, plane_height - p0[2]])
    m = np.asarray(moment, float) - np.cross(shift, f)
    fz = f[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = -m[:, 1] / fz
        y = m[:, 0] / fz
    bad = np.abs(fz) < min_fz
    x[bad] = np.nan
    y[bad] = np.nan
    cop = np.stack([p0[0] + shift[0] + x, p0[1] + shift[1] + y,
                    np.full_like(x, plane_height)], axis=1)
    tz = m[:, 2] - (x * f[:, 1] - y * f[:, 0])
    free = np.zeros_like(f)
    free[:, 2] = np.where(bad, 0.0, tz)
    return cop, free


def detect_contact(vertical_force: np.ndarray, threshold: float,
                   min_gap: int = 10):
    """Boolean contact mask plus (onset, offset) index pairs.

    Gaps shorter than min_gap samples are bridged (debounce)."""
    on = np.asarray(vertical_force) > threshold
    # bridge short gaps
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return on, []
    gaps = np.flatnonzero(np.diff(idx) > 1)
    events = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            events.append((start, prev))
            start = i
        prev = i
    events.append((start, prev))
    mask = np.zeros_like(on)
    for s, e in events:
        mask[s:e + 1] = True
    return mask, events


# ---------------------------------------------------------------------------
# Segment kinematic quantities
# ---------------------------------------------------------------------------

def segment_com(skeleton: Skeleton, kin: SegmentKinematics) -> np.ndarray:
    """COM positions [T,S,3]: proximal end + R @ com_local."""
    return kin.prox_pos + np.einsum(
        "tsij,sj->tsi", kin.r_world, skeleton.com_local)


def chain_kinematics(skeleton: Skeleton, kin: SegmentKinematics,
                     lowpass_hz: float = 0.0, filter_order: int = 4):
    """Derived motion for Newton-Euler: returns dict with
    com [T,S,3], v_com, a_com, omega [T,S,3], alpha [T,S,3] (world frame).
    Optionally low-passes poses before differentiating."""
    r = kin.r_world
    prox = kin.prox_pos
    if lowpass_hz > 0:
        prox = lowpass(prox, kin.rate, lowpass_hz, filter_order)
        r = lowpass(r, kin.rate, lowpass_hz, filter_order)
        # re-orthonormalize filtered rotations via SVD projection
        u, _, vt = np.linalg.svd(r)
        det = np.linalg.det(np.einsum("...ij,...jk->...ik", u, vt))
        fix = np.zeros_like(r)
        fix[..., 0, 0] = 1.0
        fix[..., 1, 1] = 1.0
        fix[..., 2, 2] = det
        r = np.einsum("...ij,...jk,...kl->...il", u, fix, vt)
    com = prox + np.einsum("tsij,sj->tsi", r, skeleton.com_local)
    v_com = finite_difference(com, kin.rate)
    a_com = finite_difference(v_com, kin.rate)
    omega = angular_velocity(r, kin.rate)
    alpha = finite_difference(omega, kin.rate)
    return {"r": r, "com": com, "v_com": v_com, "a_com": a_com,
            "omega": omega, "alpha": alpha, "prox": prox}


# ---------------------------------------------------------------------------
# Newton-Euler inverse dynamics
# ---------------------------------------------------------------------------

def inverse_dynamics(skeleton: Skeleton, kin: SegmentKinematics,
                     ground: GroundWrench,
                     params: AnalysisParams | None = None) -> InverseDynamicsResult:
    """Bottom-up Newton-Euler over the chain.

    For each segment (distal -> proximal), with the distal load expressed as a
    wrench (F_d, M_d about point p_d):

        F_p = m a_com - F_d - m g
        M_p = I_w alpha + omega x (I_w omega)
              - M_d - (p_d - com) x F_d - (p_p - com) x F_p

    where I_w = R I_local R^T (full similarity transform) and the gyroscopic
    term is kept — both were wrong/missing in the legacy MATLAB.

    The ground wrench (about its fixed point) is the distal load on segment 0.
    Joint reactions are wrenches about the joint center; the negated top
    reaction is reported as the residual.
    """
    if params is None:
        params = AnalysisParams()
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    t_n, s_n = ck["com"].shape[:2]
    if len(ground.force) != t_n:
        raise ValueError("ground wrench and kinematics must share the time base")
    g = params.gravity
    force = ground.force
    moment = ground.moment
    if params.force_lowpass_hz > 0:
        force = lowpass(force, ground.rate, params.force_lowpass_hz,
                        params.filter_order)
        moment = lowpass(moment, ground.rate, params.force_lowpass_hz,
                         params.filter_order)

    j_n = s_n - 1
    joint_force = np.zeros((t_n, j_n, 3))
    joint_torque = np.zeros((t_n, j_n, 3))
    joint_torque_local = np.zeros((t_n, j_n, 3))

    f_d = force                                   # [T,3] load on current seg
    m_d = moment                                  # [T,3] about p_d
    p_d = np.broadcast_to(ground.point, (t_n, 3))  # [T,3]

    for s in range(s_n):
        m_seg = skeleton.mass[s]
        com = ck["com"][:, s]
        i_w = inertia_world(ck["r"][:, s], skeleton.inertia_local[s])
        iw_omega = np.einsum("tij,tj->ti", i_w, ck["omega"][:, s])
        rot_term = (np.einsum("tij,tj->ti", i_w, ck["alpha"][:, s])
                    + np.cross(ck["omega"][:, s], iw_omega))
        p_p = ck["prox"][:, s]
        f_p = m_seg * ck["a_com"][:, s] - f_d - m_seg * g
        m_p = (rot_term - m_d - np.cross(p_d - com, f_d)
               - np.cross(p_p - com, f_p))
        if s < j_n:
            joint_force[:, s] = f_p
            joint_torque[:, s] = m_p
            # torque in the proximal segment's frame (world_from_seg^T @ tau)
            joint_torque_local[:, s] = np.einsum(
                "tji,tj->ti", ck["r"][:, s + 1], m_p)
            # reaction on the next segment: negate, applied at the joint
            f_d = -f_p
            m_d = -m_p
            p_d = p_p
        else:
            residual_force = f_p
            residual_torque = m_p

    return InverseDynamicsResult(
        t=kin.t, joint_force=joint_force, joint_torque=joint_torque,
        joint_torque_local=joint_torque_local,
        residual_force=residual_force, residual_torque=residual_torque)


# ---------------------------------------------------------------------------
# Energy audit
# ---------------------------------------------------------------------------

def energy_audit(skeleton: Skeleton, kin: SegmentKinematics,
                 ground: GroundWrench, idres: InverseDynamicsResult,
                 params: AnalysisParams | None = None) -> EnergyAudit:
    """Check d(KE+PE)/dt against joint + ground + residual wrench power.

    All powers are exact rigid-body wrench powers: a wrench (F, M about point
    p) acting on a segment delivers F . v(p_material) + M . omega, computed by
    transferring the wrench to the segment COM. The audit is independent of
    the inverse-dynamics recursion (energy is never used inside it), so a
    small imbalance is a genuine correctness instrument.
    """
    if params is None:
        params = AnalysisParams()
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    t_n, s_n = ck["com"].shape[:2]
    g = params.gravity

    ke = np.zeros(t_n)
    pe = np.zeros(t_n)
    for s in range(s_n):
        m_seg = skeleton.mass[s]
        i_w = inertia_world(ck["r"][:, s], skeleton.inertia_local[s])
        w = ck["omega"][:, s]
        ke += 0.5 * m_seg * np.sum(ck["v_com"][:, s] ** 2, axis=1)
        ke += 0.5 * np.einsum("ti,tij,tj->t", w, i_w, w)
        pe += -m_seg * ck["com"][:, s] @ g

    # ground wrench power on segment 0 (transfer to its COM)
    force = ground.force
    moment = ground.moment
    if params.force_lowpass_hz > 0:
        force = lowpass(force, ground.rate, params.force_lowpass_hz,
                        params.filter_order)
        moment = lowpass(moment, ground.rate, params.force_lowpass_hz,
                         params.filter_order)
    com0 = ck["com"][:, 0]
    m_at_com0 = moment + np.cross(np.broadcast_to(ground.point, (t_n, 3))
                                  - com0, force)
    ground_power = (np.sum(force * ck["v_com"][:, 0], axis=1)
                    + np.sum(m_at_com0 * ck["omega"][:, 0], axis=1))

    # joint powers: wrench (F_j, M_j) acts ON the DISTAL segment at the joint,
    # its reaction acts on the proximal one. Net power injected =
    # M_j . (w_dist - w_prox) + F_j . (v_joint_on_dist - v_joint_on_prox);
    # the force term vanishes for an ideal (common-point) joint but we keep
    # it, catching joint-center inconsistencies between the two segments.
    j_n = s_n - 1
    joint_power = np.zeros((t_n, j_n))
    v_prox_all = finite_difference(ck["prox"], kin.rate)  # [T,S,3]
    for j in range(j_n):
        w_rel = ck["omega"][:, j] - ck["omega"][:, j + 1]
        joint_pt = ck["prox"][:, j]
        # material velocity of joint point on each segment
        v_on_dist = (ck["v_com"][:, j]
                     + np.cross(ck["omega"][:, j], joint_pt - ck["com"][:, j]))
        v_on_prox = (ck["v_com"][:, j + 1]
                     + np.cross(ck["omega"][:, j + 1],
                                joint_pt - ck["com"][:, j + 1]))
        joint_power[:, j] = (np.sum(idres.joint_torque[:, j] * w_rel, axis=1)
                             + np.sum(idres.joint_force[:, j]
                                      * (v_on_dist - v_on_prox), axis=1))

    # residual wrench power on the top segment at its proximal end
    top = s_n - 1
    top_pt = ck["prox"][:, top]
    v_top = (ck["v_com"][:, top]
             + np.cross(ck["omega"][:, top], top_pt - ck["com"][:, top]))
    residual_power = (np.sum(idres.residual_force * v_top, axis=1)
                      + np.sum(idres.residual_torque * ck["omega"][:, top],
                               axis=1))

    de_dt = finite_difference(ke + pe, kin.rate)
    power_total = ground_power + joint_power.sum(axis=1) + residual_power
    return EnergyAudit(
        t=kin.t, kinetic=ke, potential=pe, joint_power=joint_power,
        ground_power=ground_power, residual_power=residual_power,
        de_dt=de_dt, power_total=power_total,
        imbalance=de_dt - power_total)


# ---------------------------------------------------------------------------
# Two legs sharing a pelvis (branched chain)
# ---------------------------------------------------------------------------

@dataclass
class TwoLegInverseDynamics:
    """Both legs analysed to the hip, plus a shared-pelvis balance.

    `right`/`left` are per-leg results over joints [ankle, knee, hip] (hip is
    the wrench the pelvis exerts on that thigh, same on-distal-by-proximal
    convention). The pelvis segment receives BOTH hip reactions; the residual
    is the wrench required at the pelvis' proximal end (L5S1) to close its
    Newton-Euler balance — it carries everything unmodeled above the pelvis
    (trunk, arms, head) and is a first-class output.
    """

    t: np.ndarray
    right: InverseDynamicsResult
    left: InverseDynamicsResult
    residual_force: np.ndarray      # [T,3] at the pelvis proximal end
    residual_torque: np.ndarray     # [T,3] about the pelvis proximal end


def slice_chain(skeleton: Skeleton, kin: SegmentKinematics, indices):
    """Sub-chain (Skeleton, SegmentKinematics) for segment `indices` (in
    distal->proximal order). Joint names survive only between kept neighbours
    that were neighbours before; for the common contiguous case this is just
    the corresponding slice."""
    idx = list(indices)
    joints = [skeleton.joint_names[j] for j in idx[:-1]]
    sub_skel = Skeleton(
        segment_names=[skeleton.segment_names[i] for i in idx],
        joint_names=joints,
        mass=skeleton.mass[idx],
        com_local=skeleton.com_local[idx],
        inertia_local=skeleton.inertia_local[idx],
        length=skeleton.length[idx])
    sub_kin = SegmentKinematics(
        t=kin.t, rate=kin.rate,
        r_world=kin.r_world[:, idx],
        prox_pos=kin.prox_pos[:, idx],
        dist_pos=kin.dist_pos[:, idx])
    return sub_skel, sub_kin


def _leg_with_hip(res: InverseDynamicsResult, kin_pelvis_r: np.ndarray) -> InverseDynamicsResult:
    """Repackage a 3-segment leg result so the hip appears as joint 2.

    The 3-segment chain's 'residual' IS the hip wrench (acting on the thigh by
    the pelvis). `kin_pelvis_r` [T,3,3] expresses the hip torque in the pelvis
    (proximal-segment) frame for the local component."""
    hip_local = np.einsum("tji,tj->ti", kin_pelvis_r, res.residual_torque)
    return InverseDynamicsResult(
        t=res.t,
        joint_force=np.concatenate(
            [res.joint_force, res.residual_force[:, None]], axis=1),
        joint_torque=np.concatenate(
            [res.joint_torque, res.residual_torque[:, None]], axis=1),
        joint_torque_local=np.concatenate(
            [res.joint_torque_local, hip_local[:, None]], axis=1),
        residual_force=np.zeros_like(res.residual_force),
        residual_torque=np.zeros_like(res.residual_torque))


def inverse_dynamics_two_legs(skel_r: Skeleton, kin_r: SegmentKinematics,
                              ground_r: GroundWrench,
                              skel_l: Skeleton, kin_l: SegmentKinematics,
                              ground_l: GroundWrench,
                              params: AnalysisParams | None = None,
                              ) -> TwoLegInverseDynamics:
    """Whole-lower-body inverse dynamics: two legs + shared pelvis.

    `skel_r/skel_l` and `kin_r/kin_l` are the usual 4-segment chains
    [foot, shank, thigh, pelvis] (e.g. from io_v3d.build_chain per side); the
    two pelvis entries must describe the SAME segment — the pelvis pose and
    parameters are taken from the right chain, and each leg's hip point from
    its own chain.

    Each leg runs the serial recursion foot->thigh; its top 'residual' is the
    hip wrench. The pelvis balance then uses both negated hip wrenches as
    distal loads:

        F_p = m a - (-F_hip_r) - (-F_hip_l) - m g
        M_p = I_w alpha + omega x (I_w omega)
              - sum_i [ -M_hip_i + (p_hip_i - com) x (-F_hip_i) ]
              - (p_l5s1 - com) x F_p
    """
    if params is None:
        params = AnalysisParams()
    legs = {}
    hip_pts = {}
    for tag, skel, kin, ground in (("r", skel_r, kin_r, ground_r),
                                   ("l", skel_l, kin_l, ground_l)):
        leg_skel, leg_kin = slice_chain(skel, kin, [0, 1, 2])
        res = inverse_dynamics(leg_skel, leg_kin, ground, params)
        legs[tag] = res
        hip_pts[tag] = kin.prox_pos[:, 2]        # thigh proximal = hip centre

    # pelvis kinematics (from the right chain), filtered like everything else
    pel_skel, pel_kin = slice_chain(skel_r, kin_r, [3])
    ck = chain_kinematics(pel_skel, pel_kin, params.lowpass_hz,
                          params.filter_order)
    m_p = pel_skel.mass[0]
    com = ck["com"][:, 0]
    i_w = inertia_world(ck["r"][:, 0], pel_skel.inertia_local[0])
    iw_omega = np.einsum("tij,tj->ti", i_w, ck["omega"][:, 0])
    rot_term = (np.einsum("tij,tj->ti", i_w, ck["alpha"][:, 0])
                + np.cross(ck["omega"][:, 0], iw_omega))

    f_d = -(legs["r"].residual_force + legs["l"].residual_force)
    residual_force = m_p * ck["a_com"][:, 0] - f_d - m_p * params.gravity
    m_d = np.zeros_like(f_d)
    for tag in ("r", "l"):
        m_d += (-legs[tag].residual_torque
                + np.cross(hip_pts[tag] - com, -legs[tag].residual_force))
    p_top = ck["prox"][:, 0]
    residual_torque = (rot_term - m_d
                       - np.cross(p_top - com, residual_force))

    pelvis_r = chain_kinematics(
        *slice_chain(skel_r, kin_r, [3]), params.lowpass_hz,
        params.filter_order)["r"][:, 0]
    return TwoLegInverseDynamics(
        t=kin_r.t,
        right=_leg_with_hip(legs["r"], pelvis_r),
        left=_leg_with_hip(legs["l"], pelvis_r),
        residual_force=residual_force,
        residual_torque=residual_torque)


def energy_audit_two_legs(skel_r: Skeleton, kin_r: SegmentKinematics,
                          ground_r: GroundWrench,
                          skel_l: Skeleton, kin_l: SegmentKinematics,
                          ground_l: GroundWrench,
                          two: TwoLegInverseDynamics,
                          params: AnalysisParams | None = None) -> EnergyAudit:
    """Energy audit of the branched two-leg + pelvis model.

    Same instrument as `energy_audit`, over all 7 segments: d(KE+PE)/dt must
    equal both ground-wrench powers + all 6 joint powers + the L5S1 residual
    power. joint_power columns are [r_ankle, r_knee, r_hip, l_ankle, l_knee,
    l_hip]. Closure is independent of how wrong the anthropometry is — it
    checks the numerical consistency of the recursion, not the model.
    """
    if params is None:
        params = AnalysisParams()
    t_n = len(kin_r.t)
    ke = np.zeros(t_n)
    pe = np.zeros(t_n)
    ground_power = np.zeros(t_n)
    joint_power = np.zeros((t_n, 6))

    pel_skel, pel_kin = slice_chain(skel_r, kin_r, [3])
    ck_pel = chain_kinematics(pel_skel, pel_kin, params.lowpass_hz,
                              params.filter_order)

    for col0, (skel, kin, ground, res) in enumerate(
            [(skel_r, kin_r, ground_r, two.right),
             (skel_l, kin_l, ground_l, two.left)]):
        leg_skel, leg_kin = slice_chain(skel, kin, [0, 1, 2])
        ck = chain_kinematics(leg_skel, leg_kin, params.lowpass_hz,
                              params.filter_order)
        for s in range(3):
            m_seg = leg_skel.mass[s]
            i_w = inertia_world(ck["r"][:, s], leg_skel.inertia_local[s])
            w = ck["omega"][:, s]
            ke += 0.5 * m_seg * np.sum(ck["v_com"][:, s] ** 2, axis=1)
            ke += 0.5 * np.einsum("ti,tij,tj->t", w, i_w, w)
            pe += -m_seg * ck["com"][:, s] @ params.gravity

        force = ground.force
        moment = ground.moment
        if params.force_lowpass_hz > 0:
            force = lowpass(force, ground.rate, params.force_lowpass_hz,
                            params.filter_order)
            moment = lowpass(moment, ground.rate, params.force_lowpass_hz,
                             params.filter_order)
        com0 = ck["com"][:, 0]
        m_at = moment + np.cross(
            np.broadcast_to(ground.point, (t_n, 3)) - com0, force)
        ground_power += (np.sum(force * ck["v_com"][:, 0], axis=1)
                         + np.sum(m_at * ck["omega"][:, 0], axis=1))

        # ankle, knee: between leg segments; hip: distal=thigh, prox=pelvis
        for j in range(3):
            w_dist = ck["omega"][:, j]
            v_dist = (ck["v_com"][:, j]
                      + np.cross(w_dist, ck["prox"][:, j] - ck["com"][:, j]))
            if j < 2:
                w_prox = ck["omega"][:, j + 1]
                v_prox = (ck["v_com"][:, j + 1]
                          + np.cross(w_prox,
                                     ck["prox"][:, j] - ck["com"][:, j + 1]))
            else:
                w_prox = ck_pel["omega"][:, 0]
                v_prox = (ck_pel["v_com"][:, 0]
                          + np.cross(w_prox,
                                     ck["prox"][:, j] - ck_pel["com"][:, 0]))
            joint_power[:, col0 * 3 + j] = (
                np.sum(res.joint_torque[:, j] * (w_dist - w_prox), axis=1)
                + np.sum(res.joint_force[:, j] * (v_dist - v_prox), axis=1))

    # pelvis segment energy + residual power at L5S1
    m_seg = pel_skel.mass[0]
    i_w = inertia_world(ck_pel["r"][:, 0], pel_skel.inertia_local[0])
    w = ck_pel["omega"][:, 0]
    ke += 0.5 * m_seg * np.sum(ck_pel["v_com"][:, 0] ** 2, axis=1)
    ke += 0.5 * np.einsum("ti,tij,tj->t", w, i_w, w)
    pe += -m_seg * ck_pel["com"][:, 0] @ params.gravity

    top_pt = ck_pel["prox"][:, 0]
    v_top = (ck_pel["v_com"][:, 0]
             + np.cross(w, top_pt - ck_pel["com"][:, 0]))
    residual_power = (np.sum(two.residual_force * v_top, axis=1)
                      + np.sum(two.residual_torque * w, axis=1))

    de_dt = finite_difference(ke + pe, kin_r.rate)
    power_total = ground_power + joint_power.sum(axis=1) + residual_power
    return EnergyAudit(
        t=kin_r.t, kinetic=ke, potential=pe, joint_power=joint_power,
        ground_power=ground_power, residual_power=residual_power,
        de_dt=de_dt, power_total=power_total,
        imbalance=de_dt - power_total)


# ---------------------------------------------------------------------------
# Whole body: two legs + pelvis + torso(+head) + two arms(+hands)
# ---------------------------------------------------------------------------

@dataclass
class WholeBodyInverseDynamics:
    """The full 12-segment model: nothing but soft tissue is left unmodelled.

    Segments: 2 x [foot, shank, thigh] + pelvis + torso(+head lumped)
    + 2 x [upper arm, forearm(+hand lumped)].
    Joints: 2 x [ankle, knee, hip] + L5S1 + 2 x [shoulder, elbow] = 11.

    SIGN CONVENTION, everywhere, no exceptions: a reported joint wrench is the
    wrench exerted ON THE DISTAL SEGMENT BY THE PROXIMAL SEGMENT, expressed in
    the lab frame, with the torque taken about the joint centre.  "Distal"
    means further from the torso along the chain, so:

        ankle     ON foot        BY shank
        knee      ON shank       BY thigh
        hip       ON thigh       BY pelvis
        L5S1      ON PELVIS      BY TORSO      <- the pelvis is the distal side
        shoulder  ON upper arm   BY torso
        elbow     ON forearm     BY upper arm

    L5S1 is the one that needs saying out loud, because a top-down upper-body
    analysis would call the torso distal.  We do not: the pelvis is treated as
    distal to the torso, mirroring the hips, so `l5s1_force` is what the torso
    pushes down on the pelvis with.  In quiet standing it is therefore
    NEGATIVE in z, of magnitude (torso + head + both arms) * g.

    `right`/`left` are the per-leg results over joints [ankle, knee, hip] --
    byte-identical to `inverse_dynamics_two_legs`, since adding the upper body
    cannot change a bottom-up leg recursion.

    Two-sided arrays are indexed [T, side, 3] with side 0 = RIGHT, 1 = LEFT.

    The RESIDUAL is now at the TORSO and reported AT THE TORSO COM (force, and
    torque about the COM): it is the wrench that would have to act on the torso
    for its Newton-Euler balance to close once the L5S1 and both shoulder
    reactions are applied.  With every segment of the body modelled it should
    be small -- it is the whole-model error term (anthropometry, soft tissue,
    marker artefact, force-plate calibration), not a missing body part.
    """

    t: np.ndarray
    right: InverseDynamicsResult    # legs, joints [ankle, knee, hip]
    left: InverseDynamicsResult
    l5s1_force: np.ndarray          # [T,3] N,   ON pelvis BY torso
    l5s1_torque: np.ndarray         # [T,3] N m, about the L5S1 point
    shoulder_force: np.ndarray      # [T,2,3] N,   ON upper arm BY torso
    shoulder_torque: np.ndarray     # [T,2,3] N m, about the acromion marker
    elbow_force: np.ndarray         # [T,2,3] N,   ON forearm BY upper arm
    elbow_torque: np.ndarray        # [T,2,3] N m, about the epicondyle marker
    residual_force: np.ndarray      # [T,3] N,   at the torso COM
    residual_torque: np.ndarray     # [T,3] N m, about the torso COM


#: Sub-chain segment indices inside the upper-body Skeleton/SegmentKinematics
#: (see io_v3d.build_upper_body): each arm is contiguous distal->proximal.
UPPER_BODY_ARM = {"r": [0, 1], "l": [2, 3]}
UPPER_BODY_TORSO = 4


def _zero_wrench(kin: SegmentKinematics) -> GroundWrench:
    """A zero distal load on the same time base — the free end of an arm."""
    z = np.zeros((len(kin.t), 3))
    return GroundWrench(t=kin.t, rate=kin.rate, force=z, moment=z.copy(),
                        point=np.zeros(3))


def inverse_dynamics_whole_body(skel_r: Skeleton, kin_r: SegmentKinematics,
                                ground_r: GroundWrench,
                                skel_l: Skeleton, kin_l: SegmentKinematics,
                                ground_l: GroundWrench,
                                skel_u: Skeleton, kin_u: SegmentKinematics,
                                params: AnalysisParams | None = None,
                                ) -> WholeBodyInverseDynamics:
    """Whole-body inverse dynamics over all 12 segments.

    Inputs are the two leg chains [foot, shank, thigh, pelvis] as
    `inverse_dynamics_two_legs` takes them, plus ONE upper-body pair
    (`skel_u`, `kin_u`) holding [r_forearm_hand, r_upper_arm, l_forearm_hand,
    l_upper_arm, torso] — `io_v3d.build_upper_body` produces exactly that.  The
    torso's L5S1 origin must be the same point as the pelvis chains' L5S1
    origin (both loaders use mid(mid-ASIS, SACR), so they are).

    Three stages, each a plain Newton-Euler balance:

    1. LEGS, bottom-up from the ground wrenches, then the pelvis balance with
       both hip reactions.  This is `inverse_dynamics_two_legs` unchanged, and
       what it calls its "residual" is now given its real name: the L5S1 JOINT
       wrench, the wrench the torso exerts on the pelvis.  Adding the upper
       body changes no leg number.

    2. ARMS, each a 2-segment serial recursion [forearm+hand, upper arm] with
       ZERO distal load (nothing is in the hands).  Joint 0 of that recursion
       is the elbow; its "residual" is the shoulder wrench.

    3. TORSO, balanced against the three reactions it receives — the negated
       L5S1 wrench at the L5S1 point and the negated shoulder wrenches at the
       two acromia:

           F_res = m a_com - sum_i F_i - m g
           M_res = I_w alpha + omega x (I_w omega)
                   - sum_i [ M_i + (p_i - com) x F_i ]

       with (F_i, M_i about p_i) the reaction wrenches.  The residual is
       reported at the COM, so no final transfer term appears in M_res.
    """
    if params is None:
        params = AnalysisParams()

    two = inverse_dynamics_two_legs(skel_r, kin_r, ground_r,
                                    skel_l, kin_l, ground_l, params)
    l5s1_force = two.residual_force
    l5s1_torque = two.residual_torque
    l5s1_point = chain_kinematics(*slice_chain(skel_r, kin_r, [3]),
                                  params.lowpass_hz,
                                  params.filter_order)["prox"][:, 0]

    t_n = len(kin_u.t)
    shoulder_force = np.zeros((t_n, 2, 3))
    shoulder_torque = np.zeros((t_n, 2, 3))
    elbow_force = np.zeros((t_n, 2, 3))
    elbow_torque = np.zeros((t_n, 2, 3))
    shoulder_point = np.zeros((t_n, 2, 3))

    for k, side in enumerate(("r", "l")):
        arm_skel, arm_kin = slice_chain(skel_u, kin_u, UPPER_BODY_ARM[side])
        res = inverse_dynamics(arm_skel, arm_kin, _zero_wrench(arm_kin), params)
        elbow_force[:, k] = res.joint_force[:, 0]
        elbow_torque[:, k] = res.joint_torque[:, 0]
        shoulder_force[:, k] = res.residual_force
        shoulder_torque[:, k] = res.residual_torque
        shoulder_point[:, k] = chain_kinematics(
            arm_skel, arm_kin, params.lowpass_hz,
            params.filter_order)["prox"][:, 1]

    # --- torso balance
    tor_skel, tor_kin = slice_chain(skel_u, kin_u, [UPPER_BODY_TORSO])
    ck = chain_kinematics(tor_skel, tor_kin, params.lowpass_hz,
                          params.filter_order)
    m_t = tor_skel.mass[0]
    com = ck["com"][:, 0]
    i_w = inertia_world(ck["r"][:, 0], tor_skel.inertia_local[0])
    rot_term = (np.einsum("tij,tj->ti", i_w, ck["alpha"][:, 0])
                + np.cross(ck["omega"][:, 0],
                           np.einsum("tij,tj->ti", i_w, ck["omega"][:, 0])))

    loads = [(-l5s1_force, -l5s1_torque, l5s1_point)]
    for k in (0, 1):
        loads.append((-shoulder_force[:, k], -shoulder_torque[:, k],
                      shoulder_point[:, k]))

    f_sum = np.zeros((t_n, 3))
    m_sum = np.zeros((t_n, 3))
    for f_i, m_i, p_i in loads:
        f_sum += f_i
        m_sum += m_i + np.cross(p_i - com, f_i)

    residual_force = m_t * ck["a_com"][:, 0] - f_sum - m_t * params.gravity
    residual_torque = rot_term - m_sum

    return WholeBodyInverseDynamics(
        t=kin_r.t, right=two.right, left=two.left,
        l5s1_force=l5s1_force, l5s1_torque=l5s1_torque,
        shoulder_force=shoulder_force, shoulder_torque=shoulder_torque,
        elbow_force=elbow_force, elbow_torque=elbow_torque,
        residual_force=residual_force, residual_torque=residual_torque)


def energy_audit_whole_body(skel_r: Skeleton, kin_r: SegmentKinematics,
                            ground_r: GroundWrench,
                            skel_l: Skeleton, kin_l: SegmentKinematics,
                            ground_l: GroundWrench,
                            skel_u: Skeleton, kin_u: SegmentKinematics,
                            whole: WholeBodyInverseDynamics,
                            params: AnalysisParams | None = None) -> EnergyAudit:
    """Energy audit over all 12 segments and all 11 joints.

    d(KE+PE)/dt of the whole body must equal both ground-wrench powers + the
    11 joint powers + the torso residual power.  `joint_power` columns:

        0-2   r_ankle, r_knee, r_hip
        3-5   l_ankle, l_knee, l_hip
        6     L5S1        (distal = pelvis, proximal = torso)
        7,8   r_shoulder, r_elbow
        9,10  l_shoulder, l_elbow

    Every joint power is the full wrench power M . (w_dist - w_prox)
    + F . (v_joint_on_dist - v_joint_on_prox); the force term vanishes for an
    ideal joint and catches joint-centre inconsistencies when it does not.
    Closure is a numerical-consistency instrument: it says nothing about
    whether the anthropometry is right, only that the recursion and the audit
    agree.
    """
    if params is None:
        params = AnalysisParams()
    t_n = len(kin_r.t)
    ke = np.zeros(t_n)
    pe = np.zeros(t_n)
    ground_power = np.zeros(t_n)
    joint_power = np.zeros((t_n, 11))

    def accumulate(skel, ck, s):
        i_w = inertia_world(ck["r"][:, s], skel.inertia_local[s])
        w = ck["omega"][:, s]
        return (0.5 * skel.mass[s] * np.sum(ck["v_com"][:, s] ** 2, axis=1)
                + 0.5 * np.einsum("ti,tij,tj->t", w, i_w, w),
                -skel.mass[s] * ck["com"][:, s] @ params.gravity)

    def wrench_power(force, torque, point, ck_d, s_d, ck_p, s_p):
        w_d = ck_d["omega"][:, s_d]
        w_p = ck_p["omega"][:, s_p]
        v_d = ck_d["v_com"][:, s_d] + np.cross(w_d, point - ck_d["com"][:, s_d])
        v_p = ck_p["v_com"][:, s_p] + np.cross(w_p, point - ck_p["com"][:, s_p])
        return (np.sum(torque * (w_d - w_p), axis=1)
                + np.sum(force * (v_d - v_p), axis=1))

    pel_skel, pel_kin = slice_chain(skel_r, kin_r, [3])
    ck_pel = chain_kinematics(pel_skel, pel_kin, params.lowpass_hz,
                              params.filter_order)
    tor_skel, tor_kin = slice_chain(skel_u, kin_u, [UPPER_BODY_TORSO])
    ck_tor = chain_kinematics(tor_skel, tor_kin, params.lowpass_hz,
                              params.filter_order)

    # --- legs + ground
    for col0, (skel, kin, ground, res) in enumerate(
            [(skel_r, kin_r, ground_r, whole.right),
             (skel_l, kin_l, ground_l, whole.left)]):
        leg_skel, leg_kin = slice_chain(skel, kin, [0, 1, 2])
        ck = chain_kinematics(leg_skel, leg_kin, params.lowpass_hz,
                              params.filter_order)
        for s in range(3):
            d_ke, d_pe = accumulate(leg_skel, ck, s)
            ke += d_ke
            pe += d_pe

        force = ground.force
        moment = ground.moment
        if params.force_lowpass_hz > 0:
            force = lowpass(force, ground.rate, params.force_lowpass_hz,
                            params.filter_order)
            moment = lowpass(moment, ground.rate, params.force_lowpass_hz,
                             params.filter_order)
        com0 = ck["com"][:, 0]
        m_at = moment + np.cross(
            np.broadcast_to(ground.point, (t_n, 3)) - com0, force)
        ground_power += (np.sum(force * ck["v_com"][:, 0], axis=1)
                         + np.sum(m_at * ck["omega"][:, 0], axis=1))

        for j in range(3):
            ck_p, s_p = (ck, j + 1) if j < 2 else (ck_pel, 0)
            joint_power[:, col0 * 3 + j] = wrench_power(
                res.joint_force[:, j], res.joint_torque[:, j],
                ck["prox"][:, j], ck, j, ck_p, s_p)

    # --- pelvis segment + L5S1 (distal = pelvis, proximal = torso)
    d_ke, d_pe = accumulate(pel_skel, ck_pel, 0)
    ke += d_ke
    pe += d_pe
    joint_power[:, 6] = wrench_power(
        whole.l5s1_force, whole.l5s1_torque, ck_pel["prox"][:, 0],
        ck_pel, 0, ck_tor, 0)

    # --- torso segment + residual at its COM
    d_ke, d_pe = accumulate(tor_skel, ck_tor, 0)
    ke += d_ke
    pe += d_pe
    residual_power = (np.sum(whole.residual_force * ck_tor["v_com"][:, 0],
                             axis=1)
                      + np.sum(whole.residual_torque * ck_tor["omega"][:, 0],
                               axis=1))

    # --- arms: shoulder (distal = upper arm, prox = torso), elbow
    for k, side in enumerate(("r", "l")):
        arm_skel, arm_kin = slice_chain(skel_u, kin_u, UPPER_BODY_ARM[side])
        ck_arm = chain_kinematics(arm_skel, arm_kin, params.lowpass_hz,
                                  params.filter_order)
        for s in range(2):
            d_ke, d_pe = accumulate(arm_skel, ck_arm, s)
            ke += d_ke
            pe += d_pe
        joint_power[:, 7 + 2 * k] = wrench_power(
            whole.shoulder_force[:, k], whole.shoulder_torque[:, k],
            ck_arm["prox"][:, 1], ck_arm, 1, ck_tor, 0)
        joint_power[:, 8 + 2 * k] = wrench_power(
            whole.elbow_force[:, k], whole.elbow_torque[:, k],
            ck_arm["prox"][:, 0], ck_arm, 0, ck_arm, 1)

    de_dt = finite_difference(ke + pe, kin_r.rate)
    power_total = ground_power + joint_power.sum(axis=1) + residual_power
    return EnergyAudit(
        t=kin_r.t, kinetic=ke, potential=pe, joint_power=joint_power,
        ground_power=ground_power, residual_power=residual_power,
        de_dt=de_dt, power_total=power_total,
        imbalance=de_dt - power_total)


# ---------------------------------------------------------------------------
# Whole-body power decompositions
# ---------------------------------------------------------------------------
#
# Koenig's decomposition splits the mechanical energy of a body into a
# center-of-mass ("external") part and a part describing motion RELATIVE to
# the COM ("peripheral" / "internal"):
#
#     KE + PE = [ 1/2 M |v_com|^2 + M g h_com ]                      (COM)
#             + sum_i [ 1/2 m_i |v_i - v_com|^2 + 1/2 w_i . I_i w_i ] (peripheral)
#
# and, by Newton for the COM, the time derivative of the first bracket is
# exactly the ground-reaction power sum_limbs F_limb . v_com. The functions
# below compute each piece. Together they are the individual-limbs method of
# Donelan, Kram & Kuo (2002) plus the peripheral term of Zelik & Kuo (2010).

def com_velocity_from_grf(forces, body_mass: float, rate: float,
                          gravity: np.ndarray = GRAVITY,
                          detrend: bool = True) -> np.ndarray:
    """COM velocity [T,3] by integrating the summed ground reaction forces.

    `forces` is a list of [T,3] lab-frame GRFs, one per limb / force plate
    (a single [T,3] array is accepted too). The COM acceleration follows from
    Newton for the whole body,

        a_com = (sum_limbs F_limb) / M + g,

    and is integrated (cumulative trapezoid) to give velocity. The integration
    constant is unknowable from force alone; with `detrend=True` the mean of
    each velocity component is removed, which is the standard treatment for
    steady locomotion on a treadmill: over an integer number of strides the
    lab-frame COM velocity is periodic, so its mean is zero (Donelan, Kram &
    Kuo 2002, J Biomech 35:117-124).

    CAVEAT (the caller's responsibility): detrending is only correct over a
    whole number of strides of steady locomotion. Crop the record to
    integer-stride windows before calling; on a partial stride, an accelerating
    subject, or overground data the removed mean is not the true offset. Any
    force calibration/zero drift also lands in this mean.
    """
    if isinstance(forces, np.ndarray):
        forces = [forces]
    total = np.zeros_like(np.asarray(forces[0], dtype=float))
    for f in forces:
        total = total + np.asarray(f, dtype=float)
    a = total / float(body_mass) + np.asarray(gravity, dtype=float)
    dt = 1.0 / rate
    v = np.zeros_like(a)
    v[1:] = np.cumsum(0.5 * (a[1:] + a[:-1]), axis=0) * dt
    if detrend:
        v = v - v.mean(axis=0, keepdims=True)
    return v


def com_power(force: np.ndarray, v_com: np.ndarray) -> np.ndarray:
    """Individual-limb COM power [T] W: P_i = F_i . v_com.

    Donelan, Kram & Kuo (2002), "Simultaneous positive and negative external
    mechanical work in human walking", J Biomech 35:117-124: the external
    mechanical work rate of one limb is that limb's ground reaction force
    dotted with the whole-body COM velocity. Call once per limb (one force
    plate each) and sum for the total external power; summing the FORCES first
    instead would hide the simultaneous positive/negative work the two limbs do
    against each other in double support, which is the whole point of the
    individual-limbs method.

    `v_com` is the whole-body COM velocity [T,3] — either measured
    kinematically (sum m_i v_i / M) or from `com_velocity_from_grf`.
    """
    return np.sum(np.asarray(force, float) * np.asarray(v_com, float), axis=1)


def peripheral_power(skeleton: Skeleton, kin: SegmentKinematics,
                     v_com: np.ndarray,
                     params: AnalysisParams | None = None) -> np.ndarray:
    """Peripheral (COM-relative) power [T] W of the MODELED segments.

    Zelik & Kuo (2010), "Human walking isn't all hard work: evidence of soft
    tissue contributions to total work", J Exp Biol 213:4257-4264. The
    peripheral energy is the mechanical energy of the segments in a frame
    translating with the COM,

        E_per = sum_i [ 1/2 m_i |v_i - v_com|^2 + 1/2 w_i . I_w,i w_i ]

    and this returns its time derivative (central difference), the peripheral
    power. Added to the COM power (`com_power`) it gives the total rate of
    change of the segments' mechanical energy — Koenig's decomposition.

    IMPORTANT — partial models: the sum runs over the segments in `skeleton`
    ONLY. Our usual chain is legs + pelvis, roughly half of body mass, with the
    trunk, arms and head unmodeled, so what comes back is the peripheral power
    OF THE MODELED SEGMENTS, not of the body. It is not comparable to a
    whole-body peripheral power, and Koenig's identity only closes against the
    energy of the same modeled set. (Zelik & Kuo used a full-body model; arm
    and trunk motion carry real peripheral power.)

    `v_com` [T,3] is the whole-body COM velocity used as the reference frame —
    pass the best available estimate, which need not come from `skeleton`.
    Segment kinematics are recomputed with `chain_kinematics` under the same
    `params` used everywhere else, so filtering stays consistent.
    """
    if params is None:
        params = AnalysisParams()
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    t_n, s_n = ck["com"].shape[:2]
    v_com = np.asarray(v_com, dtype=float)
    e_per = np.zeros(t_n)
    for s in range(s_n):
        v_rel = ck["v_com"][:, s] - v_com
        w = ck["omega"][:, s]
        i_w = inertia_world(ck["r"][:, s], skeleton.inertia_local[s])
        e_per += 0.5 * skeleton.mass[s] * np.sum(v_rel ** 2, axis=1)
        e_per += 0.5 * np.einsum("ti,tij,tj->t", w, i_w, w)
    return finite_difference(e_per, kin.rate)


def foot_power_ud(skeleton: Skeleton, kin: SegmentKinematics,
                  ground: GroundWrench,
                  params: AnalysisParams | None = None,
                  cop: np.ndarray | None = None,
                  plane_height: float = 0.0,
                  surface_velocity: np.ndarray | None = None) -> np.ndarray:
    """Distal-to-foot ("unified deformable segment") power [T] W.

    Takahashi, Kepple & Stanhope (2012), "A unified deformable (UD) segment
    model for quantifying total power of anatomical and prosthetic below-knee
    structures during stance in gait", J Biomech 45:2662-2667, as advocated by
    Zelik & Honert (2018), "Ankle and foot power in gait analysis: implications
    for science, technology and clinical assessment", J Biomech 75:1-12. The
    power delivered by everything distal to the (rigid) foot segment is

        P_UD = F_grf . v_ecb + M_free . w_foot,
        v_ecb = v_com_foot + w_foot x (COP - com_foot),

    i.e. the GRF dotted with the velocity of the RIGID-BODY material point of
    the foot instantaneously coincident with the center of pressure, plus the
    free moment acting through the foot's angular velocity. Any difference
    between this and the true deformable-structure power shows up as the
    deformation (shoe, heel pad, arch) work the rigid model cannot represent —
    which is exactly what the measure is for: for a perfectly rigid foot welded
    to the ground P_UD is identically zero however large the GRF.

    Segment 0 of the chain is the foot. The ground wrench is filtered with
    `params.force_lowpass_hz` like everywhere else. If `cop` is None the COP
    and vertical free moment are derived from the wrench with `cop_from_wrench`
    on the horizontal plane z=`plane_height` (default 0.0 — pass the force
    plate surface height, e.g. -0.1746 m for the Visual3D exports here).
    Frames with no contact have an undefined (NaN) COP and get zero power.

    The plane choice does not actually matter: sliding the COP along the line
    of action of the force changes `cop` and the transfer moment by equal and
    opposite amounts, so P_UD is exactly the ground wrench's power on the rigid
    foot segment, F . v_com_foot + M_about_com . w_foot, whichever plane the
    COP was reduced to (verified to 1e-12 W on real data). The COP form is kept
    because it is how the measure is defined and reported in the literature.

    FRAME (matters on a treadmill). P_UD is a wrench power and therefore
    Galilean frame-dependent: changing the analysis frame by a velocity u
    changes it by -F . u. Everything else in this module works in the lab
    frame, and so does this by default. For treadmill data the lab frame is NOT
    the belt frame — the stance foot is dragged backwards at belt speed, adding
    a large F . v_belt term that overground data does not have. Pass
    `surface_velocity` (a [3] or [T,3] lab-frame velocity of the contact
    surface, e.g. the belt) to evaluate the power in the surface's frame, which
    is the overground-equivalent quantity the literature reports. Nothing here
    guesses the belt direction; `AnalysisParams.treadmill_speed` is a bare
    scalar and is deliberately not used.

    SINGLE-SEGMENT FOOT, DELIBERATELY. This markerset (CAL, 5TH, malleoli)
    fixes only the heel, the 5th metatarsal head and the ankle: it defines one
    rigid foot and nothing more. Splitting the foot at the MTP joint into
    hindfoot and forefoot — as in Honert & Zelik's multi-segment foot work —
    needs forefoot and hallux markers we do not have, so no MTP joint and no
    hindfoot/forefoot power split is offered here rather than inventing a
    convention. The UD power below therefore lumps the shoe, heel pad, arch and
    all foot joints together into one distal-to-foot term.
    """
    if params is None:
        params = AnalysisParams()
    force = np.asarray(ground.force, dtype=float)
    moment = np.asarray(ground.moment, dtype=float)
    if params.force_lowpass_hz > 0:
        force = lowpass(force, ground.rate, params.force_lowpass_hz,
                        params.filter_order)
        moment = lowpass(moment, ground.rate, params.force_lowpass_hz,
                         params.filter_order)

    if cop is None:
        cop, free = cop_from_wrench(force, moment, ground.point,
                                    plane_height=plane_height,
                                    min_fz=params.contact_threshold_n)
    else:
        cop = np.asarray(cop, dtype=float)
        # free moment = whatever moment is left once the force is placed at
        # the COP; for a flat plate this is vertical, and only the vertical
        # component is the physical free moment.
        resid = moment - np.cross(cop - np.asarray(ground.point, float), force)
        free = np.zeros_like(force)
        free[:, 2] = resid[:, 2]

    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    com_foot = ck["com"][:, 0]
    v_foot = ck["v_com"][:, 0]
    w_foot = ck["omega"][:, 0]

    bad = ~np.isfinite(cop).all(axis=1)
    cop_safe = np.where(bad[:, None], com_foot, cop)
    v_ecb = v_foot + np.cross(w_foot, cop_safe - com_foot)
    if surface_velocity is not None:
        v_ecb = v_ecb - np.asarray(surface_velocity, dtype=float)
    p = (np.sum(force * v_ecb, axis=1) + np.sum(free * w_foot, axis=1))
    p[bad] = 0.0
    return p
