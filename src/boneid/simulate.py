"""boneid.simulate — synthetic motions with exact physics for testing.

Every generator returns datatypes from boneid.core plus whatever ground truth
the corresponding test needs. Conventions match core.py: chain ordered
distal -> proximal, world z up, sagittal plane x-z (rotations about +y).

Segment local frames: local +z points from the segment's DISTAL end toward
its PROXIMAL end (i.e. "up" the chain in a neutral standing pose), so
    dist_pos = prox_pos + R @ [0, 0, -L]
    com_local = [0, 0, -c*L]   (c = COM fraction measured from proximal end)
"""

from __future__ import annotations

import numpy as np

from .core import (GroundWrench, SegmentKinematics, Skeleton, GRAVITY,
                   angular_velocity, finite_difference, inertia_world)
from .io_v3d import orthonormal_frame


def minjerk(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk 0->1 profile: zero velocity & acceleration at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def rot_y(angle: np.ndarray) -> np.ndarray:
    """Rotation matrices about +y for angles [T] -> [T,3,3]."""
    c, s = np.cos(angle), np.sin(angle)
    r = np.zeros(angle.shape + (3, 3))
    r[..., 0, 0] = c
    r[..., 0, 2] = s
    r[..., 1, 1] = 1.0
    r[..., 2, 0] = -s
    r[..., 2, 2] = c
    return r


def rod_inertia(mass: float, length: float, radius: float) -> np.ndarray:
    """Solid-cylinder inertia about COM, long axis = local z."""
    i_t = mass * (3 * radius**2 + length**2) / 12.0
    i_l = 0.5 * mass * radius**2
    return np.diag([i_t, i_t, i_l])


def minjerk_derivatives(t: np.ndarray, duration: float):
    """(s, ds/dt, d2s/dt2) of the minimum-jerk profile over `duration` —
    the closed-form derivatives the brick tutorials use as analytic truth."""
    u = np.clip(np.asarray(t, float) / duration, 0.0, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    return s, ds, dds


def rot_x(angle: np.ndarray) -> np.ndarray:
    """Rotation matrices about +x for angles [T] -> [T,3,3]."""
    c, s = np.cos(angle), np.sin(angle)
    r = np.zeros(np.shape(angle) + (3, 3))
    r[..., 0, 0] = 1.0
    r[..., 1, 1] = c
    r[..., 1, 2] = -s
    r[..., 2, 1] = s
    r[..., 2, 2] = c
    return r


def box_inertia(mass: float, size: np.ndarray) -> np.ndarray:
    """Solid rectangular box inertia about its COM; `size` = (x, y, z) extents."""
    x, y, z = np.asarray(size, float)
    return np.diag(mass / 12.0 * np.array([y**2 + z**2, x**2 + z**2,
                                           x**2 + y**2]))


def box_corners(size: np.ndarray, top_center: bool = True) -> np.ndarray:
    """[8,3] corners of a box in its own frame.  With `top_center` the origin
    is the centre of the +z face (a brick hanging from its proximal end, the
    segment convention of this module); otherwise the box centre."""
    x, y, z = 0.5 * np.asarray(size, float)
    c = np.array([[sx, sy, sz] for sx in (-x, x) for sy in (-y, y)
                  for sz in (-z, z)])
    if top_center:
        c[:, 2] -= z
    return c


# ---------------------------------------------------------------------------
# Synthetic markers: rigid points riding on the simulated segments
# ---------------------------------------------------------------------------

def segment_markers(kin: SegmentKinematics,
                    layout: dict[str, tuple[int, np.ndarray]]
                    ) -> dict[str, np.ndarray]:
    """Marker trajectories [T,3] for points fixed in segment frames.

    `layout` maps a marker name to (segment index, position in that segment's
    frame relative to its proximal end).  This is the only place the tutorials
    "attach" a marker: the marker moves rigidly with its segment, exactly as a
    skin marker would if skin did not move.
    """
    out = {}
    for name, (seg, local) in layout.items():
        local = np.asarray(local, float)
        out[name] = (kin.prox_pos[:, seg]
                     + np.einsum("tij,j->ti", kin.r_world[:, seg], local))
    return out


def tetrahedral_corners(size: np.ndarray) -> np.ndarray:
    """Four of a box's eight corners that do NOT lie in one plane — the
    smallest cluster that pins down a rigid pose without ambiguity."""
    x, y, z = 0.5 * np.asarray(size, float)
    return np.array([[x, y, 0.0], [x, -y, -2 * z],
                     [-x, y, -2 * z], [-x, -y, 0.0]])


def _ground_wrench_from_motion(skeleton: Skeleton, kin: SegmentKinematics,
                               point: np.ndarray,
                               gravity: np.ndarray = GRAVITY) -> GroundWrench:
    """Exact external wrench required by whole-body Newton-Euler.

    F = sum m_i (a_i - g)
    M_about_point = dH_point/dt - sum (r_i - point) x m_i g
    with H_point = sum [ I_w_i w_i + (r_i - point) x m_i v_i ].
    Uses the same central differences the analysis uses, so a subsequent
    inverse-dynamics pass closes to numerical precision.
    """
    rate = kin.rate
    com = kin.prox_pos + np.einsum("tsij,sj->tsi", kin.r_world,
                                   skeleton.com_local)
    v = finite_difference(com, rate)
    a = finite_difference(v, rate)
    t_n, s_n = com.shape[:2]
    m = skeleton.mass

    force = np.einsum("s,tsi->ti", m, a) - m.sum() * gravity

    # angular momentum about `point`
    h = np.zeros((t_n, 3))
    omega = angular_velocity(kin.r_world, rate)
    for s in range(s_n):
        i_w = inertia_world(kin.r_world[:, s], skeleton.inertia_local[s])
        h += np.einsum("tij,tj->ti", i_w, omega[:, s])
        h += np.cross(com[:, s] - point, m[s] * v[:, s])
    dh = finite_difference(h, rate)
    grav_moment = np.einsum(
        "tsi->ti", np.cross(com - point, m[:, None] * gravity))
    moment = dh - grav_moment
    return GroundWrench(t=kin.t, rate=rate, force=force, moment=moment,
                        point=np.asarray(point, float))


# Squat-to-stand "anthropometrics": the simulator's own regression table.
# Rows: segment -> (mass fraction, COM fraction from the proximal end,
# cylinder radius).  Lengths are measured from the pose, as in a real study.
S2S_TABLE = {
    "foot":  (0.03, 0.50, 0.04),
    "shank": (0.06, 0.45, 0.05),
    "thigh": (0.12, 0.45, 0.07),
    "torso": (0.60, 0.55, 0.16),   # big torso
}
S2S_LENGTHS = np.array([0.20, 0.43, 0.45, 0.60])   # foot shank thigh torso


def s2s_skeleton(body_mass: float, lengths: np.ndarray) -> Skeleton:
    """The squat-to-stand Skeleton from body mass and measured segment
    lengths, via `S2S_TABLE` — the same shape of call as
    `io_v3d.deleva_skeleton`, with the simulator's table in place of de Leva.

    Frames follow the io_v3d convention: local z is the long axis, pointing
    distal -> proximal for shank/thigh/torso and ankle -> toe for the foot;
    local y is the subject's left.  So the COM sits at -c*L along z for the
    leg/torso segments and at +c*L along z for the foot (measured from the
    ankle toward the toe: this simulated foot has no heel).
    """
    names = list(S2S_TABLE)
    lengths = np.asarray(lengths, float)
    mass = np.array([S2S_TABLE[n][0] * body_mass for n in names])
    com = np.zeros((4, 3))
    inertia = np.zeros((4, 3, 3))
    for i, n in enumerate(names):
        _, c, r = S2S_TABLE[n]
        com[i, 2] = (c if n == "foot" else -c) * lengths[i]
        inertia[i] = rod_inertia(mass[i], lengths[i], r)
    return Skeleton(segment_names=names, joint_names=["ankle", "knee", "hip"],
                    mass=mass, com_local=com, inertia_local=inertia,
                    length=lengths.copy())


def simulate_squat_to_stand(rate: float = 500.0, duration: float = 2.0,
                            body_mass: float = 80.0,
                            torso_sway: float = 0.0):
    """TEST-SIMULATED-S2S: 3D 3-joint chain (foot, shank, thigh, torso;
    ankle-knee-hip) with a large torso, rising from deep squat to standing.

    The foot is welded to the ground; joint angles follow minimum-jerk
    profiles (so velocities and accelerations vanish at both endpoints,
    giving analytic static limits). The ground wrench is computed EXACTLY
    from whole-body dynamics — no force model, no noise.

    `torso_sway` (rad) adds a lateral lean of the torso about its own
    anterior axis, shaped as a bump that is zero — with zero velocity — at
    both ends, so the static limits still hold.  With it the motion is
    genuinely three-dimensional: the torso's angular velocity leaves its
    principal axes, which is what the legacy inertia/gyroscopic errors need
    in order to show up at all (planar motion forgives them exactly).

    Returns (skeleton, kin, ground, truth) where truth holds the segment
    angle trajectories and the static joint torques at t=0 and t=end.
    """
    t = np.arange(int(round(rate * duration))) / rate
    s = minjerk(t / duration)

    lengths = S2S_LENGTHS
    skeleton = s2s_skeleton(body_mass, lengths)
    masses = skeleton.mass

    # segment tilt from vertical (about +y, positive = leaning +x)
    tilt_start = np.array([0.0, 0.7, -1.9, 1.15])  # deep squat
    tilt_end = np.array([0.0, 0.03, -0.03, 0.02])  # standing
    tilt = tilt_start[None, :] + s[:, None] * (tilt_end - tilt_start)[None, :]

    r_world = rot_y(tilt)                          # [T,4,3,3]
    if torso_sway:
        bump = (4.0 * s * (1.0 - s)) ** 2          # 0 at both ends, flat there
        r_world[:, 3] = r_world[:, 3] @ rot_x(torso_sway * bump)

    ankle = np.array([0.05, 0.0, 0.08])
    up = np.array([0.0, 0.0, 1.0])
    t_n = len(t)
    prox = np.zeros((t_n, 4, 3))
    dist = np.zeros((t_n, 4, 3))
    # foot: static, proximal end at the ankle, distal end at the toe on the
    # ground just in front of it; its frame follows the io_v3d convention
    # (local z along ankle -> toe, local y = subject's left)
    prox[:, 0] = ankle
    toe_dir = np.array([lengths[0], 0.0, -ankle[2] + 0.015])
    dist[:, 0] = ankle + lengths[0] * toe_dir / np.linalg.norm(toe_dir)
    r_world[:, 0] = orthonormal_frame(dist[:, 0] - prox[:, 0],
                                      np.array([0.0, 1.0, 0.0]))
    p = np.broadcast_to(ankle, (t_n, 3)).copy()
    for seg in range(1, 4):
        axis = np.einsum("tij,j->ti", r_world[:, seg], up)
        dist[:, seg] = p
        p = p + lengths[seg] * axis
        prox[:, seg] = p

    kin = SegmentKinematics(t=t, rate=rate, r_world=r_world,
                            prox_pos=prox, dist_pos=dist)
    ground = _ground_wrench_from_motion(skeleton, kin, np.zeros(3))

    # analytic static joint torques at both endpoints (zero vel/acc there).
    # Convention: torque ON the distal segment BY the proximal one, so the
    # static value is +sum_{i>j} (com_i - p_j) x m_i g (it holds up the
    # weight of everything above the joint).
    def static_torques(frame):
        com = prox[frame] + np.einsum("sij,sj->si", r_world[frame],
                                      skeleton.com_local)
        out = {}
        for j, jn in enumerate(skeleton.joint_names):
            p_j = prox[frame, j]
            m_sum = np.zeros(3)
            for i in range(j + 1, 4):
                m_sum += np.cross(com[i] - p_j, masses[i] * GRAVITY)
            out[jn] = m_sum
        return out

    truth = {"tilt": tilt, "static_torque_start": static_torques(0),
             "static_torque_end": static_torques(t_n - 1)}
    return skeleton, kin, ground, truth


def s2s_marker_layout(radius_scale: float = 1.0) -> dict:
    """Where the synthetic markers sit on the squat-to-stand body.

    Mirrors the walking markerset the quickstart reads: a medial/lateral
    pair at every joint (rigid in the segment ABOVE the joint, as malleoli
    ride the shank and epicondyles the thigh), a toe marker on the foot, and
    a shoulder pair at the top of the torso standing in for the acromions.
    Returns the `segment_markers` layout: name -> (segment, local position).
    """
    r = {n: S2S_TABLE[n][2] * radius_scale for n in S2S_TABLE}
    L = dict(zip(S2S_TABLE, S2S_LENGTHS))
    lat, med = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    return {
        "toe": (0, np.array([0.0, 0.0, L["foot"]])),
        "ankle_lat": (1, r["shank"] * lat - np.array([0, 0, L["shank"]])),
        "ankle_med": (1, r["shank"] * med - np.array([0, 0, L["shank"]])),
        "knee_lat": (2, r["thigh"] * lat - np.array([0, 0, L["thigh"]])),
        "knee_med": (2, r["thigh"] * med - np.array([0, 0, L["thigh"]])),
        "hip_lat": (3, r["torso"] * lat - np.array([0, 0, L["torso"]])),
        "hip_med": (3, r["torso"] * med - np.array([0, 0, L["torso"]])),
        "shoulder_r": (3, r["torso"] * lat),
        "shoulder_l": (3, r["torso"] * med),
    }


# ---------------------------------------------------------------------------
# Squat-to-stand as a c3d RECORDING, for the real (Asics) pipeline
# ---------------------------------------------------------------------------
#
# The squat-to-stand above hands core.py its kinematics directly.  This one
# instead writes what a lab would hand you: a static c3d and a dynamic c3d
# in the Asics markerset, with a force plate, so the analysis runs through
# the exact calls a real subject does (load_c3d, calibrate_static,
# io_c3d.build_chain, core.inverse_dynamics, core.energy_audit).
#
# The Asics study marked nothing above the pelvis.  This idealized subject
# also wears torso markers (io_c3d.TORSO_CLUSTER + acromions), so the chain
# runs foot -> shank -> thigh -> pelvis -> torso and L5S1 is a real joint.
# The body is exactly the de Leva segments of a `body_mass` person that the
# pipeline will build: right leg, pelvis, torso (trunk + head).  No left leg,
# no arms -- so the plate weighs ~70% of `body_mass`, and the analysis must
# be given `body_mass` (as from a scale).  Nothing acts above the torso, so
# the true residual is ZERO.  Ideal joints, rigid markers, no noise.

#: native frame of the construction: x forward, y subject-left, z up.
#: Q maps it to the Asics lab (+x subject-left, +y posterior, +z up).
_ASICS_FROM_NATIVE = np.array([[0.0, 1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0]])


def _wrench_to_move(bodies, point: np.ndarray, rate: float,
                    gravity: np.ndarray = GRAVITY):
    """(F [T,3], M [T,3] about `point` [T,3]) that must act on a set of
    rigid bodies for them to follow their motion: F = sum m (a - g),
    M = sum [I_w alpha + w x I_w w + (c - p) x m (a - g)].

    `bodies` is a list of (mass, com [T,3], r_world [T,3,3], I_local [3,3]).
    Valid for a moving point.  This is the TOP-DOWN route: it never sees the
    pipeline's bottom-up recursion, so the two can check each other.
    """
    f = np.zeros_like(point)
    m_out = np.zeros_like(point)
    for mass, com, r, i_local in bodies:
        a = finite_difference(finite_difference(com, rate), rate)
        w = angular_velocity(r, rate)
        alpha = finite_difference(w, rate)
        i_w = inertia_world(r, i_local)
        lin = mass * (a - gravity)
        f = f + lin
        m_out = (m_out + np.einsum("tij,tj->ti", i_w, alpha)
                 + np.cross(w, np.einsum("tij,tj->ti", i_w, w))
                 + np.cross(com - point, lin))
    return f, m_out


def simulate_squat_to_stand_c3d(out_dir, subject: str = "SIM",
                                condition: str = "S2S", trial: str = "RISE",
                                body_mass: float = 80.0,
                                torso_sway: float = 0.25,
                                knee_slip_m: float = 0.0,
                                marker_noise_m: float = 0.0,
                                rate: float = 300.0,
                                analog_rate: float = 3000.0,
                                hold_s: float = 0.5, rise_s: float = 2.0,
                                static_s: float = 1.0, sex: str = "male",
                                seed: int = 0) -> dict:
    """Write ``{subject}_{condition}_Static.c3d`` and
    ``{subject}_{condition}_{trial}.c3d`` into `out_dir` and return the truth.

    One right leg, a pelvis and a torso rise from a deep squat to standing
    (smooth tilt profiles flat to the 4th derivative, a `hold_s` hold at each
    end) with the foot flat on one plate.  With `torso_sway` the torso also
    leans sideways and back, which makes the motion 3-D.  `knee_slip_m`
    makes the knee a non-ideal joint: the femur slides forward on the tibia
    by up to that much mid-rise (zero, and flat, at both ends), so the
    joint does translational work -- the 6-DoF test.  Medial ankle/knee
    markers exist only in the static trial, as in the Asics study.
    Everything is built at `analog_rate`; the markers are every
    `ratio`-th sample, so plate and markers share one clock.

    Returns a dict: paths, ``t`` [T], ``skeleton`` (the true de Leva body the
    pipeline should rebuild), ``body_mass`` (what to tell the pipeline),
    ``plate_mass`` (what the plate will say), and TOP-DOWN truth on the
    mocap clock: ``joint_force``/``joint_torque`` [T,4,3] for ankle, knee,
    hip, l5s1 (on the distal segment by the proximal one, about the joint,
    as core.inverse_dynamics reports) and ``residual_force``/
    ``residual_torque`` [T,3] (zero: nothing acts above the torso), and
    ``joint_power_rotational``/``joint_power_translational`` [T,4]
    (tau . (w_dist - w_prox) and F . (v_dist(p) - v_prox(p)) at the joint
    point p, from the true motion; their sum is the 6-DoF joint power).
    """
    import os
    from .io_c3d import write_c3d
    from .io_v3d import append_deleva_torso, deleva_skeleton

    q = _ASICS_FROM_NATIVE
    ratio = int(round(analog_rate / rate))
    total = 2 * hold_s + rise_s
    n_a = ratio * int(round(total * rate))
    t_a = np.arange(n_a) / analog_rate
    # a smoothstep flat to the 4th derivative at both ends (minimum jerk is
    # flat only to the 2nd): the ground force then has no kink where the
    # motion starts and stops, so 300 Hz sampling does not blur one.
    u = np.clip((t_a - hold_s) / rise_s, 0.0, 1.0)
    s = u**5 * (126 - 420 * u + 540 * u**2 - 315 * u**3 + 70 * u**4)

    # --- geometry in the standing pose (native frame, metres) -------------
    leg_y = -0.064                         # right hip = 0.7 RGT + 0.3 LGT
    gt_half = 0.16                         # half the RGT-LGT distance
    ankle = np.array([0.0, leg_y, 0.08])
    foot_markers = {                       # the foot is static: world coords
        "HeelLat": [-0.06, leg_y - 0.03, 0.03],
        "HeelMed": [-0.06, leg_y + 0.03, 0.03],
        "DorsalFoot": [0.06, leg_y, 0.075],
        "MTP1Head": [0.13, leg_y + 0.035, 0.02],
        "MTP5Head": [0.13, leg_y - 0.035, 0.02],
        "ToeboxAnt": [0.20, leg_y, 0.03],
        "AnkleLat": [0.0, leg_y - 0.035, 0.08],
        "AnkleMed": [0.0, leg_y + 0.035, 0.08],
    }
    foot_markers = {k: np.array(v) for k, v in foot_markers.items()}
    l_shank, l_thigh, l_pelvis, l_torso = 0.43, 0.45, 0.11, 0.50
    # segment-local marker positions, origin at the segment's proximal joint
    shank_local = {"ShankAntProx": [0.05, 0.0, -0.10],
                   "ShankPostProx": [-0.05, -0.01, -0.12],
                   "ShankPostDist": [-0.04, 0.0, -0.30],
                   "KneeLat": [0.0, -0.05, 0.0], "KneeMed": [0.0, 0.05, 0.0]}
    thigh_local = {"ThighAntProx": [0.07, 0.0, -0.12],
                   "ThighPostProx": [-0.07, -0.02, -0.15],
                   "ThighPostDist": [-0.06, 0.01, -0.33]}
    pelvis_local = {"RASIS": [0.07, -0.12, 0.0], "LASIS": [0.07, 0.12, 0.0],
                    "RPSIS": [-0.07, -0.05, 0.0], "LPSIS": [-0.07, 0.05, 0.0],
                    "RGT": [0.0, -gt_half, -l_pelvis],
                    "LGT": [0.0, gt_half, -l_pelvis]}
    # torso origin = mid-acromion (its proximal end in the serial chain)
    torso_local = {"RAC": [0.0, -0.18, 0.0], "LAC": [0.0, 0.18, 0.0],
                   "C7": [-0.08, 0.0, 0.03], "CLAV": [0.07, 0.0, -0.02],
                   "STRN": [0.10, 0.0, -0.18], "T10": [-0.11, 0.0, -0.26]}
    static_only = {"AnkleMed", "KneeMed"}
    hip_in_pelvis = np.array([0.0, leg_y, -l_pelvis])   # 0.7 RGT + 0.3 LGT

    heel = 0.5 * (foot_markers["HeelLat"] + foot_markers["HeelMed"])
    toe = 0.5 * (foot_markers["MTP1Head"] + foot_markers["MTP5Head"])
    r_foot = orthonormal_frame(toe - heel, foot_markers["AnkleMed"]
                               - foot_markers["AnkleLat"])

    # the true body: de Leva on these lengths (what calibration must measure)
    skeleton = append_deleva_torso(deleva_skeleton(
        body_mass=body_mass,
        foot_length=float(np.linalg.norm(foot_markers["ToeboxAnt"] - heel)),
        shank_length=l_shank, thigh_length=l_thigh, pelvis_length=l_pelvis,
        foot_heel_local=r_foot.T @ (heel - ankle), sex=sex),
        body_mass, l_torso, sex=sex)

    # --- motion (native) ---------------------------------------------------
    tilt_start = np.array([0.75, -1.75, 0.55, 0.95])  # shank thigh pelvis torso
    tilt_end = np.array([0.03, -0.03, 0.02, 0.01])

    def pose(sv, sway_scale):
        tilt = tilt_start + sv[:, None] * (tilt_end - tilt_start)
        r_sh, r_th, r_pe, r_to = (rot_y(tilt[:, i]) for i in range(4))
        bump = (4.0 * sv * (1.0 - sv)) ** 2       # 0, flat, at both ends
        sway = sway_scale * torso_sway * bump
        r_pe = r_pe @ rot_x(0.3 * sway)
        r_to = r_to @ rot_x(sway) @ rot_y(-0.5 * sway)
        n = len(sv)
        up = np.array([0.0, 0.0, 1.0])
        knee = ankle + l_shank * np.einsum("tij,j->ti", r_sh, up)
        # the femur's end of the knee, slid forward along the tibial plateau
        slip = (knee_slip_m * sway_scale * bump)[:, None] * r_sh[:, :, 0]
        hip = knee + slip + l_thigh * np.einsum("tij,j->ti", r_th, up)
        l5s1 = hip - np.einsum("tij,j->ti", r_pe, hip_in_pelvis)
        acromion = l5s1 + l_torso * np.einsum("tij,j->ti", r_to, up)
        r = np.stack([np.broadcast_to(r_foot, (n, 3, 3)), r_sh, r_th, r_pe,
                      r_to], axis=1)
        prox = np.stack([np.broadcast_to(ankle, (n, 3)), knee, hip, l5s1,
                         acromion], axis=1)
        return r, prox

    def to_lab(r, prox):
        return (np.einsum("ij,...jk->...ik", q, r),
                np.einsum("ij,...j->...i", q, prox))

    def markers_of(r, prox, static):
        out = {}
        for name, x in foot_markers.items():
            out[name] = np.broadcast_to(x, (len(r), 3)).copy()
        for seg, table in ((1, shank_local), (2, thigh_local),
                           (3, pelvis_local), (4, torso_local)):
            for name, loc in table.items():
                out[name] = prox[:, seg] + np.einsum("tij,j->ti", r[:, seg],
                                                     np.array(loc, float))
        out = {k: np.einsum("ij,tj->ti", q, v) for k, v in out.items()}
        if not static:
            out = {k: v for k, v in out.items() if k not in static_only}
        return out

    def bodies_of(r_lab, prox_lab, segs):
        return [(skeleton.mass[i],
                 prox_lab[:, i] + r_lab[:, i] @ skeleton.com_local[i],
                 r_lab[:, i], skeleton.inertia_local[i]) for i in segs]

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_seg = len(skeleton.mass)
    paths = {}
    truth = {}
    for kind in ("static", "dynamic"):
        if kind == "static":
            r, prox = pose(np.ones(ratio * int(round(static_s * rate))), 0.0)
        else:
            r, prox = pose(s, 1.0)
        r_lab, prox_lab = to_lab(r, prox)
        f_g, m_g = _wrench_to_move(bodies_of(r_lab, prox_lab, range(n_seg)),
                                   np.zeros((len(r), 3)), analog_rate)
        mk = markers_of(r[::ratio], prox[::ratio], kind == "static")
        if marker_noise_m > 0:
            mk = {k: v + rng.normal(0.0, marker_noise_m, v.shape)
                  for k, v in mk.items()}
        name = "Static" if kind == "static" else trial
        path = os.path.join(out_dir, f"{subject}_{condition}_{name}.c3d")
        write_c3d(path, mk, rate, analog_rate, f_g, m_g)
        paths[kind] = path
        if kind == "dynamic":
            # top-down truth: each joint carries everything above it
            j_f, j_m = [], []
            for j in range(n_seg - 1):
                f, m = _wrench_to_move(
                    bodies_of(r_lab, prox_lab, range(j + 1, n_seg)),
                    prox_lab[:, j], analog_rate)
                j_f.append(-f)
                j_m.append(-m)
            # true joint powers, from the true motion at the analog rate
            bodies = bodies_of(r_lab, prox_lab, range(n_seg))
            vel = []
            for mass, com, rr, _ in bodies:
                vel.append((finite_difference(com, analog_rate), com,
                            angular_velocity(rr, analog_rate)))
            p_rot, p_tr = [], []
            for j in range(n_seg - 1):
                p = prox_lab[:, j]
                (v_d, c_d, w_d), (v_p, c_p, w_p) = vel[j], vel[j + 1]
                on_d = v_d + np.cross(w_d, p - c_d)
                on_p = v_p + np.cross(w_p, p - c_p)
                p_rot.append(np.sum(j_m[j] * (w_d - w_p), axis=1))
                p_tr.append(np.sum(j_f[j] * (on_d - on_p), axis=1))
            truth = {
                "joint_power_rotational": np.stack(p_rot, axis=1)[::ratio],
                "joint_power_translational": np.stack(p_tr, axis=1)[::ratio],
                "joint_force": np.stack(j_f, axis=1)[::ratio],
                "joint_torque": np.stack(j_m, axis=1)[::ratio],
                "residual_force": np.zeros((len(r) // ratio, 3)),
                "residual_torque": np.zeros((len(r) // ratio, 3)),
                "ground_force": f_g[::ratio],
            }
    n_t = len(truth["joint_force"])
    return {"static_path": paths["static"], "dynamic_path": paths["dynamic"],
            "t": np.arange(n_t) / rate, "skeleton": skeleton,
            "body_mass": body_mass, "plate_mass": float(skeleton.mass.sum()),
            **truth}


# ---------------------------------------------------------------------------
# Bricks: the two planar tutorials (2-D is 3-D with the y-components zero)
# ---------------------------------------------------------------------------

BRICK_SIZE = np.array([0.15, 0.10, 0.30])     # x (thickness), y (width), z (long)


def simulate_brick_2d(rate: float = 500.0, duration: float = 2.0,
                      mass: float = 2.0, size: np.ndarray = BRICK_SIZE):
    """One brick carried by an unseen hand: lifted 0.4 m and 0.5 m forward
    along a minimum-jerk path while tilting sinusoidally about +y.

    The brick is a one-segment chain whose proximal end is the hand's grip
    point (the centre of the top face).  Nothing else touches it, so the
    GroundWrench is zero and `inverse_dynamics` reports the hand's wrench as
    the RESIDUAL.  Returns (skeleton, kin, markers, truth) with four
    non-coplanar corner markers and the analytic hand wrench in `truth`
    (`force` [T,3] N, `torque` [T,3] N m about the grip point).
    """
    t = np.arange(int(round(rate * duration))) / rate
    size = np.asarray(size, float)
    L = size[2]
    s, ds, dds = minjerk_derivatives(t, duration)
    travel = np.array([0.5, 0.0, 0.4])
    com0 = np.array([0.0, 0.0, 0.6])
    com = com0 + s[:, None] * travel
    a_com = dds[:, None] * travel
    w = 2 * np.pi / duration
    theta = 0.6 * np.sin(w * t)
    dtheta = 0.6 * w * np.cos(w * t)
    ddtheta = -0.6 * w**2 * np.sin(w * t)

    inertia = box_inertia(mass, size)
    skeleton = Skeleton(segment_names=["brick"], joint_names=[],
                        mass=np.array([mass]),
                        com_local=np.array([[0.0, 0.0, -0.5 * L]]),
                        inertia_local=inertia[None], length=np.array([L]))
    r = rot_y(theta)
    up = np.einsum("tij,j->ti", r, np.array([0.0, 0.0, 1.0]))
    prox = com + 0.5 * L * up
    dist = com - 0.5 * L * up
    kin = SegmentKinematics(t=t, rate=rate, r_world=r[:, None],
                            prox_pos=prox[:, None], dist_pos=dist[:, None])
    corners = tetrahedral_corners(size)
    markers = segment_markers(kin, {f"c{k + 1}": (0, corners[k])
                                    for k in range(4)})

    # analytic hand wrench: F = m (a - g);  M_grip = I_yy θ'' ŷ + (com - grip) x F
    force = mass * (a_com - GRAVITY)
    torque = np.zeros_like(force)
    torque[:, 1] = inertia[1, 1] * ddtheta
    torque += np.cross(com - prox, force)
    truth = {"force": force, "torque": torque, "theta": theta,
             "omega_y": dtheta, "alpha_y": ddtheta, "a_com": a_com}
    return skeleton, kin, markers, truth


def simulate_two_bricks_2d(rate: float = 500.0, duration: float = 2.0):
    """Two hinged bricks: the lower one is bolted to the floor, the upper one
    swings about the hinge on top of it.

    Chain order distal -> proximal: [lower, upper]; the one joint is the
    hinge at the lower brick's top centre.  The ground wrench about the lab
    origin is computed EXACTLY from the motion — for a bolted base that
    wrench legitimately includes a moment, and the point where its line of
    action meets the floor (the COP) can leave the brick's footprint, which
    is exactly when a free-standing brick would have tipped.  Returns
    (skeleton, kin, ground, markers, hinge_local) where `hinge_local` gives
    the hinge in each brick's own frame (the "ruler measurement" the tutorial
    carries as a virtual point) and `markers` holds four corner markers per
    brick.
    """
    t = np.arange(int(round(rate * duration))) / rate
    size_a = np.array([0.15, 0.15, 0.35])
    size_b = np.array([0.12, 0.12, 0.45])
    m_a, m_b = 4.0, 3.0
    L_a, L_b = size_a[2], size_b[2]

    phi = (0.7 * np.sin(2 * np.pi * t / duration)
           + 0.2 * np.sin(4 * np.pi * t / duration))              # the swing
    t_n = len(t)
    r_a = np.broadcast_to(np.eye(3), (t_n, 3, 3)).copy()          # bolted down
    r_b = rot_y(phi)

    prox_a = np.broadcast_to([0.0, 0.0, L_a], (t_n, 3)).copy()    # hinge
    dist_a = np.zeros((t_n, 3))                                    # on the floor
    prox_b = prox_a + np.einsum("tij,j->ti", r_b, np.array([0, 0, L_b]))
    dist_b = prox_a.copy()

    skeleton = Skeleton(
        segment_names=["lower", "upper"], joint_names=["hinge"],
        mass=np.array([m_a, m_b]),
        com_local=np.array([[0, 0, -0.5 * L_a], [0, 0, -0.5 * L_b]]),
        inertia_local=np.stack([box_inertia(m_a, size_a),
                                box_inertia(m_b, size_b)]),
        length=np.array([L_a, L_b]))
    kin = SegmentKinematics(t=t, rate=rate,
                            r_world=np.stack([r_a, r_b], axis=1),
                            prox_pos=np.stack([prox_a, prox_b], axis=1),
                            dist_pos=np.stack([dist_a, dist_b], axis=1))
    ground = _ground_wrench_from_motion(skeleton, kin, np.zeros(3))
    layout = {}
    for seg, size, tag in ((0, size_a, "a"), (1, size_b, "b")):
        for k, c in enumerate(tetrahedral_corners(size)):
            layout[f"{tag}{k + 1}"] = (seg, c)
    markers = segment_markers(kin, layout)
    hinge_local = {"lower": np.zeros(3),                    # its own origin
                   "upper": np.array([0.0, 0.0, -L_b])}     # its bottom
    return skeleton, kin, ground, markers, hinge_local


def simulate_pendulum(rate: float = 1000.0, duration: float = 1.0,
                      mass: float = 2.0, length: float = 0.8,
                      radius: float = 0.03):
    """Single rigid pendulum pinned at a fixed point, prescribed swing about
    +y. Chain of one segment: inverse dynamics reports the pivot wrench as
    the RESIDUAL. Returns (skeleton, kin, ground(zero), truth_torque [T,3])
    with the analytic pivot torque I_pivot*acc + m g d sin(theta)."""
    t = np.arange(int(round(rate * duration))) / rate
    theta = 0.6 * np.sin(2 * np.pi * 1.5 * t)          # tilt from vertical
    dtheta = 0.6 * 2 * np.pi * 1.5 * np.cos(2 * np.pi * 1.5 * t)
    ddtheta = -0.6 * (2 * np.pi * 1.5) ** 2 * np.sin(2 * np.pi * 1.5 * t)

    pivot = np.array([0.0, 0.0, 1.5])
    d = 0.5 * length                                    # COM from pivot
    skeleton = Skeleton(
        segment_names=["rod"], joint_names=[],
        mass=np.array([mass]),
        com_local=np.array([[0.0, 0.0, -d]]),
        inertia_local=rod_inertia(mass, length, radius)[None],
        length=np.array([length]),
    )
    r = rot_y(theta)[:, None]                           # [T,1,3,3]
    t_n = len(t)
    prox = np.broadcast_to(pivot, (t_n, 1, 3)).copy()
    up = np.array([0, 0, 1.0])
    dist = prox - length * np.einsum("tij,j->ti", r[:, 0], up)[:, None]
    kin = SegmentKinematics(t=t, rate=rate, r_world=r,
                            prox_pos=prox, dist_pos=dist)
    ground = GroundWrench(t=t, rate=rate, force=np.zeros((t_n, 3)),
                          moment=np.zeros((t_n, 3)), point=np.zeros(3))

    i_com = rod_inertia(mass, length, radius)[1, 1]
    i_pivot = i_com + mass * d**2
    tau_y = i_pivot * ddtheta + mass * 9.81 * d * np.sin(theta)
    truth_torque = np.zeros((t_n, 3))
    truth_torque[:, 1] = tau_y
    return skeleton, kin, ground, truth_torque


def simulate_gyro(rate: float = 1000.0, duration: float = 1.0,
                  mass: float = 3.0):
    """Free rigid body spinning at constant WORLD angular velocity about a
    non-principal axis, COM fixed, gravity compensated externally.

    alpha = 0, yet the required torque is omega x (I_w omega) != 0 — this is
    exactly the term the legacy code dropped, and I_w = R I R^T is exactly
    the transform it botched. Returns (skeleton, kin, ground, truth) where
    truth[t] = omega x (I_w[t] omega) + com x m*(-g)-compensating force... —
    here we simply supply gravity-cancelling ground force through the COM so
    truth torque = omega x (I_w omega)."""
    t = np.arange(int(round(rate * duration))) / rate
    inertia = np.diag([0.4, 0.2, 0.1])
    omega = np.array([3.0, 0.0, 2.0])                  # non-principal axis
    w_norm = np.linalg.norm(omega)
    axis = omega / w_norm

    # Rodrigues rotation about fixed world axis
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    ang = w_norm * t
    r = (np.eye(3)[None] + np.sin(ang)[:, None, None] * k
         + (1 - np.cos(ang))[:, None, None] * (k @ k))

    com_point = np.array([0.0, 0.0, 1.0])
    skeleton = Skeleton(
        segment_names=["body"], joint_names=[],
        mass=np.array([mass]),
        com_local=np.array([[0.0, 0.0, 0.0]]),        # prox point AT the COM
        inertia_local=inertia[None],
        length=np.array([0.3]),
    )
    t_n = len(t)
    prox = np.broadcast_to(com_point, (t_n, 1, 3)).copy()
    dist = prox.copy()
    kin = SegmentKinematics(t=t, rate=rate, r_world=r[:, None],
                            prox_pos=prox, dist_pos=dist)
    # ground force cancels gravity, acting exactly through the COM
    force = np.tile(-mass * GRAVITY, (t_n, 1))
    moment = np.cross(np.broadcast_to(com_point, (t_n, 3)), force)  # about O
    ground = GroundWrench(t=t, rate=rate, force=force, moment=moment,
                          point=np.zeros(3))

    i_w = inertia_world(r, inertia)
    truth = np.cross(omega[None], np.einsum("tij,j->ti", i_w, omega))
    return skeleton, kin, ground, truth


# ---------------------------------------------------------------------------
# Counterfactual kinematics: a knee pushed laterally (varus), hip/ankle fixed
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[T,3,3] smallest rotation taking unit vectors a [T,3] onto b [T,3]."""
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    vx = np.zeros(a.shape[:-1] + (3, 3))
    vx[..., 0, 1], vx[..., 0, 2] = -v[..., 2], v[..., 1]
    vx[..., 1, 0], vx[..., 1, 2] = v[..., 2], -v[..., 0]
    vx[..., 2, 0], vx[..., 2, 1] = -v[..., 1], v[..., 0]
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))[..., None, None]


def varus_rotations(kin: SegmentKinematics, d: float):
    """(knee_new [T,3], r_shank [T,3,3], r_thigh [T,3,3]) for a right-leg
    chain [foot, shank, thigh, ...] whose knee centre is moved `d` metres
    laterally, perpendicular to the hip-ankle line (lateral = minus the
    thigh's subject-left axis, with the hip-ankle component removed).  The
    shank turns about the ankle and the thigh about the hip by the smallest
    rotations that reach the new knee; hip, ankle and foot stay put.

    The knee moves square to the hip-ankle line, not on the spheres about
    hip and ankle, so the joint-centre distances are NOT kept: on P1 the
    shank and thigh lengthen by ~0.6 / 2 / 5.5 mm on average (max ~8 mm, ~2%)
    at d = 10 / 25 / 50 mm.  (The OpenSim sweep, with rigid model segments,
    has no stretch; the knee position — what sets the KAM — is the same.)
    A few frames shorten slightly instead."""
    knee, hip, ankle = kin.prox_pos[:, 1], kin.prox_pos[:, 2], kin.prox_pos[:, 0]
    a = _unit(hip - ankle)
    lat = -kin.r_world[:, 2, :, 1]
    lat = _unit(lat - a * np.sum(lat * a, axis=-1, keepdims=True))
    knee_new = knee + d * lat
    r_shank = min_rotation(_unit(knee - ankle), _unit(knee_new - ankle))
    r_thigh = min_rotation(_unit(hip - knee), _unit(hip - knee_new))
    return knee_new, r_shank, r_thigh


def varus_shift(kin: SegmentKinematics, d: float) -> SegmentKinematics:
    """`kin` with the knee pushed `d` metres laterally (`varus_rotations`):
    shank and thigh re-posed, every other segment untouched.  Used with the
    MEASURED ground force, it is "the same walk on a bowed leg"."""
    knee_new, r_shank, r_thigh = varus_rotations(kin, d)
    r = kin.r_world.copy()
    r[:, 1] = r_shank @ kin.r_world[:, 1]
    r[:, 2] = r_thigh @ kin.r_world[:, 2]
    prox, dist = kin.prox_pos.copy(), kin.dist_pos.copy()
    prox[:, 1] = knee_new
    dist[:, 2] = knee_new
    return SegmentKinematics(t=kin.t, rate=kin.rate, r_world=r, prox_pos=prox,
                             dist_pos=dist)


#: markers that ride the shank / thigh in the varus sweep (P1 markerset)
VARUS_SHANK_MARKERS = ("RSH1", "RSH2", "RSH3", "RLML", "RMML")
VARUS_THIGH_MARKERS = ("RTH1", "RTH2", "RTH3", "RLEP", "RMEP", "RGTR")


def varus_markers(markers: dict, kin: SegmentKinematics, d: float) -> dict:
    """The shank and thigh markers of `markers` carried by `varus_shift`:
    shank ones rotated about the ankle centre, thigh ones about the hip
    centre (the same rotations as the segments).  Only those keys return."""
    _, r_shank, r_thigh = varus_rotations(kin, d)
    out = {}
    for names, r, pivot in ((VARUS_SHANK_MARKERS, r_shank, kin.prox_pos[:, 0]),
                            (VARUS_THIGH_MARKERS, r_thigh, kin.prox_pos[:, 2])):
        for name in names:
            if name in markers:
                rel = markers[name] - pivot
                out[name] = pivot + np.einsum("tij,tj->ti", r, rel)
    return out
