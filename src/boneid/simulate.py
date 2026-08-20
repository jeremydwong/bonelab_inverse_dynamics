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


def simulate_squat_to_stand(rate: float = 500.0, duration: float = 2.0,
                            body_mass: float = 80.0):
    """TEST-SIMULATED-S2S: 3D 3-joint chain (foot, shank, thigh, torso;
    ankle-knee-hip) with a large torso, rising from deep squat to standing.

    The foot is welded to the ground; joint angles follow minimum-jerk
    profiles (so velocities and accelerations vanish at both endpoints,
    giving analytic static limits). The ground wrench is computed EXACTLY
    from whole-body dynamics — no force model, no noise.

    Returns (skeleton, kin, ground, truth) where truth holds the segment
    angle trajectories and the static joint torques at t=0 and t=end.
    """
    t = np.arange(int(round(rate * duration))) / rate
    s = minjerk(t / duration)

    lengths = np.array([0.20, 0.43, 0.45, 0.60])   # foot shank thigh torso
    masses = body_mass * np.array([0.03, 0.06, 0.12, 0.60])  # big torso
    radii = np.array([0.04, 0.05, 0.07, 0.16])
    com_frac = np.array([0.5, 0.45, 0.45, 0.55])   # from proximal end

    names = ["foot", "shank", "thigh", "torso"]
    skeleton = Skeleton(
        segment_names=names,
        joint_names=["ankle", "knee", "hip"],
        mass=masses,
        com_local=np.stack([[0, 0, -c * L]
                            for c, L in zip(com_frac, lengths)]),
        inertia_local=np.stack([rod_inertia(m, L, r) for m, L, r
                                in zip(masses, lengths, radii)]),
        length=lengths,
    )

    # segment tilt from vertical (about +y, positive = leaning +x)
    tilt_start = np.array([0.0, 0.7, -1.9, 1.15])  # deep squat
    tilt_end = np.array([0.0, 0.03, -0.03, 0.02])  # standing
    tilt = tilt_start[None, :] + s[:, None] * (tilt_end - tilt_start)[None, :]

    r_world = rot_y(tilt)                          # [T,4,3,3]

    ankle = np.array([0.05, 0.0, 0.08])
    up = np.array([0.0, 0.0, 1.0])
    t_n = len(t)
    prox = np.zeros((t_n, 4, 3))
    dist = np.zeros((t_n, 4, 3))
    # foot: static, proximal end at the ankle, distal end at the toe on the
    # ground just in front of it
    prox[:, 0] = ankle
    dist[:, 0] = ankle + np.array([lengths[0], 0.0, -ankle[2] + 0.015])
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
