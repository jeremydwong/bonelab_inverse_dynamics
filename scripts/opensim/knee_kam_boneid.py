"""boneid's external knee adduction moment under the varus sweep (cross-check).

Run in the REPO's uv env:  uv run python scripts/opensim/knee_kam_boneid.py

Same geometry as the AFF notebook (notebooks/aff_bowed_femur.ipynb,
`lateral_dir` / `varus_shift` / `kam`): the right knee centre is moved
laterally by d perpendicular to the hip-ankle line, thigh and shank re-posed
by the minimal rotation, hip/ankle/foot/GRF untouched, then boneid's
right-leg inverse dynamics.  Writes data/opensim/varus/kam_boneid.npz:
t [T], d_m [D], kam [D,T] (N m, + = external varus/adduction, thigh frame),
knee_sagittal [D,T] (N m, OpenSim sign: + = flexion moment on the shank).
"""

from __future__ import annotations

import os

import numpy as np

from boneid import io_v3d as io
from boneid.core import SegmentKinematics, chain_kinematics, inverse_dynamics
from boneid.report import p1_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLE = os.path.join(ROOT, "data", "p1_trial13_sample.mat")
OUT = os.path.join(ROOT, "data", "opensim", "varus", "kam_boneid.npz")
D_MM = list(range(0, 55, 5))


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def min_rotation(a, b):
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    vx = np.zeros(a.shape[:-1] + (3, 3))
    vx[..., 0, 1], vx[..., 0, 2] = -v[..., 2], v[..., 1]
    vx[..., 1, 0], vx[..., 1, 2] = v[..., 2], -v[..., 0]
    vx[..., 2, 0], vx[..., 2, 1] = -v[..., 1], v[..., 0]
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))[..., None, None]


def lateral_dir(kin):
    hip, ankle = kin.prox_pos[:, 2], kin.prox_pos[:, 0]
    a = unit(hip - ankle)
    lat = -kin.r_world[:, 2, :, 1]
    return unit(lat - a * np.sum(lat * a, axis=-1, keepdims=True))


def varus_shift(kin, d):
    knee, hip, ankle = kin.prox_pos[:, 1], kin.prox_pos[:, 2], kin.prox_pos[:, 0]
    knee_new = knee + d * lateral_dir(kin)
    r = kin.r_world.copy()
    r[:, 1] = min_rotation(unit(knee - ankle), unit(knee_new - ankle)) @ kin.r_world[:, 1]
    r[:, 2] = min_rotation(unit(hip - knee), unit(hip - knee_new)) @ kin.r_world[:, 2]
    prox, dist = kin.prox_pos.copy(), kin.dist_pos.copy()
    prox[:, 1] = knee_new
    dist[:, 2] = knee_new
    return SegmentKinematics(t=kin.t, rate=kin.rate, r_world=r, prox_pos=prox,
                             dist_pos=dist)


def main():
    trial = io.load_v3d_trial(SAMPLE, 0)
    params = p1_params()
    skel, kin, ground = io.build_chain(trial, side="r")
    kam, sag = [], []
    for d in D_MM:
        k_d = varus_shift(kin, d / 1000) if d else kin
        res = inverse_dynamics(skel, k_d, ground, params)
        r = chain_kinematics(skel, k_d, params.lowpass_hz, params.filter_order)["r"]
        x, y = r[:, 2, :, 0], r[:, 2, :, 1]          # thigh anterior, subject-left
        kam.append(-np.sum(res.joint_torque[:, 1] * x, axis=-1))
        sag.append(np.sum(res.joint_torque[:, 1] * y, axis=-1))
    np.savez(OUT, t=trial.t, d_m=np.array(D_MM) / 1000, kam=np.array(kam),
             knee_sagittal=np.array(sag))
    print("boneid peak KAM (N m) vs d:", np.round(np.array(kam)[:, 12:-12].max(1), 1))


if __name__ == "__main__":
    main()
