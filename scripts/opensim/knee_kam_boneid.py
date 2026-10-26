"""boneid's external knee adduction moment under the varus sweep (cross-check).

Run in the REPO's uv env:  uv run python scripts/opensim/knee_kam_boneid.py

Same geometry as the AFF notebook (notebooks/aff_bowed_femur.ipynb), now
`boneid.simulate.varus_shift`: the right knee centre is moved laterally by d
perpendicular to the hip-ankle line, thigh and shank re-posed by the minimal
rotation, hip/ankle/foot/GRF untouched, then boneid's
right-leg inverse dynamics.  Writes data/opensim/varus/kam_boneid.npz:
t [T], d_m [D], kam [D,T] (N m, + = external varus/adduction, thigh frame),
knee_sagittal [D,T] (N m, OpenSim sign: + = flexion moment on the shank).
"""

from __future__ import annotations

import os

import numpy as np

from boneid import io_v3d as io
from boneid.core import chain_kinematics, inverse_dynamics
from boneid.report import p1_params
from boneid.simulate import varus_shift

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLE = os.path.join(ROOT, "data", "p1_trial13_sample.mat")
OUT = os.path.join(ROOT, "data", "opensim", "varus", "kam_boneid.npz")
D_MM = list(range(0, 55, 5))


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
