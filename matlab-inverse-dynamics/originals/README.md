# Original MATLAB sources and implementation review

Recovered 2026-09-10 from `/Users/jeremy/Dropbox/2019/BrentEdwards/matlab`.
These are **originals**, copied byte-for-byte, with original filenames and
comments retained. The Dropbox files were left in place. `manifest.json`
records SHA-256 checksums and the complete source-folder inventory.
This establishes provenance from that folder, not that these exact versions
were used for the ASICS analysis.

After Dropbox downloaded the content, all **79 MATLAB files** (`.m` and `.M`)
were nonempty. Of the **ten** dependencies in the report, **six** are present
and **four** exact filenames are absent. The ten include eight active
functions and two commented-out calls; they are not an exhaustive dependency
closure of the original script.

We imported 13 files: the six recovered dependencies, five helpers needed by
SIMMpatellaCS (Cardan2RM, repmat2 and the three global rotation functions),
and two related force/contact routines for comparison. Existing reference
files were not overwritten. Add this directory to the MATLAB path along with
`../Inverse Dynamics Subroutines` to resolve recovered functions; the full
script still needs the missing functions and its external configuration.

| Requested original | Recovery | Current Python implementation / assessment |
|---|---|---|
| RoomSegAng.m | Recovered | `core.angular_velocity` and `chain_kinematics` differentiate rotation matrices. The original is different; see below. |
| multiprod.m | Recovered | NumPy `einsum` and matrix products cover the products used by our solver. Original `[2 3], [2 3]` means ordinary per-frame matrix multiplication, with no implicit transpose. We do not port its general N-dimensional API. |
| matexp.m | Recovered | The script's `matexp(I, T, [1 2], 2)` replicates a 3×3 tensor to T×3×3. Python broadcasts a single local tensor in `inertia_world` and `_rot_term`, avoiding the replicated allocation. Equivalent replication purpose; the default inertia rotation deliberately differs. |
| VirtualPoint_v2.m | Absent | `fit_rigid_transform`, `cluster_poses`, `virtual_points` provide SVD fitting and point transport. The existing `VirtualPoint.m` is related, but returns a transposed rotation; v2 equivalence is unverified. |
| getViconForces_Motek.m | Absent | `io_c3d.load_c3d` uses ezc3d platform extraction and SI conversion; wrench construction and foot attribution occur separately. Recovered `getViconForces.m` has a different signature, hardcodes plate 2 and calibration offsets, and applies slope rotation. It is not a Motek substitute. |
| ContactEvents_Motek.m | Absent | `core.detect_contact` thresholds force and joins nearby samples. Recovered `ContactEvents.m` smooths for peak detection, removes first/last peaks, searches 20 N crossings and uses COP plus slope to choose a leg. Different algorithm and signature; no exact event parity established. |
| G2L.m | Recovered | `inverse_dynamics` computes local torque with the proximal rotation transpose. Original reconstructs a matrix from nine entries and explicitly inverts it. For matching orthonormal frame/layout these express a coordinate transform; flattened versus T×3×3 MATLAB indexing and the missing v2 output convention must be resolved before numerical parity can be claimed. Python does not expose an identical G2L API or local-force output here. |
| SIMMpatellaCS.m | Recovered | No equivalent patella model in our solver. Original interpolates knee angle to patella rotation using a fixed table, then calls Cardan2RM; its `scale` argument is unused. This is a modeling/output omission, not an equivalent implementation. |
| TreadOffsets.m | Recovered; call commented out | No equivalent automatic analog-baseline subtraction in `load_c3d`. Original chooses samples using analog columns 3/9 > −20 and subtracts all-channel means. It subtracts a constant, despite its comment about removing a trend. Calibration-specific behavior should not be applied blindly. |
| generateMotFile.m | Absent; call commented out | No matching GRF `.mot` exporter identified. Not needed by that commented path. |

## Error 3: incorrect angular-velocity matrix product

The recovered code computes `TMdot = FirstCentral(...)`, then
`S = multiprod(TMdot, TM, [2 3], [2 3])`. It extracts S(3,2), S(1,3),
S(2,1), with an alternate skew entry selected for an entire component if its
maximum exceeds 200, and differentiates those velocities with FirstCentral.
The velocity filter is commented out. For three or fewer frames AA is a
single zero row. It does **not** differentiate Cardan angles.

Our `angular_velocity` forms `dR/dt @ R.T`, averages the two skew entries,
and differentiates that velocity for acceleration. For world-from-segment
R this is the appropriate world angular velocity construction. The original
`TMdot @ TM` is generally not skew-symmetric for a rotation trajectory; it
is neither the usual world nor body angular-velocity product. This is a matrix-product error for rotation inputs under either usual frame
convention; reversing the convention changes the required order/sign, not the
need for a transpose. The product is at **RoomSegAng.m line 12**; matrix
differentiation is at **10–11**, velocity extraction at **26–28**, and
acceleration at **32**. Reproducing its effect on the original ASICS results
still needs `VirtualPoint_v2` to establish what TM held.
The existing non-v2 VirtualPoint exports a transpose, so its convention must
not be silently substituted for the missing version.

The previous claim that RoomSegAng used second Cardan derivatives is
withdrawn. `core.cardan_alpha` and report curves (c) are retained only as
**hypothetical sensitivity experiments**, not recovered-source parity. Neither
setting of that switch reproduces RoomSegAng. Numerical behavior and existing
experiment defaults were not changed in this source-recovery task.
Reports are exported from executed notebooks; the recovered-source correction
is included in the notebook build.

## What the comparison supports

The recovered multiprod confirms ordinary products: it does not secretly add
an inertia transpose or a gyroscopic term. The current solver's two-sided
inertia transformation and gyroscopic term remain deliberate corrections.
SVD fitting, coordinate transforms and broadcasting have direct mathematical
counterparts, but the overall Python pipeline is a replacement, not a faithful
port: frame conventions, filtering, contact selection, calibration and the
patella model differ or remain unresolved.

Validation: all 28 existing KEY, coverage and ASICS sensitivity tests passed
(`.venv/bin/python -m pytest tests/test_key.py tests/test_coverage.py
tests/test_asicssource.py -q`).
These check Python physics and geometry, not MATLAB execution parity.
No MATLAB/Octave comparison was run. Checksums verify all 13 imported originals.
