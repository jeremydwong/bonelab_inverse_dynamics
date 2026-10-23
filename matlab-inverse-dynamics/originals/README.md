# Original MATLAB sources and implementation review

Recovered 2026-09-10 from `/Users/jeremy/Dropbox/2019/BrentEdwards/matlab`.
These are **originals**, copied byte-for-byte, with original filenames and
comments retained. The Dropbox files were left in place. `manifest.json`
records SHA-256 checksums and the complete source-folder inventory.
This establishes provenance from that folder, not that these exact versions
were used for the ASICS analysis.

After Dropbox downloaded the content, all **79 MATLAB files** (`.m` and `.M`)
were nonempty. Of the **ten** dependencies in the report, **six** were present
in that folder; the three remaining ACTIVE calls — `VirtualPoint_v2.m`,
`getViconForces_Motek.m`, `ContactEvents_Motek.m` — were supplied by the
study author on 2026-09-11 (checksums appended to `manifest.json`), so only
the commented-out `generateMotFile.m` is absent. The whole inverse-dynamics
path of the script is now ported in `src/boneid/legacy.py` and executed
against the corrected pipeline in `notebooks/asics_running_c3d.ipynb`.
(`filterdata.m` and `MyPSD.m`, which the recovered routines call, remain in
the Dropbox folder; the port reproduces `filterdata` — plain `filtfilt` —
and skips `MyPSD`, whose cutoffs the script overwrites with 12 Hz.)

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
| VirtualPoint_v2.m | Recovered 2026-09-11 | Identical to `VirtualPoint.m` in substance: SODER fit per frame, `NewPoint = T*[pneut;1]`, and `RM = T(1:3,1:3)'` — the TRANSPOSED rotation. So every downstream `TM` is Rᵀ. `legacy.virtual_point_v2` ports it; `io_c3d.cluster_poses` computes the same fit with `r_world = R`. |
| getViconForces_Motek.m | Recovered 2026-09-11 | Filters plate force/moment at 18 Hz, rotates to SIMM, picks the plate with the larger mean vertical force, COP = moment about the plate centre / Fy plus the hardcoded plate-centre offsets (which match this data's plate corners exactly), and stores the vertical free moment in `Md(:,3)` — the medio-lateral slot. `legacy.get_vicon_forces_motek` ports it; with the filter off its COP matches ezc3d's to < 1 mm (test). |
| ContactEvents_Motek.m | Recovered 2026-09-11 | Peaks of the 5 Hz-smoothed vertical force above 100 N (first/last dropped), HS/TO by walking out to 5 N on the 20 Hz-smoothed force, right vs left by the alternating medio-lateral COP; the script then tightens to 20 N. `legacy.contact_events_motek` ports it (including `HS(1:2:end-1)=[]` leaving the last element when the count is odd). On S1 AM RP it finds 42 right stances to our 43. |
| G2L.m | Recovered | Resolved: the script passes a T×3×3 TM, and MATLAB's `LCS(i,1:3)` on a 3-D array folds the trailing dims, so the assembled matrix is TMᵀ and `inv(TMᵀ)*Global = TM*Global = Rᵀ·Global` — a proper global-to-local transform. `legacy.g2l` ports it. |
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
acceleration at **32**. With `VirtualPoint_v2` recovered, TM is known to be the transposed SODER
rotation (Rᵀ), so `TMdot*TM = (d/dt Rᵀ) Rᵀ`, which is neither `Ṙ Rᵀ` nor
`Rᵀ Ṙ`. In closed form, for a rotation θ(t) about one fixed axis measured
from the static pose, the extracted component is **−θ̇ cos 2θ** instead of
θ̇: the right size but the wrong sign at the static pose, zero at 45°, the
right sign (wrong size) beyond (verified numerically in
`tests/test_legacy.py`). This one is not hidden by planar motion (nor,
depending on how the lab axes line up with the segment axes, is the
one-sided inertia); it is
softened only because running segments stay within a few tens of degrees
of standing and I·α is small next to the force lever arms. `legacy.room_seg_ang` reproduces it and `legacy.true_seg_ang` is the
corrected counterpart; the ported pipeline's `physics="corrected"` switch
swaps one for the other.

The previous claim that RoomSegAng used second Cardan derivatives is
withdrawn. `core.cardan_alpha` and report curves (c) are retained only as
**hypothetical sensitivity experiments**, not recovered-source parity. Neither
setting of that switch reproduces RoomSegAng. Numerical behavior and existing
experiment defaults were not changed in this source-recovery task.
Reports are exported from executed notebooks; the recovered-source correction
is included in the notebook build.

## Fourth mistake, found in the recovered force routine

`getViconForces_Motek.m` computes the vertical free moment `FM` correctly and
then returns `Md = [0 0 FM]`: the moment about the vertical (SIMM y) axis is
handed to `inv3d` in the z (medio-lateral, flexion/extension) column. On the
running data its magnitude is up to ~10 N m, and it enters the ankle sagittal
moment directly. `legacy.run_legacy_pipeline(free_moment_axis="y")` puts it
back on the vertical axis.

## What the comparison supports

The recovered multiprod confirms ordinary products: it does not secretly add
an inertia transpose or a gyroscopic term. The current solver's two-sided
inertia transformation and gyroscopic term remain deliberate corrections.
SVD fitting, coordinate transforms and broadcasting have direct mathematical
counterparts, but the overall Python pipeline is a replacement, not a faithful
port: frame conventions, filtering, contact selection, calibration and the
patella model differ or remain unresolved.

## MATLAB execution parity (2026-09-14)

MATLAB R2024b (Update 5) executed the original routines on S1 AM RP through
`../harness/run_legacy_trial.m`, which copies the script's inverse-dynamics
lines verbatim and reads the c3d contents from `.mat` files written by
`boneid.legacy.export_for_matlab` (there is no Mac build of the ezc3d MEX).
Against `boneid.legacy.run_legacy_pipeline`: heel strikes and toe-offs
identical (42 right stances), global ankle/knee/hip moments within
6e-11 N m, exported per-stance 12 Hz-filtered local moments within 5e-11 N m,
joint powers within 3e-11 W. The one implementation detail that mattered was
`filtfilt` padding: SciPy's default pad length differs from MATLAB's
`3*(nfilt-1)` and moved stance-edge samples by up to 12 N m until matched.
Reference outputs: `data/asics/matlab_reference/S1_AM_RP_legacy.mat`;
tests: `tests/test_legacy.py`. Checksums verify all imported originals.
