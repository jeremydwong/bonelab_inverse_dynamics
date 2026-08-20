# bonelab_inverse_dynamics

Whole-body inverse dynamics from Visual3D-aligned forces and kinematics.
Two generations live here:

1. `matlab-inverse-dynamics/` — legacy MATLAB pipeline (kept as reference, do not extend).
2. `src/boneid/` — Python replacement (uv-managed, tested, this is where work happens).

Run everything through uv: `uv run pytest`, `uv run python -m boneid ...`.

## Legacy MATLAB: lay of the land

### Entry point

`matlab-inverse-dynamics/Asics_Moments4JCF_v3.m` (~1400 lines) is the single entry
point — a study-specific batch script (Asics shoe study), not a library. Flow:

1. **Static trial** → read c3d (ezc3d), markers → per-marker variables via `eval`,
   rotate into SIMM frame (`NewCS`), joint centers from marker midpoints,
   anthropometrics from an Excel sheet (or hardcoded defaults), segment masses from
   Vaughan regressions, segment inertias from Vaughan/Zatsiorsky regressions.
2. **Dynamic trial** → read c3d, interpolate analog forces to mocap rate,
   rotate/zero force plates (`getViconForces_Motek` — **missing from repo**),
   contact events by vertical-force threshold (`ContactEvents_Motek` — **missing**).
3. **Kinematics** — cluster-based pose per segment via SVD fit
   (`VirtualPoint_v2`/`SODER`), Cardan angles (`RTOCARDA`), central-difference
   velocities/accelerations (`FirstCentral`), Butterworth filtering (`filterdata1`).
4. **Inverse dynamics** — bottom-up Newton–Euler per segment (`inv3d`),
   foot split into rearfoot/forefoot at the MTP joint by which side of the MTP
   axis the COP falls on; moments rotated to proximal-segment frame (`G2L` — missing).
5. **Outputs** — Excel sheets of discrete stride metrics, SIMM/TimSIMM scale,
   motion, and kinetics files, joint powers.

### Subroutines present (`Inverse Dynamics Subroutines/`)

`inv3d` (Newton–Euler core), `SODER` (SVD rigid-body fit), `VirtualPoint`,
`RTOCARDA`/`CardanSegAngles`/`CardanJointAngles`, `FirstCentral` (central diff),
`filterdata1` (Butterworth, skips leading/trailing zeros), `NewCS` (frame change),
`unitvec`, `readC3D`, ezc3d wrappers, plotting helpers (`arrow`, `StickFigure1`,
`plotCS`, `COP_mtpaxis_plot*`, `Main_CS`).

### Functions called but MISSING from the repo

`getViconForces_Motek`, `ContactEvents_Motek`, `VirtualPoint_v2`, `RoomSegAng`
(global angular vel/acc), `G2L`, `multiprod`, `matexp`, `SIMMpatellaCS`,
`generateMotFile`, `TreadOffsets`. The MATLAB pipeline **cannot run** from this
repo alone.

### Confirmed defects in the legacy code (why we rewrote)

- **Rotational inertia transform is wrong** (`Asics_Moments4JCF_v3.m:369-380`):
  global inertia computed as `I_G = R * I_local` — a one-sided product. The
  correct similarity transform is `I_G = R * I_local * R'`. The resulting
  "inertia" isn't even symmetric.
- **Euler's equation is incomplete** (`inv3d.m`): moment balance uses
  `I*alpha` only — the gyroscopic term `omega × (I*omega)` is omitted.
- **Copy-paste inertia typo** (`Asics_Moments4JCF_v3.m:362`):
  `Ileg = [Ilx 0 0; 0 Ify 0; 0 0 Ilz]` uses the **foot's** transverse moment
  `Ify` for the shank. Thigh `Ity` regression uses foot length `A(6)`.
- **"Angular acceleration" is Cardan-angle second derivatives** (`RoomSegAng`,
  missing, but usage implies it) rather than the true angular acceleration
  vector; only valid for small/planar rotations.
- **Hardcoded study specifics throughout**: absolute `Z:\` paths, subject lists,
  shoe/speed condition codes, force-plate offsets, a ~150-line hardcoded
  bad-trial table keyed by subject number, per-condition shoe heights.
- **`eval`-generated variables** for every marker name; state leaks across
  loop iterations (`clear markernames` mid-loop).
- **No tests, no energy/consistency checks, no residual reporting.**

### What is worth conserving (the genuine approach)

- SVD rigid-body pose fit from marker clusters (SODER / Söderkvist & Wedin 1993)
  with fit residual as data-quality signal.
- Bottom-up Newton–Euler recursion foot→shank→thigh(→pelvis), reaction wrench
  negated and passed proximally.
- Contact detection by vertical-force threshold with edge tightening.
- Moments reported in the proximal segment frame; joint power = τ · ω_rel.
- Vaughan / Zatsiorsky anthropometric regressions (verified against sources).

## Python replacement: `src/boneid/`

Design rules (Casey Muratori-inspired):

- **No member functions.** Plain `@dataclass` records of numpy arrays; free
  functions that operate on them.
- **Few files.** Don't add a file unless a human juggling the codebase in their
  head needs the boundary.
- All arrays SI units, time-major: `[T, ...]`. Rotations are `[T, 3, 3]`
  world-from-segment matrices.
- Joint torques computed **about the lab/treadmill origin wrench representation**
  by default (no numerically-unstable COP division); COP path supported for
  comparison.
- Every analysis run produces an **energy-balance report** and the **residual
  wrench at the top of the chain** — these are the correctness instruments.

Files:

- `core.py` — datatypes + math (rotations, differentiation, filtering,
  Newton–Euler serial/two-leg/whole-body recursions, energy audits, COM /
  peripheral / unified-deformable foot power decompositions).
- `io_v3d.py` — Visual3D-style inputs: the `*_5StridesData.mat` export format
  (forces, markers, Visual3D reference kinetics), force plate geometry,
  de Leva anthropometrics (legs + torso/head + arms).
- `simulate.py` — synthetic-data generators (squat-to-stand, pendulum) used by
  key tests.
- `viz.py` — meshcat visualization of markers, segments, forces; pelvis band,
  mass-sized torso, URDF-skeleton option (vendored `assets/urdf/`, LGPL).
- `report.py` — HTML reports with figures (energy balance, per-step analysis).
- `validate.py` — cross-subject validation sweep (RMSE vs Visual3D, p1–p9).
- `examples/quickstart.py` — the canonical 10-step run-through.

Model tree (whole body, 12 segments): two legs [foot, shank, thigh] bottom-up
into a shared pelvis; L5S1 is a real joint to the torso (trunk + head lumped,
de Leva); arms [forearm+hand, upper arm] hang from the shoulders (AC/EP/WR
markers). The residual wrench lives at the torso and its mean is ~0 on good
data — it measures soft-tissue/model error, not missing anatomy.

Tests: `tests/test_key.py` holds the small intelligible KEY suite
(KEY-STATIC, KEY-PENDULUM, KEY-GYRO, KEY-WRENCH-EQUIV, TEST-SIMULATED-S2S);
`tests/test_coverage.py` holds broader coverage. Run: `uv run pytest`.

## Rules (reporting & visualization)

- **Bilateral comparisons share axes.** When the same joint is plotted for the
  left and right side (torques, powers, forces, angles), the two panels MUST
  have identical y-limits — unequal axes make the comparison meaningless.
- **Animations loop.** Every meshcat animation is sent with a large
  `repetitions` (e.g. 10000) so it plays continuously; an animation that plays
  once and parks on its final frame reads as broken.
- **The drawn body must read as one body.** The pelvis is drawn with real
  width (spanning the two hip centers) and the unmodeled upper body is drawn
  as a torso volume sized from its actual mass and marker-measured dimensions
  (L5S1 to mid-acromion, radius from mass at ~1000 kg/m^3) — segments must
  not float disconnected.
- **Never invent conventions or data.** Alternative joint-center conventions,
  extra joints (e.g. an MTP joint), or anthropometrics are only offered when
  the markerset actually supports them; otherwise say so and omit.
- **Figures are SVG** (`report.fig_svg`), so they stay sharp at any zoom.

## Validation data

- Local: `/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p{1..9}_5StridesData.mat`
  — 9 subjects × 33 trial slots (5 empty placeholders; 28 real trials forming a
  **walk→run treadmill speed sweep**); Visual3D exports at 120 Hz mocap /
  1200 Hz analog including marker positions, dual-belt force plates (corners in
  mm; plate surface at z = −0.1746 m; +z up, +x subject-left, +y posterior),
  **and Visual3D's own computed outputs** to validate against
  (`Kinetic_Kinematic.*ProxEndForce/Torque` in N and N m lab frame;
  `Link_Model_Based.*_moment` in N m/kg, angles in degrees).
  Kinematic and kinetic arrays are pre-cropped to the 5-stride window sharing
  one origin (`t_mocap = i/120`, `t_analog = k/1200`; verified aligned to
  ≤ 2 analog samples). Naming trap: right shank is `rSk` for AngVel/CG* but
  `rSh` for ProxEnd*; use `io_v3d.reference_series`.
- Published dataset & description: https://www.nature.com/articles/s41597-022-01817-1
- In-repo sample: `data/p1_trial13_sample.mat` (2.5 MB, trial-13 slot only,
  made by `io_v3d.trim_v3d_mat`; its `.index` is 0) — reproduces validation
  numbers bit-for-bit. Full P1 export (84 MB) is a GitHub release asset,
  tag `data-p1-v1`.
- End-to-end check: `io_v3d.build_chain` + `core.inverse_dynamics` reproduces
  Visual3D's rFt/rSh ProxEndTorque to ~1.6/1.7 N m RMS on p1 trial 0
  (peaks ~105/70 N m); hip ~9 N m (hip-center model differences).
