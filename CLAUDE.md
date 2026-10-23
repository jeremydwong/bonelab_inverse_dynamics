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
   rotate/zero force plates (`getViconForces_Motek`),
   contact events from smoothed-force peaks (`ContactEvents_Motek`).
3. **Kinematics** — cluster-based pose per segment via SVD fit
   (`VirtualPoint_v2`/`SODER`), Cardan angles (`RTOCARDA`), central-difference
   velocities/accelerations (`FirstCentral`), Butterworth filtering (`filterdata1`).
4. **Inverse dynamics** — bottom-up Newton–Euler per segment (`inv3d`),
   foot split into rearfoot/forefoot at the MTP joint by which side of the MTP
   axis the COP falls on; moments rotated to proximal-segment frame (`G2L`).
5. **Outputs** — Excel sheets of discrete stride metrics, SIMM/TimSIMM scale,
   motion, and kinetics files, joint powers.

### Subroutines present (`Inverse Dynamics Subroutines/`)

`inv3d` (Newton–Euler core), `SODER` (SVD rigid-body fit), `VirtualPoint`,
`RTOCARDA`/`CardanSegAngles`/`CardanJointAngles`, `FirstCentral` (central diff),
`filterdata1` (Butterworth, skips leading/trailing zeros), `NewCS` (frame change),
`unitvec`, `readC3D`, ezc3d wrappers, plotting helpers (`arrow`, `StickFigure1`,
`plotCS`, `COP_mtpaxis_plot*`, `Main_CS`).

### Originals (`matlab-inverse-dynamics/originals/`) and the Python port

Every routine the script actively calls is now in the repo (`RoomSegAng`,
`multiprod`, `matexp`, `G2L`, `SIMMpatellaCS`, `TreadOffsets` recovered
2026-09-10; `getViconForces_Motek`, `ContactEvents_Motek`, `VirtualPoint_v2`
on 2026-09-11; only the commented-out `generateMotFile` is absent).
`src/boneid/legacy.py` is a line-by-line Python port of the script's
inverse-dynamics path (SIMM frame, Vaughan anthropometrics with the typos,
foot split at the MTP axis, one-sided inertia, no gyroscopic term,
`TMdot*TM` angular velocity, free moment in the wrong slot, no kinematic
pre-filter, 12 Hz per-stance post-filter) with switches that correct one
thing at a time; `asics_running_c3d` runs it against the corrected
pipeline on the same c3d files. MATLAB R2024b IS installed
(`/Applications/MATLAB_R2024b.app`, Signal + Statistics toolboxes, no Mac
ezc3d MEX): `matlab-inverse-dynamics/harness/run_legacy_trial.m` runs the
original routines on `.mat` exports of the c3d files
(`legacy.run_matlab_legacy`), and the committed reference
`data/asics/matlab_reference/S1_AM_RP_legacy.mat` shows the port matches the
executed MATLAB to ~1e-11 N m (`test_port_matches_the_executed_matlab`).

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
- **Angular velocity from the wrong matrix product** (`originals/RoomSegAng.m`
  line 12): `TMdot*TM` with `TM = Rᵀ` (per `VirtualPoint_v2`). For a planar
  rotation θ from the static pose it yields −θ̇·cos 2θ instead of θ̇: right
  size but WRONG SIGN at the static pose, zero at 45°, right sign (wrong size)
  beyond; on S1_AM_RP the shank ω correlates −0.975 with the true one (see
  `legacy_error_math`). Planar motion does NOT
  forgive it (`legacy.room_seg_ang`, test `test_room_seg_ang_planar_closed_form`).
  The earlier "Cardan second derivatives" guess was wrong.
- **Free moment in the wrong column** (`originals/getViconForces_Motek.m`):
  the vertical free moment `FM` is returned as `Md = [0 0 FM]`, i.e. in the
  medio-lateral (flexion/extension) slot, up to ~10 N m in running.
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
- `io_c3d.py` — raw c3d inputs (ezc3d) with a STATIC calibration trial: joint
  centres/frames measured in the static pose and carried through dynamic
  trials as virtual points on SVD cluster fits (CAST / legacy
  `VirtualPoint_v2` approach); marker label-swap repair + despiking; per-bout
  foot attribution when both feet strike one belt. Optional torso: if the
  static trial marks C7/CLAV/STRN/T10 + RAC/LAC, `build_chain` adds a de Leva
  torso (L5S1 -> mid-acromion) as a 5th segment (the Asics data has none).
- `simulate.py` — synthetic-data generators (squat-to-stand, pendulum, the
  brick tutorials, synthetic markers) used by key tests and the tutorials.
  `simulate_squat_to_stand_c3d` writes an idealized squat (de Leva right
  leg + pelvis + torso of an 80 kg person, no left leg/arms) as static +
  dynamic c3d files in the Asics markerset PLUS torso markers (via
  `io_c3d.write_c3d`) so it runs through the REAL Asics calls; truth is
  computed top-down and the residual is zero. The plate weighs only the
  modelled ~56 kg, so it is analysed with `calibrate_static(body_mass=80)`.
  `knee_slip_m` makes the knee slide (non-ideal joint) and the truth then
  includes rotational/translational joint power: `notebooks/tutorial_6dof`
  shows 6-DoF power recovered, the audit closing only with it, and 6-DoF
  power invariant to where the joint centre is put.
- `legacy.py` — faithful port of the legacy MATLAB inverse-dynamics path
  (`run_legacy_pipeline`), with `physics="corrected"`, `free_moment_axis`
  and `marker_lowpass_hz` switches for the comparison ladder in the c3d report.
- `viz.py` — meshcat visualization of markers, segments, forces; pelvis band,
  mass-sized torso, URDF-skeleton option (vendored `assets/urdf/`, LGPL).
- `report.py` — HTML reports with figures (energy balance, per-step analysis).
- `validate.py` — cross-subject validation sweep (RMSE vs Visual3D, p1–p9).
- `examples/quickstart.py` — the canonical 10-step run-through.
- `examples/quickstart_c3d.py` — the c3d/static-trial variant (Asics running
  study), 8 steps → `reports/asics_running_c3d.html`.

Model tree (whole body, 12 segments): two legs [foot, shank, thigh] bottom-up
into a shared pelvis; L5S1 is a real joint to the torso (trunk + head lumped,
de Leva); arms [forearm+hand, upper arm] hang from the shoulders (AC/EP/WR
markers). The residual wrench lives at the torso and its mean is ~0 on good
data — it measures soft-tissue/model error, not missing anatomy.

Tests: `tests/test_key.py` holds the small intelligible KEY suite
(KEY-STATIC, KEY-PENDULUM, KEY-GYRO, KEY-WRENCH-EQUIV, TEST-SIMULATED-S2S,
KEY-S2S-C3D — the whole c3d pipeline on a known body);
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
- **Titles are "WHAT THIS IS: WHY IT'S HERE", in plain words.** Say what the
  page contains, then the point of it. No teasers, no drama, no cleverness
  ("Forces You Cannot See", "a Body With No Secrets" are out). Good:
  "Squat-to-stand: idealized motion of four segments, no residual forces or
  noise". Sentence case.
- **Prose sounds like Jeremy**, the voice of jeremydwong.github.io: first
  person, plain and a bit casual, short sentences, asides in parentheses,
  honest hedges ("I think"), the odd "=)". Not polished essay prose. It is
  fine for it to read as amateurish; relatability is the point. Numbers and
  technical claims stay exact.

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
- Asics running c3d set (io_c3d): `data/asics/S1_{AG,AH,AM}_{Static,PI,RP}.c3d`
  — subject S1, three shoe conditions × two 30 s running trials + one static
  each; 300 Hz mocap / 3000 Hz analog, two type-2 plates, same lab axes as
  the walking set (+z up, +x subject-left, +y posterior). Only the RIGHT leg
  is instrumented (legacy hip = 0.70·RGT + 0.30·LGT); medial ankle/knee
  markers exist ONLY in the static trial, so joint centres are virtual points
  carried by cluster fits. BOTH feet strike plate 1, so contact bouts are
  attributed by COP-to-foot-marker proximity (~0.1 m own vs ~0.3 m other).
  Known data warts, both handled in `load_c3d`: thigh-marker LABEL SWAPS
  (brief in AM_RP/AH_PI, permanent from t=4.5 s in AM_PI) and belt ringing at
  50-100 N between steps (bout peak filter). Body mass ≈ 49.9 kg from static
  Fz. Sanity: ankle/knee/hip peak sagittal torques ≈ 2.9/2.2/1.6 N m/kg,
  energy imbalance < 2% on all six trials.
- RBDS public running dataset (Fukuchi, Fukuchi & Duarte 2017, PeerJ 5:e3298,
  CC-BY; figshare 10.6084/m9.figshare.4543435): 39 subjects, single-belt
  treadmill 2.5/3.5/4.5 m/s, 150/300 Hz, bilateral CAST markerset with a
  static trial, Y-UP lab (load with `io_c3d.Y_UP_AXES`), plus the dataset's
  own Visual3D processed output for validation. `io_c3d` RBDS section +
  `validate.rbds_sweep` (downloads to ~/.cache/boneid/rbds, ~300 MB;
  `uv run python -m boneid.validate rbds`). Gotchas: subjects 3-9 use
  2-digit file names; some run trials drop an MT marker; the NEWER
  RBDSinfo.txt (figshare file 39452932) covers all 39 subjects.
  Result (101 subject-speeds, right leg vs their V3D): median torque RMSE
  0.23/0.26/0.11 N m/kg, correlations 0.98/0.99/0.99, energy imbalance
  ~1.6%. `data/rbds_running_reference.npz` holds their ensemble mean
  stance-normalized torques/powers + mean stance durations, overlaid
  (button-toggle) on the Asics c3d report's figures.
- In-repo sample: `data/p1_trial13_sample.mat` (2.5 MB, trial-13 slot only,
  made by `io_v3d.trim_v3d_mat`; its `.index` is 0) — reproduces validation
  numbers bit-for-bit. Full P1 export (84 MB) is a GitHub release asset,
  tag `data-p1-v1`.
- End-to-end check: `io_v3d.build_chain` + `core.inverse_dynamics` reproduces
  Visual3D's rFt/rSh ProxEndTorque to ~1.6/1.7 N m RMS on p1 trial 0
  (peaks ~105/70 N m); hip ~9 N m (hip-center model differences).
