# OpenSim muscle forces for P1 trial 13 (SO + MocoInverse)

This estimates right-leg muscle forces for the in-repo walking sample
(`data/p1_trial13_sample.mat`) two ways and writes them where the viewer reads
them:

- **Static optimization**: the classic OpenSim `AnalyzeTool` +
  `StaticOptimization` analysis (sum of activation², reserves allowed).
- **Dynamic optimization**: OpenSim Moco `MocoInverse`, which is direct
  collocation (CasADi + IPOPT) with activation dynamics. Kinematics are
  prescribed from IK, and it minimizes squared excitations plus squared activations.

Both run on the **same** processed model, external loads and 12 Hz-filtered
IK. That way the only thing that differs is static vs dynamic.

## Environments

There are two Python environments. Don't mix them.

| step | env | why |
|---|---|---|
| `export_inputs.py` | repo's uv env (`uv run ...`) | it imports `boneid` |
| `pipeline.py`, `collect.py` | conda env `opensim` | OpenSim 4.6 Python bindings |

Make the conda env once (osx-arm64 builds are on the opensim-org channel):

```sh
/Users/jeremy/.julia/conda/3/aarch64/bin/mamba create -y -n opensim \
    -c opensim-org -c conda-forge opensim=4.6 python=3.11 scipy
```

## Reproduce

```sh
cd /Users/jeremy/Git/brent/bonelab_inverse_dynamics
uv run python scripts/opensim/export_inputs.py          # trc / grf mot / boneid ID
PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
$PY scripts/opensim/pipeline.py scale ik id so moco collect
```

Stages can be run one at a time; each stage reads the previous stage's files
in `data/opensim/`. `moco` is the slow one (see the runtime in the JSON).
`scale` downloads `RajagopalLaiUhlrich2023.osim` from opensim-models if it's
not already there. The bone `.vtp` meshes are NOT downloaded, so expect a
wall of "Couldn't find file *.vtp" warnings. They are harmless (display only).

## What each step does

1. **Export** (`export_inputs.py`): writes markers as `.trc` (m, 120 Hz) plus
   the Visual3D hip centres as virtual markers RHJC/LHJC, GRF per foot as
   `.mot` (1200 Hz), and boneid's own right-leg ID. Lab to OpenSim:
   `X = -y_lab` (anterior), `Y = z_lab` (up), `Z = -x_lab` (right), with no shift.
   The COP gets slid along the force line from the plate-origin height
   (z = -0.1746 m) up to z = 0. The wrench stays the same, only the drawn point moves.
2. **Scale**: RajagopalLaiUhlrich2023 with the lab markerset attached. The
   export has no static trial, so I compute uniform per-body factors from the
   trial-mean per-frame marker distances (HJC to mid-epicondyles, etc.) and
   feed them to `ScaleTool` as `manualScale`. Mass is boneid's estimate
   (85.16 kg) with the mass distribution kept. Feet and forearms use leg and
   humerus factors, because the lab's 5TH and WR markers aren't at the
   model's landmarks.
3. **IK**: three passes with marker placement in between. Pass 1 uses only
   anatomical markers. Then the non-anatomical markers go to their trial-mean
   body-frame spots. Then all markers, place again, and a final pass. ASIS,
   HJC and the epicondyle/malleolus midpoints never move.
4. **ID**: `InverseDynamicsTool`, 12 Hz coordinate filter (same as boneid),
   GRF applied to `calcn_r` / `calcn_l`, muscles excluded.
5. **SO / MocoInverse**: processed model = DeGrooteFregly2016 muscles (rigid
   tendon, passive fibre force off, F-L curve width x1.5, i.e. the Moco
   walking example settings), subtalar/mtp/wrist welded, pelvis residual
   actuators (optimal force 100), and 1 N m reserves on every other joint.
6. **Collect** (`collect.py`): moment arms come from `Muscle.computeMomentArm`
   at the filtered IK poses. Everything gets resampled to the trial clock
   `t = i/120` and written as `data/opensim/p1_trial13_muscles.npz` + `.json`.

## Output contract (`data/opensim/p1_trial13_muscles.npz`)

`t [T]`, `muscle_names [M]` (right-leg, OpenSim names), `so_force`,
`dyn_force`, `so_activation`, `dyn_activation [T,M]` (NaN outside each solve
window), `joint_names = [ankle, knee, hip]` (`ankle_angle_r`, `knee_angle_r`,
`hip_flexion_r`), `moment_arm [T,M,3]` (m), `group [M]`, `crosses [M,3]`,
`so_moment`, `dyn_moment`, `id_moment_opensim`, `id_moment_boneid [T,3]`
(N m, OpenSim sign: + = dorsiflexion / knee FLEXION (Rajagopal walker knee) /
hip flexion), `window [2]` (MocoInverse window, s). The JSON sidecar holds
scaling, IK errors, solver settings, runtimes, reserve usage and notes.

## Fibre kinematics and muscle parameters

`fiber_kinematics.py` (conda env, ~10 s) adds fibre kinematics and the
per-muscle parameters to the same npz, leaving every key above untouched:

```sh
$PY scripts/opensim/fiber_kinematics.py
```

With a rigid tendon and prescribed kinematics, fibre length and velocity are
pure geometry (l_MT = l_slack + l_M cos(alpha), constant fibre width), so SO and
MocoInverse see the same fibres. The script measures both and checks that:
the SO side is an `AnalyzeTool` + `MuscleAnalysis` on the exact SO setup
(`p1_setup_fiber_analysis.xml`, results in `fiber/`), and the Moco side is the
saved solution re-analysed through its own `MocoStudy` (`study.analyze`).
They agree to 2.4e-4 RMS in l_M/l_opt and 7.3e-4 RMS in v_M/(v_max l_opt).
The worst single samples are glmax1 (0.019 and 0.009), and I think that's the
two tools filtering/splining the same IK slightly differently. The stored
values are the MocoInverse ones.

New keys, all on `t`, NaN outside the Moco window:
`fiber_length_norm`, `fiber_velocity_norm [T,M]` (l_M/l_opt, and
v_M/(v_max l_opt) with + = lengthening), `mtu_length`, `tendon_length [T,M]`
(m), `pennation [T,M]` (rad). Per muscle: `f_iso` (N), `l_opt`, `l_slack` (m),
`v_max` (l_opt/s, 10 for every muscle), `alpha_opt` (rad), `passive_strain_e0`.
The JSON gets a `fiber_kinematics` block: the SO-vs-dyn agreement, force
reconstructions, the model constants and an activation-ODE check.

Things this turned up about the processed model (also in the JSON):

- Activation time constants are **0.010/0.040 s**, inherited from the Rajagopal
  muscles, not DGF's 0.015/0.060 defaults. The smoothing is b = 0.1 (the
  default), so activation and deactivation end up about equally fast.
- The muscles carry **fibre damping 0.1**. MocoInverse's tendon force is
  `F_iso (a f_L f_V + 0.1 v_M_norm) cos(alpha)` (to 3e-11 N).
- StaticOptimization's force is `a F_iso f_L f_V`, with **no cos(alpha) and no
  damping** (to 4e-5 N), and that's what it applies along the path.

## Knee contact forces and the varus sweep

Two more scripts sit on top of the solves above (`knee_contact.py`, `knee_collect.py`
in the conda env; `knee_kam_boneid.py` in the uv env):

```sh
PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
$PY scripts/opensim/knee_contact.py baseline              # contact on the existing SO + MocoInverse
$PY scripts/opensim/knee_contact.py step 0 5 10 15 20 25 30 35 40 45 50   # ~1 min each serially; run them in parallel if you like
$PY scripts/opensim/knee_contact.py moco 25 50            # MocoInverse on two sweep steps (~20-30 min each)
uv run python scripts/opensim/knee_kam_boneid.py          # boneid KAM under the same shift
$PY scripts/opensim/knee_contact.py collect               # -> data/opensim/p1_trial13_knee_contact.{npz,json}
```

- **Knee reaction**: a Newton-Euler balance of the subtree below the knee
  (tibia + talus + calcn + toes): IK accelerations (12 Hz low-pass, quintic
  spline), gravity, measured GRF + free moment, and muscle tension along every
  path segment that touches the subtree. It's taken about the walker knee's tibial
  child-frame origin and expressed in the tibia frame. I didn't use OpenSim's
  JointReaction analysis, because it realizes forward dynamics. Without muscles
  this reproduces OpenSim ID's knee moment (r = -0.999).
- **Medial/lateral split**: two point contacts on a flat plateau perpendicular to
  the tibial axis, +-d_ic/2 from the knee centre. d_ic = 40 mm x the tibia scale
  factor (Winby 2009, Lerner 2015, Gerus 2013). Ligaments carry nothing. If a
  condyle goes negative, it's clamped to zero, and the ligament tension it would
  need is stored separately (`*_lig_lateral`).
- **Varus sweep**: thigh markers are rotated about the hip, and shank markers about
  the ankle, so the knee moves laterally by d. The model gets a matching fixed
  varus: the walker knee's tibia offset frame is rotated by
  theta = d(1/L_thigh + 1/L_shank) about tibial X. After that it's one IK pass,
  then ID, SO and contact. Intermediates are in `data/opensim/varus/dXX/`, with
  `info.json` per step (IK errors, theta, runtimes).
- **IK check**: the model knee centre's lateral offset from its hip-ankle line is
  compared against d = 0 (`ik_verification` in the JSON).

Output contract (`data/opensim/p1_trial13_knee_contact.npz`), on t = i/120 with NaN
outside the solve windows. Contact is in BW (body_mass x 9.80665) and moments are
in N m.
- **Contact, SO and dynamic**: `so_contact_{total,medial,lateral}`,
  `dyn_contact_{total,medial,lateral}`, `*_lig_lateral`, `*_medial_share`, and
  `*_frontal_moment`.
- **Sweep, 11 steps by T**: `sweep_d_m [11]`, `sweep_theta_deg`,
  `sweep_so_{total,medial,lateral,lig_lateral}`, `sweep_frontal_moment_os` (the
  external adduction moment, + = varus), `sweep_kam_boneid`,
  `sweep_knee_sagittal_os` and `sweep_knee_sagittal_boneid` (OpenSim sign, +
  = flexion), and `sweep_dyn_d_m` + `sweep_dyn_{total,medial,lateral,lig_lateral}`.
- **Constants**: `d_ic` and `body_mass`.

The JSON has the assumptions, citations (with Semantic Scholar counts), per-stance
peaks, the sweep table, the IK verification and runtimes.

## Turning on tendon compliance (C1 / C2 / C3)

`compliance.py` (conda env) re-solves the MocoInverse with compliant tendons,
one stage at a time, each warm-started from the one before
(rigid -> C1 -> C2 -> C3):

```sh
PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
$PY scripts/opensim/compliance.py list       # l_slack/l_opt, tendon stiffness, the C1/C2/C3 lists
$PY scripts/opensim/compliance.py c1         # then c2, then c3 (each needs the previous one)
$PY scripts/opensim/compliance.py collect    # c*_ keys -> p1_trial13_muscles.npz, JSON block "tendon_compliance"
$PY scripts/opensim/compliance.py contact    # c*_contact_* -> p1_trial13_knee_contact.npz
```

- **C1**: compliant soleus, gasmed, gaslat (both legs, since the MocoInverse
  solves both legs' muscles).
- **C2**: compliant wherever l_slack / l_opt > 1 (23 muscles per leg). At
  eps0 = 0.049 a tendon at F_iso stretches 0.049 l_slack, so at a ratio of 1
  that's ~5% of l_opt of fibre shift, small next to the f-L width. The
  triceps surae (5.4-6.6) shift 27-32% of l_opt.
- **C3**: every muscle compliant, and passive fibre force switched on.

Everything else is the rigid solve's: same processed model, IK, window,
mesh (256 intervals, Hermite-Simpson), tolerances. Compliance is DGF with
IMPLICIT tendon dynamics (`set_tendon_compliance_dynamics_mode("implicit")`),
plus a `MocoInitialVelocityEquilibriumDGFGoal`, which MocoInverse doesn't
add by itself. The problem is solved via `inverse.initialize()` +
`study.solve()` (that's the only way to hand it a guess), so the kinematic
states are put back into the solution afterwards from the rigid solution.
One gotcha: in Python the solver of the returned study must be re-pointed
with `solver.resetProblem(study.getProblem())` before `createGuess()`, or it
segfaults.

Tendon stiffness: every muscle in the processed model has
`tendon_strain_at_one_norm_force = 0.049` (DGF default; the Rajagopal
Millard muscles use the default tendon curve, no 0.10 Achilles). I kept it.

Intermediates live in `data/opensim/compliant/` (gitignored): per stage
`*_solution.sto`, `*_outputs.sto`, `*_info.json`, `*_study.xml`, `*.log`.
Only stages that converged get npz keys (`c1_force`, `c1_activation`,
`c1_fiber_length_norm`, `c1_fiber_velocity_norm`, `c1_tendon_length`,
`c1_tendon_strain` [T,M], `c1_moment` [T,3], `compliant_mask_c1` [M], same
for c2/c3). The JSON gets a `tendon_compliance` block with settings,
iterations, runtime, status, reserve usage, per-muscle tendon parameters,
peak timing, operating ranges and the stance work split (tendon vs fibre).

What happened on P1 trial 13 (details in the JSON block):

| stage | compliant (both legs) | iterations | runtime | objective | right ankle reserve max |
|---|---|---|---|---|---|
| rigid | 0 | 335 | 18 min | 153.05 | 2.99 N m |
| C1 | 6 | 540 | 58 min | 155.27 | 3.85 N m |
| C2 | 46 | 578 | 63 min | 189.08 | 8.79 N m |
| C3 (+ passive) | 80 | 104 | 85 min | 237.06 | 8.24 N m |

All converged (IPOPT "Optimal Solution Found", tolerances 1e-3) on the
rigid solve's 256-interval mesh. Checks I ran on the side: C1 from
MocoInverse's own cold guess lands on the same optimum (objective 155.271
vs 155.272, right tendon forces 0.43 N RMS apart, 27 N max; 513 iterations).
A 128-interval C3 started as a fallback while the full one crawled (~49 s
per iteration) also converged, but at objective 316.51, so it's a worse
local optimum; it's recorded as `coarse_mesh_run` and not used.

## Step-frequency trials (P1, 1.25 m/s, slots 16-24)

Knee contact (SO + the two-point split above) for P1's nine fixed-speed
walking trials from van der Zee et al. 2022 Table 2 (f* = 0.7/0.8/0.9, three
preferred repeats, 1.1/1.2/1.3), plus 11 synthetic cadences made from slot 19
with `boneid.simulate.step_frequency_trial` (f0 + d steps/min, d = -10..+10).

```sh
MAT=/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p1_5StridesData.mat
for s in 16 17 18 19 20 21 22 23 24; do
  uv run python scripts/opensim/export_inputs.py --mat $MAT --slot $s \
      --tag p1_slot$s --out data/opensim/stepfreq/work
done
for d in -10 -8 -6 -4 -2 0 2 4 6 8 10; do
  uv run python scripts/opensim/export_inputs.py --mat $MAT --slot 19 \
      --syn-d-spm $d --tag $(printf "p1_syn%+d" $d) --out data/opensim/stepfreq/work
done
PY=/Users/jeremy/.julia/conda/3/aarch64/envs/opensim/bin/python
$PY scripts/opensim/stepfreq.py run 16      # ~1.5-4 min each; one process per trial, run them in parallel
$PY scripts/opensim/stepfreq.py run syn-10  # ... syn+0 ... syn+10
$PY scripts/opensim/stepfreq.py collect     # -> data/opensim/p1_stepfreq_knee_contact.{npz,json}
```

- Same subject and markerset, so I don't rescale: every trial uses the
  trial-13 `p1_scaled.osim` (scale + placed markers) and one IK pass with the
  trial-13 final weights. ID, SO and contact are the trial-13 functions with
  their paths re-pointed (`pipeline.TAG`, `T_END`, `WINDOW` default to trial 13).
  SO window is 0.10 s to T_end - 0.10 s.
- Intermediates live in `data/opensim/stepfreq/` (gitignored): `work/` has the
  exports, `slotNN/` and `syn<+d>/` have IK, ID, SO, `wrench.npz`, `info.json`.
- Output `p1_stepfreq_knee_contact.npz`: `slots`, `f_star`, `stride_s`,
  `step_freq_hz` (2 / mean right heel-strike interval), `body_mass` (85.16 kg,
  trial 13's), `d_ic`; per trial `s16_t`, `s16_total`, `_medial`, `_lateral`,
  `_lig_lateral` (BW, NaN outside the SO window), `_fz_r` (N),
  `_stance_frames` [n,2], `_knee_flexion_deg`, `_knee_adduction_moment`
  (external, + = varus, N m; the walker knee has no frontal DOF so this is the
  inertia+gravity+GRF frontal moment at the knee, same as trial 13's
  `ext_frontal_moment`), `_knee_sagittal_moment` (OpenSim ID, + = flexion).
  Synthetic trials use the prefix `syn-10_` ... `syn+0_` ... `syn+10_`, with
  `syn_d_spm`, `syn_k`, `syn_stride_s`, `syn_step_freq_hz`. The JSON has
  per-trial IK RMS (and per marker), stance peaks and impulses, reserve use
  and runtimes.
