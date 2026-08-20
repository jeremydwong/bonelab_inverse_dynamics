# boneid — whole-body inverse dynamics

Takes Visual3D-aligned forces and motion-capture kinematics, produces **net
joint torques**, the **residual wrench** at the top of the body, and an
**energy audit** on every run. Python, managed by
[uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock` pin the whole
environment for reproducibility).

Validated against Visual3D's own computed kinetics across 9 subjects and 502
trial×leg comparisons of the van der Zee, Mundinger & Kuo (2022) walking
dataset: median torque RMSE **ankle 2.5 N·m, knee 3.7 N·m, hip 11.3 N·m**
(the hip gap is dominated by hip-joint-center convention differences, which
the P1 report quantifies directly).

This package replaces the legacy MATLAB pipeline in
`matlab-inverse-dynamics/`, which is kept as reference only. The MATLAB
contained real physics errors — explained slowly and completely in
[What the MATLAB got wrong](#what-the-matlab-got-wrong-and-why) below,
because each one is a lesson worth keeping.

---

## Quick start (the entry points)

```sh
uv sync                                  # create the pinned environment
uv run pytest                            # 81 tests, ~5 s
uv run python examples/quickstart.py     # the canonical 10-step run-through
uv run python -m boneid.report           # simulated squat-to-stand report
uv run python -m boneid.report v3d       # subject-P1 whole-body walkthrough
uv run python -m boneid.validate         # 9-subject validation sweep
```

**`examples/quickstart.py` is the front door.** It is exactly ten steps, one
high-level call per step — the same shape as the old Asics batch script
(study config at the top, a linear pipeline below) with the hardcoding
removed:

1. **Mocap data** — `load_v3d_trial(path, index)`
2. **Force data** — rides on the trial: per-belt force, COP, free moment
3. **Skeleton** — `estimate_body_mass(trial)` + de Leva regressions
4. **Treadmill definition** — `AnalysisParams(...)` + a fixed lab point
5. **Segment kinematics** — `build_chain(trial, side)` per side, `build_upper_body(trial)`
6. **Contact & crossover** — `detect_contact`, `crossover_flags`
7. **Inverse dynamics** — `inverse_dynamics_whole_body(...)`
8. **Outputs** — joint torques + the residual wrench
9. **Energy audit** — `energy_audit_whole_body(...)`, every run
10. **Report** — a self-contained HTML page

Swap the config block for your study; the ten steps do not change.

---

## Data structures

Design rules (Casey Muratori-inspired): **plain dataclasses of numpy arrays,
free functions that operate on them, no member functions, few files.** All SI
units, time-major `[T, ...]` arrays, rotations as `[T, 3, 3]`
world-from-segment matrices. A human should be able to hold the whole package
in their head: `core.py`, `io_v3d.py`, `simulate.py`, `viz.py`, `report.py`,
`validate.py`.

| Dataclass | What it holds |
|---|---|
| `Skeleton` | per-segment mass, COM in the segment frame, inertia tensor about the COM, lengths; chain ordered distal → proximal |
| `SegmentKinematics` | `[T,S,3,3]` rotations + `[T,S,3]` proximal/distal endpoint trajectories |
| `GroundWrench` | external load as force + **moment about a fixed lab point** (the treadmill origin) — no center of pressure anywhere in the pipeline |
| `ForcePlate` | plate corners, origin, orientation in the lab frame |
| `AnalysisParams` | gravity, contact threshold, filter cutoffs, treadmill speed, crossover flagging |
| `InverseDynamicsResult` | `[T,J,3]` joint forces/torques (wrench **on the distal segment by the proximal**), residual wrench at the chain top |
| `TwoLegInverseDynamics` / `WholeBodyInverseDynamics` | both legs into a shared pelvis; the whole-body version adds L5S1, shoulders, elbows as real joints and puts the residual at the torso |
| `EnergyAudit` | KE, PE, per-joint power, ground power, residual power, and the frame-by-frame imbalance `d(KE+PE)/dt − ΣP` |

Two deliberate physics choices:

- **Wrenches about a fixed point, not COPs.** Computing a center of pressure
  divides by vertical force, which explodes at contact edges. Representing
  the ground load as (force, moment-about-the-treadmill-origin) avoids the
  division entirely; `KEY-WRENCH-EQUIV` proves the two representations give
  bit-identical torques when the COP is well defined.
- **The residual and the energy audit are outputs, not afterthoughts.** With
  the full 12-segment body modeled, the residual's mean is ~0 (−1 N vertical
  on subject P1, vs −406 N when the torso was left out) and what remains
  measures soft tissue and marker error. The energy audit is computed by a
  path independent of the torque recursion, so when it closes (0.85% of peak
  power on real data, 10⁻⁶ on simulation) it is genuine evidence, not
  bookkeeping.

---

## What the MATLAB got wrong (and why)

These are not style complaints; each one changes the numbers. Each is now
guarded by a named test.

### 1. The inertia tensor was rotated on one side only

To evaluate Euler's equation in lab coordinates you need the segment's
inertia tensor **in lab coordinates**. The tensor is naturally known in the
segment's own frame (`I_local`, a diagonal matrix for a symmetric segment).
The legacy code converted it as

```matlab
IlegG = multiprod(R_LegTM, Ileg, ...)      % I_world = R * I_local   — wrong
```

Why this can't be right: the inertia tensor is a machine that eats an angular
velocity and returns an angular momentum, `L = I ω`. If `ω` arrives in lab
coordinates and `I_local` only understands body coordinates, you must first
translate the input into body coordinates (`Rᵀ ω`), apply the tensor
(`I_local Rᵀ ω`), and translate the answer back to lab coordinates:

```
L_world = R · I_local · Rᵀ · ω_world      ⟹      I_world = R I_local Rᵀ
```

`R I_local` alone is a machine that eats *body*-frame vectors and emits
*lab*-frame vectors — it is not a tensor in any single frame. You can see the
damage with a rod, `I_local = diag(4, 4, 1)`, rotated 90° about x:

```
correct:  R I Rᵀ = diag(4, 1, 4)      (the long axis now points along y,
                                       so the small moment moves to y — sensible)
legacy:   R I    = [ 4  0  0
                     0  0 -1
                     0  4  0 ]         (asymmetric; no physical body has
                                       an asymmetric inertia tensor)
```

Every real inertia tensor is symmetric with positive eigenvalues; the
one-sided product destroys both properties, and the torque error it produces
depends on segment orientation, so it waxes and wanes within every stride.
Guarded by `test_inertia_world_is_symmetric_similarity` and `KEY-GYRO`.

### 2. Euler's equation was missing its gyroscopic term

The moment balance in `inv3d.m` was

```matlab
Mp = I*aANG - Md - cross(Jd-CM,Rd) - cross(Jp-CM,Rp)     % M = I·α  — incomplete
```

Euler's equation is not `M = I α`. Torque equals the rate of change of
angular momentum, `M = dL/dt` with `L = I_world ω`, and `I_world` itself
changes as the body rotates. The product rule gives two terms:

```
M = d(I ω)/dt = I α  +  ω × (I ω)
```

The second term is easy to feel: spin a body at **constant** ω about an axis
that is not a principal axis. Then α = 0, so the legacy formula says zero
torque is needed — yet the angular momentum vector `L = Iω` is not parallel
to ω, and as the body turns, `L` sweeps around a cone. Changing `L` requires
torque: exactly `ω × (Iω)`. (This is why a tossed tennis racket tumbles about
its intermediate axis.) In near-planar gait ω stays close to one principal
axis and the term is modest; in crossover steps, turning, and any genuinely
3-D movement it is not. The fix costs one line. Guarded by `KEY-GYRO`, which
the legacy formula fails outright.

### 3. The shank's inertia tensor contained the foot's moment

A copy-paste slip in the main script:

```matlab
Ifoot = [Ifx 0 0; 0 Ify 0; 0 0 Ifz];
Ileg  = [Ilx 0 0; 0 Ify 0; 0 0 Ilz];     % middle entry: foot's Ify, not Ily
```

The shank's transverse moment about its long axis was replaced by the
*foot's*. The thigh's `Ity` regression likewise used `A(6)` — foot length —
as its input. Silent, dimension-compatible, wrong for every subject and every
trial processed. In the rewrite, anthropometrics are built in one place
(`deleva_skeleton` / `deleva_upper_body`) from a cited table, and the tensors
are constructed by a loop rather than retyped per segment, so this class of
error cannot recur.

### Honorable mentions

- "Angular acceleration" was the second derivative of Cardan angles, which is
  only the angular acceleration vector for small or planar rotations. The
  rewrite computes ω from `Ṙ Rᵀ` (exact) and differentiates that.
- Ten called functions (`getViconForces_Motek`, `VirtualPoint_v2`, `G2L`,
  `multiprod`, …) are missing from the repository — the MATLAB cannot run
  from what is committed.
- Absolute `Z:\` paths, a ~150-line hardcoded bad-trial table keyed by
  subject number, per-condition shoe heights, and `eval`-generated variables
  for every marker name.

What *was* worth keeping, and is kept: the SVD rigid-body fit from marker
clusters (Söderkvist & Wedin 1993) with its residual as a data-quality
signal, the bottom-up Newton–Euler recursion, threshold-based contact
detection, and joint power as τ · ω_rel.

---

## The model

Two legs `[foot, shank, thigh]` run bottom-up from their measured ground
wrenches into a shared pelvis; **L5S1 is a real joint** to a torso (trunk +
head, de Leva, rigidly lumped — no head markers); the arms
`[forearm+hand, upper arm]` hang from the shoulders (acromion, elbow, wrist
markers — verified empirically to be what they claim). Twelve segments, 100%
of GRF-estimated body mass, eleven joints. The residual wrench lives at the
torso; on good data its mean is ~0 and its excursions are soft tissue, not
missing anatomy.

Anthropometrics are de Leva (1996) throughout, with every geometric
approximation (mid-acromion standing in for suprasternale, sample-mean head
length, markers as arm joint centers) documented at the definition site. An
MTP / multi-segment foot is deliberately **not** offered: this markerset has
no forefoot or hallux markers, and we do not invent joints we cannot measure.

Also in `core.py`: COM power (Donelan, Kram & Kuo 2002 individual-limbs
method), peripheral power (Zelik & Kuo 2010), and the unified-deformable foot
power (Takahashi et al. 2012, per Zelik & Honert 2018) — the last with an
explicit `surface_velocity` argument because foot power is frame-dependent
and a treadmill's lab frame is not its belt frame.

---

## Notebooks & Colab

`notebooks/` holds executable equivalents of the reports; each has the
conversion instruction in its first cell
(`uv run jupyter nbconvert --to script notebooks/<name>.ipynb`).

| Notebook | Needs data? | Colab |
|---|---|---|
| `s2s_report.ipynb` | no — pure simulation | runs as-is once the repo is public |
| `quickstart_report.ipynb` | yes — one `*_5StridesData.mat` | set `DATA_URL` in the first code cell |
| `p1_report.ipynb` | yes — same | set `DATA_URL` in the first code cell |

Each notebook begins with a Colab bootstrap cell that is a no-op locally: on
Colab it pip-installs `boneid` from GitHub and (where needed) downloads the
`.mat` from a URL you paste in (a Dropbox share link with `?dl=1` works).

**To make the Colab badges live** (repo is currently private):

```sh
gh repo edit jeremydwong/bonelab_inverse_dynamics --visibility public --accept-visibility-change-consequences
git push   # if anything is uncommitted
```

Then the badges in each notebook's first cell resolve, e.g.
`https://colab.research.google.com/github/jeremydwong/bonelab_inverse_dynamics/blob/main/notebooks/s2s_report.ipynb`.
If you prefer to keep the repo private: Colab can still open the notebooks
via **File → Open notebook → GitHub** after authorizing your account, but the
`pip install git+https://…` cell will then need a token, and the badge links
will 404 for anyone but you.

---

## Tests

`tests/test_key.py` is the small, intelligible KEY suite — each test guards
one distinct way inverse dynamics can be wrong:

| Test | Guards |
|---|---|
| `KEY-STATIC` | static torques = analytic gravity moments, residual = 0 |
| `KEY-PENDULUM` | analytic pivot torque of a swinging pendulum |
| `KEY-GYRO` | MATLAB bugs 1 & 2: `R I Rᵀ` and `ω×(Iω)` |
| `KEY-WRENCH-EQUIV` | COP path ≡ origin-wrench path |
| `TEST-SIMULATED-S2S` | 3-joint squat-to-stand, large torso: zero residual, analytic static limits, energy balance closing across time |

Broader coverage in `test_coverage.py`, `test_powers.py`, `test_io_v3d.py`,
`test_whole_body.py`, `test_validate.py`, `test_viz.py`. Run: `uv run pytest`.

## Validation data

- Local: `/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p{1..9}_5StridesData.mat`
  — Visual3D exports (120 Hz mocap / 1200 Hz dual-belt forces) of the
  van der Zee, Mundinger & Kuo (2022) dataset: 33 controlled **walking**
  conditions per subject (speed 0.7–2.0 m/s × step length × step width — no
  running exists in this data), including Visual3D's own computed kinetics
  for cross-validation.
- Paper: https://www.nature.com/articles/s41597-022-01817-1
