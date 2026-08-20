# boneid — whole-body inverse dynamics

Python replacement for the legacy MATLAB pipeline in
`matlab-inverse-dynamics/` (kept as reference; see `CLAUDE.md` for its audit,
including the inertia-transform and gyroscopic-term bugs that motivated the
rewrite). Takes Visual3D-aligned forces and kinematics and produces net joint
torques plus the residual wrench at the top of the chain, with an energy
audit on every run.

## Install & run

Everything is managed by [uv](https://docs.astral.sh/uv/) for reproducibility
(`pyproject.toml` + `uv.lock` pin the environment):

```sh
uv sync                          # create the environment
uv run pytest                    # full test suite
uv run pytest tests/test_key.py  # just the KEY suite
uv run python -m boneid.report   # regenerate the TEST-SIMULATED-S2S report
```

## Design

Casey Muratori-inspired: plain dataclasses of numpy arrays, free functions
that operate on them, no member functions, few files. All SI, time-major
`[T, ...]`, rotations are `[T, 3, 3]` world-from-segment matrices. The model
is a serial chain ordered distal → proximal (foot, shank, thigh, pelvis/torso).

| File | Contents |
|---|---|
| `src/boneid/core.py` | Datatypes (`Skeleton`, `SegmentKinematics`, `GroundWrench`, `ForcePlate`, `AnalysisParams`, `InverseDynamicsResult`, `EnergyAudit`) and the math: SVD rigid-body fit, differentiation, filtering, Newton–Euler recursion, energy audit |
| `src/boneid/io_v3d.py` | Visual3D-style inputs: the `*_5StridesData.mat` export format, plate geometry, de Leva anthropometrics, chain construction |
| `src/boneid/simulate.py` | Synthetic motions with exact physics (squat-to-stand, pendulum, gyroscope) |
| `src/boneid/viz.py` | meshcat visualization: skeleton, markers, force arrows, animation, pelvis band + mass-sized torso, URDF-skeleton option, static-HTML export |
| `src/boneid/report.py` | Figures and self-contained HTML reports (S2S, p1 whole-body walkthrough, quickstart) |
| `src/boneid/validate.py` | Cross-subject validation sweep: torque RMSE vs Visual3D over p1–p9 |
| `examples/quickstart.py` | The canonical 10-step run-through, raw data → torques + residual + energy audit |
| `notebooks/` | Executable notebook equivalents of the reports (`nbconvert --to script` to get a .py) |

Two deliberate physics choices:

- **External loads are wrenches about a fixed lab point** (e.g. the treadmill
  origin), not a center of pressure — the COP division is numerically unstable
  at low force. COP conversion helpers exist and `KEY-WRENCH-EQUIV` proves the
  two representations give identical torques.
- **The residual wrench at the top of the chain and a frame-by-frame energy
  audit are first-class outputs.** The audit (d(KE+PE)/dt vs summed wrench
  power) is independent of the recursion, so it is a genuine correctness
  instrument, and it must close even on real data because residual power is
  included.

## Tests

`tests/test_key.py` is the small, intelligible KEY suite — each test guards a
distinct way inverse dynamics can be wrong:

| Test | Guards |
|---|---|
| `KEY-STATIC` | static torques = analytic gravity moments, residual = 0 |
| `KEY-PENDULUM` | analytic pivot torque of a swinging pendulum |
| `KEY-GYRO` | the two legacy bugs: `R I Rᵀ` transform and `ω×(Iω)` |
| `KEY-WRENCH-EQUIV` | COP path ≡ origin-wrench path |
| `TEST-SIMULATED-S2S` | 3-joint squat-to-stand with a large torso: zero residual, analytic static limits, energy balance closing across time |

`tests/test_coverage.py`, `tests/test_io_v3d.py`, `tests/test_viz.py` provide
broader coverage (utilities, the loader against real data, headless viz).

## Validation data

- Local: `/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/p{1..9}_5StridesData.mat`
  — 9 subjects, walk→run treadmill speed sweep, 120 Hz mocap / 1200 Hz forces,
  including Visual3D's own computed joint kinetics for cross-validation.
- Published description: https://www.nature.com/articles/s41597-022-01817-1
