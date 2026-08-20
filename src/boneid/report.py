"""boneid.report — matplotlib figures and HTML reports.

Free functions only. Figures are rendered to base64 PNGs and embedded in a
self-contained HTML page (no external assets except Google Fonts).

Chart conventions (kept consistent across every report):
  ankle #2a78d6 (blue)   knee #eb6834 (orange)   hip #1baf7a
  extra series #eda100 (yellow); neutrals for reference lines.
  One axis per chart, thin marks, faint grid, no top/right spines.

Run `uv run python -m boneid.report` to generate the TEST-SIMULATED-S2S
report at reports/s2s_report.html, or
`uv run python -m boneid.report v3d [matpath] [trial_index]` for the real-data
walkthrough at reports/p1_report.html.
"""

from __future__ import annotations

import base64
import io
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#52514e"
FAINT = "#b9b8b2"
SURFACE = "#fcfcfb"


def style_axes(ax, xlabel="", ylabel=""):
    """House style: recessive frame, faint grid, labels in neutral ink."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(FAINT)
    ax.tick_params(colors=NEUTRAL, labelsize=9)
    ax.grid(True, color=FAINT, alpha=0.35, linewidth=0.6)
    ax.set_xlabel(xlabel, color=NEUTRAL, fontsize=10)
    ax.set_ylabel(ylabel, color=NEUTRAL, fontsize=10)
    ax.set_facecolor(SURFACE)


def new_fig(n_axes=1, height=3.0, width=8.6):
    fig, axes = plt.subplots(1, n_axes, figsize=(width, height),
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def new_grid(rows=1, cols=1, height=3.0, width=8.6, sharex=False, sharey=False):
    """A rows x cols panel grid in the house style (always a 2-D `axes`)."""
    fig, axes = plt.subplots(rows, cols, figsize=(width, height),
                             constrained_layout=True, squeeze=False,
                             sharex=sharex, sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def fig_svg(fig, dpi=150) -> str:
    """Render a figure to an <img>-ready SVG data URI and close it.

    (Emits SVG so report figures stay
    sharp at any zoom. `dpi` only affects the nominal canvas size.)"""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ("data:image/svg+xml;base64,"
            + base64.b64encode(buf.getvalue()).decode())


def figure_block(uri: str, number: int, caption: str) -> str:
    return (f'<figure><img src="{uri}" alt="figure {number}">'
            f'<figcaption><b>Figure {number}.</b> {caption}</figcaption>'
            f"</figure>")


def html_page(title: str, favicon_note: str, body: str) -> str:
    """Wrap report body in the house page style (light-committed: figures are
    baked PNGs, so the page keeps one deliberate paper theme and paints its
    own background explicitly)."""
    return f"""<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {{
    --paper: #f7f6f2; --card: #fcfcfb; --ink: #1c1b18; --ink-2: #52514e;
    --accent: #2a78d6; --rule: #dedcd4; --good: #0a7d43; --bad: #b3261e;
  }}
  html {{ color-scheme: light; }}
  body {{ background: var(--paper); color: var(--ink); margin: 0;
         font: 16px/1.55 "IBM Plex Sans", system-ui, sans-serif; }}
  main {{ max-width: 880px; margin: 0 auto; padding: 48px 24px 96px; }}
  h1, h2 {{ font-family: "Source Serif 4", Georgia, serif;
            text-wrap: balance; line-height: 1.15; }}
  h1 {{ font-size: 2.1rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.35rem; margin: 40px 0 8px; border-top: 1px solid var(--rule);
        padding-top: 24px; }}
  p {{ max-width: 65ch; }}
  .eyebrow {{ font: 600 0.72rem/1 "IBM Plex Mono", monospace; letter-spacing: .14em;
             text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }}
  .subtitle {{ color: var(--ink-2); margin-top: 6px; }}
  figure {{ margin: 20px 0; background: var(--card); border: 1px solid var(--rule);
           border-radius: 6px; padding: 12px 12px 8px; }}
  figure img {{ max-width: 100%; display: block; }}
  figcaption {{ font-size: 0.85rem; color: var(--ink-2); padding: 8px 6px 4px;
               max-width: none; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums;
          font-size: 0.92rem; margin: 16px 0; }}
  th, td {{ text-align: right; padding: 6px 14px; border-bottom: 1px solid var(--rule); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ font-weight: 600; color: var(--ink-2); font-size: 0.8rem;
       text-transform: uppercase; letter-spacing: .06em; }}
  .stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 24px 0; }}
  .stat {{ background: var(--card); border: 1px solid var(--rule); border-radius: 6px;
          padding: 14px 18px; min-width: 160px; flex: 1; }}
  .stat .k {{ font: 500 0.72rem/1.2 "IBM Plex Mono", monospace; letter-spacing: .1em;
             text-transform: uppercase; color: var(--ink-2); }}
  .stat .v {{ font-size: 1.5rem; font-weight: 600; margin-top: 6px;
             font-variant-numeric: tabular-nums; }}
  .stat .u {{ font-size: 0.85rem; color: var(--ink-2); font-weight: 400; }}
  .pass {{ color: var(--good); }} .fail {{ color: var(--bad); }}
  code {{ font-family: "IBM Plex Mono", monospace; font-size: 0.88em;
         background: var(--card); border: 1px solid var(--rule);
         border-radius: 4px; padding: 1px 5px; }}
  iframe.viewer {{ width: 100%; height: 520px; border: 1px solid var(--rule);
                  border-radius: 6px; background: #fff; }}
  .overflow {{ overflow-x: auto; }}
  h3 {{ font-family: "Source Serif 4", Georgia, serif; font-size: 1.08rem;
       margin: 28px 0 6px; }}
  ul {{ max-width: 65ch; }}
  .strips {{ background: var(--card); border: 1px solid var(--rule);
            border-radius: 6px; padding: 12px 12px 8px; margin: 16px 0;
            overflow-x: auto; }}
  .strips svg {{ display: block; min-width: 520px; }}
  .strip-legend {{ font-size: 0.78rem; color: var(--ink-2); padding-top: 6px;
                  line-height: 1.5; }}
</style>
<main>
{body}
</main>
"""


def stat(label: str, value: str, unit: str = "", cls: str = "") -> str:
    return (f'<div class="stat"><div class="k">{label}</div>'
            f'<div class="v {cls}">{value} <span class="u">{unit}</span></div></div>')


# ---------------------------------------------------------------------------
# TEST-SIMULATED-S2S report
# ---------------------------------------------------------------------------

def s2s_report_html(skeleton, kin, ground, idres, audit, truth,
                    viewer_html: str | None = None) -> str:
    """Assemble the full Report 1 HTML for the simulated squat-to-stand."""
    t = kin.t
    joints = skeleton.joint_names
    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    # 1 — the movement
    fig, ax = new_fig(height=2.9)
    for s_i, name in enumerate(skeleton.segment_names):
        ax.plot(t, np.degrees(truth["tilt"][:, s_i]), color=JOINT_COLORS[s_i],
                lw=1.8, label=name)
    style_axes(ax, "time (s)", "segment tilt from vertical (deg)")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
    add(fig, "The prescribed movement: minimum-jerk segment tilts from a deep "
             "squat to standing. Velocities and accelerations vanish at both "
             "endpoints, giving analytic static limits for the torques.")

    # 2 — ground reaction force
    fig, ax = new_fig(height=2.9)
    ax.plot(t, ground.force[:, 2], color=JOINT_COLORS[0], lw=1.8,
            label="vertical")
    ax.plot(t, ground.force[:, 0], color=JOINT_COLORS[1], lw=1.8,
            label="fore-aft")
    ax.axhline(skeleton.mass.sum() * 9.81, color=FAINT, lw=1.0, ls="--")
    ax.annotate("body weight", (t[-1], skeleton.mass.sum() * 9.81),
                ha="right", va="bottom", fontsize=8.5, color=NEUTRAL)
    style_axes(ax, "time (s)", "ground reaction force (N)")
    ax.legend(frameon=False, fontsize=9)
    add(fig, "Simulated ground reaction force, computed exactly from "
             "whole-body Newton–Euler (no force model): unweighting, then a "
             "propulsive overshoot above body weight, settling to standing.")

    # 3 — joint torques with static truth markers
    fig, ax = new_fig(height=3.2)
    for j, name in enumerate(joints):
        ax.plot(t, idres.joint_torque[:, j, 1], color=JOINT_COLORS[j], lw=1.8,
                label=name)
        for frame, key in ((0, "static_torque_start"), (-1, "static_torque_end")):
            ax.plot(t[frame], truth[key][name][1], "o", ms=7, mfc="none",
                    mec=JOINT_COLORS[j], mew=1.6)
    ax.plot([], [], "o", ms=7, mfc="none", mec=NEUTRAL, mew=1.6,
            label="analytic static value")
    style_axes(ax, "time (s)", "sagittal joint torque (N m)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    add(fig, "Net joint torques (sagittal component, acting on the distal "
             "segment) from the inverse-dynamics recursion. Open circles are "
             "the independently derived analytic static torques at the two "
             "endpoints — the recursion lands on them.")

    # 4 — residual wrench
    fig, axes = new_fig(2, height=2.7)
    bw = skeleton.mass.sum() * 9.81
    axes[0].plot(t, idres.residual_force, lw=1.4)
    for line, c in zip(axes[0].lines, JOINT_COLORS[:3]):
        line.set_color(c)
    style_axes(axes[0], "time (s)", "residual force (N)")
    axes[0].legend(["x", "y", "z"], frameon=False, fontsize=8.5)
    axes[1].plot(t, idres.residual_torque, lw=1.4)
    for line, c in zip(axes[1].lines, JOINT_COLORS[:3]):
        line.set_color(c)
    style_axes(axes[1], "time (s)", "residual torque (N m)")
    add(fig, f"Residual wrench at the top of the torso — the force and torque "
             f"unexplained by the model. Peak residual force is "
             f"{np.abs(idres.residual_force[5:-5]).max():.2e} N against a "
             f"body weight of {bw:.0f} N.")

    # 5 — energy components
    fig, ax = new_fig(height=3.0)
    pe0 = audit.potential[0]
    ax.plot(t, audit.kinetic, color=JOINT_COLORS[1], lw=1.8, label="kinetic")
    ax.plot(t, audit.potential - pe0, color=JOINT_COLORS[0], lw=1.8,
            label="potential (rel.)")
    ax.plot(t, audit.kinetic + audit.potential - pe0, color=NEUTRAL, lw=2.2,
            label="total")
    style_axes(ax, "time (s)", "energy (J)")
    ax.legend(frameon=False, fontsize=9)
    add(fig, "Energy across the movement: the body gains potential energy "
             "with a transient kinetic bump mid-rise.")

    # 6 — power balance
    fig, axes = new_fig(2, height=2.9)
    axes[0].plot(t, audit.de_dt, color=NEUTRAL, lw=2.6, alpha=0.45,
                 label="d(KE+PE)/dt")
    axes[0].plot(t, audit.power_total, color=JOINT_COLORS[0], lw=1.2,
                 label="joint + ground + residual power")
    style_axes(axes[0], "time (s)", "power (W)")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].plot(t[5:-5], audit.imbalance[5:-5], color=JOINT_COLORS[1], lw=1.4)
    style_axes(axes[1], "time (s)", "imbalance (W)")
    add(fig, "The energy audit. Left: the rate of change of total mechanical "
             "energy against the summed power of every wrench in the model — "
             "the curves are indistinguishable. Right: their difference "
             "(note the scale), the frame-by-frame energy imbalance.")

    # 7 — joint powers
    fig, ax = new_fig(height=3.0)
    for j, name in enumerate(joints):
        ax.plot(t, audit.joint_power[:, j], color=JOINT_COLORS[j], lw=1.8,
                label=name)
    ax.plot(t, audit.ground_power + audit.residual_power, color=FAINT, lw=1.2,
            label="ground + residual")
    style_axes(ax, "time (s)", "power (W)")
    ax.legend(frameon=False, fontsize=9)
    add(fig, "Joint powers (τ · ω_rel). The hip and knee dominate the work of "
             "raising the large torso; the ground wrench does no work on the "
             "welded foot, and the residual power is negligible.")

    # summary numbers
    interior = slice(5, -5)
    imb = np.abs(audit.imbalance[interior]).max()
    scale = np.abs(audit.de_dt).max()
    res_f = np.abs(idres.residual_force[interior]).max()
    res_t = np.abs(idres.residual_torque[interior]).max()
    tq = np.abs(idres.joint_torque).max()
    work = np.trapezoid(audit.joint_power.sum(axis=1), t)
    de = (audit.kinetic[-1] + audit.potential[-1]
          - audit.kinetic[0] - audit.potential[0])

    stats = "".join([
        stat("peak energy imbalance", f"{imb:.2e}", "W",
             "pass" if imb < 5e-3 * scale else "fail"),
        stat("… relative to peak dE/dt", f"{imb / scale:.1e}", "",
             "pass" if imb < 5e-3 * scale else "fail"),
        stat("peak residual force", f"{res_f:.1e}", f"N (BW {bw:.0f} N)",
             "pass" if res_f < 1e-3 * bw else "fail"),
        stat("peak residual torque", f"{res_t:.1e}", "N m",
             "pass" if res_t < 1e-3 * tq else "fail"),
        stat("∫ joint power dt", f"{work:.2f}", "J"),
        stat("ΔE total", f"{de:.2f}", "J"),
    ])

    viewer = ""
    if viewer_html is not None:
        esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
        viewer = ('<h2>3-D animation</h2><p>Meshcat rendering of the chain '
                  '(large torso, thigh, shank, foot) with the ground-force '
                  'arrow. Use the timeline controls in the viewer, or drag '
                  'to orbit.</p>'
                  f'<iframe class="viewer" srcdoc="{esc}"></iframe>')

    body = f"""
<p class="eyebrow">Report 1 · boneid · TEST-SIMULATED-S2S</p>
<h1>Squat-to-Stand Verification</h1>
<p class="subtitle">Inverse dynamics of a simulated 3-D, 3-joint chain
(foot–shank–thigh–torso; ankle–knee–hip) with an 80 kg body and a large
torso, rising from a deep squat over {t[-1]:.1f} s at {kin.rate:.0f} Hz.</p>

<div class="stat-row">{stats}</div>

<h2>What is being tested</h2>
<p>The motion is prescribed (minimum-jerk segment tilts), the foot is welded
to the ground, and the ground reaction wrench is computed <em>exactly</em>
from whole-body dynamics — so the inverse-dynamics pipeline can be judged
against physics rather than against another implementation. Three independent
instruments: the residual wrench at the top of the chain must vanish, the
joint torques must hit analytic static values at the endpoints, and the
energy audit — d(KE+PE)/dt versus the summed wrench powers, never used inside
the recursion — must close frame by frame.</p>
<p>Forces are represented as a wrench about the lab origin throughout: no
center-of-pressure division anywhere in the pipeline (a COP round-trip is
separately verified equivalent in <code>KEY-WRENCH-EQUIV</code>).</p>
<p>The legacy MATLAB failed here in two ways this test is built to catch:
it transformed inertia one-sidedly (<code>R·I</code> instead of
<code>R·I·Rᵀ</code>) and dropped the gyroscopic term
<code>ω×(Iω)</code> from Euler's equation.</p>

{"".join(figs)}

{viewer}

<h2>How to reproduce</h2>
<p><code>uv run pytest tests/test_key.py</code> runs the KEY suite;
<code>uv run python -m boneid.report</code> regenerates this report from
<code>simulate_squat_to_stand()</code>.</p>
"""
    return html_page("Squat-to-Stand Verification", "s2s", body)


def main_s2s():
    from pathlib import Path

    from .core import AnalysisParams, energy_audit, inverse_dynamics
    from .simulate import simulate_squat_to_stand

    params = AnalysisParams(lowpass_hz=0.0, force_lowpass_hz=0.0)
    skeleton, kin, ground, truth = simulate_squat_to_stand()
    idres = inverse_dynamics(skeleton, kin, ground, params)
    audit = energy_audit(skeleton, kin, ground, idres, params)

    viewer_html = None
    try:
        from . import viz
        vis = viz.start_viewer()
        viz.animate(vis, skeleton, kin, ground=ground, decimate=10,
                    repetitions=10000)
        viewer_html = viz.render_static_html(vis)
        viz.stop_viewer(vis)
    except Exception as exc:  # viz is optional for the report
        print(f"viz skipped: {exc}")

    out = Path(__file__).resolve().parents[2] / "reports"
    out.mkdir(exist_ok=True)
    path = out / "s2s_report.html"
    path.write_text(s2s_report_html(skeleton, kin, ground, idres, audit,
                                    truth, viewer_html))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Real-data report: subject P1 (Visual3D 5-stride export)
# ---------------------------------------------------------------------------
#
# Everything below reads the `p<N>_5StridesData.mat` export through io_v3d and
# validates our inverse dynamics against Visual3D's own kinetics.
#
# FILTERING — where each low-pass happens, once each (this is stated in the
# report too, because double-filtering is the classic silent error here):
#   * force/COP: 1200 Hz analog -> 4th-order zero-lag Butterworth at 50 Hz
#     (anti-alias, inside io_v3d.ground_wrench) -> linear interpolation to the
#     120 Hz mocap clock.  AnalysisParams.force_lowpass_hz is therefore set to
#     0 here, so core.inverse_dynamics does NOT filter the wrench a second time.
#   * markers: the RAW `_pos` targets (not Visual3D's `_pos_proc`) are used, and
#     core.chain_kinematics low-passes segment poses at 12 Hz before
#     differentiating.  That is the only kinematic filter in the chain.

V3D_PATH = ("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
            "p1_5StridesData.mat")
V3D_TRIAL = 13                  #: default trial slot, see `p1_report_html`
SIDES = ("r", "l")              #: both legs, right first (the chain order used
                                #   everywhere: joints [ankle, knee, hip])
EDGE = 10                     #: frames trimmed at both ends of every stat
                                #   (central differences are one-sided there)
JOINT_LABELS = ("R ankle", "R knee", "R hip", "L ankle", "L knee", "L hip")


def v3d_ref_torque(side: str) -> tuple[str, str, str]:
    """Visual3D reference torque names for one leg's [ankle, knee, hip].

    `<side><seg>ProxEndTorque` for seg in Ft/Sk/Th. The right shank is
    exported as `rSh` rather than `rSk` for the ProxEnd* family (and the left
    as `lSh`); `io_v3d.reference_series` absorbs that, so the names below are
    written the regular way on purpose.
    """
    s = side.lower()[0]
    return (f"{s}FtProxEndTorque", f"{s}SkProxEndTorque",
            f"{s}ThProxEndTorque")


def p1_params():
    """AnalysisParams for this export — see the FILTERING note above."""
    from .core import AnalysisParams
    return AnalysisParams(lowpass_hz=12.0, force_lowpass_hz=0.0,
                          contact_threshold_n=20.0)


def belt_speed(trial, threshold_n: float = 20.0) -> float:
    """Treadmill belt speed [m/s] from the stance-phase heel marker.

    During stance the foot is stationary relative to the belt, so the median
    horizontal speed of the calcaneus marker over contact frames is the belt
    speed. Independent of any force scaling — it is pure kinematics.
    """
    from . import io_v3d as io
    from .core import finite_difference
    p = io.plate_for_side(trial, "r", threshold_n)
    mask = io.contact_mask(trial, p, threshold_n)
    v = finite_difference(trial.markers["RCAL"], trial.rate)
    return float(np.median(np.linalg.norm(v[mask][:, :2], axis=1)))


def belt_velocity(trial, threshold_n: float = 20.0) -> np.ndarray:
    """Lab-frame velocity [3] of the treadmill belt surface, MEASURED.

    Direction and speed both come from the data, never from the lab-frame
    convention: during stance the foot is carried by the belt, so the median
    horizontal velocity VECTOR of the calcaneus marker over contact frames is
    the belt velocity. On p1 that comes out as +y (about +1.7 m/s at trial 13)
    with an x component under 1% of it — i.e. the belt carries the foot
    posteriorly, which is the same statement as `io_v3d`'s "the subject faces
    −y", arrived at from the marker trajectories instead of assumed. The
    vertical component is dropped (the heel rises during late stance; the belt
    does not).
    """
    from . import io_v3d as io
    from .core import finite_difference
    p = io.plate_for_side(trial, "r", threshold_n)
    mask = io.contact_mask(trial, p, threshold_n)
    v = finite_difference(trial.markers["RCAL"], trial.rate)
    med = np.median(v[mask], axis=0)
    return np.array([med[0], med[1], 0.0])


# ---------------------------------------------------------------------------
# Hip joint centre conventions
# ---------------------------------------------------------------------------
#
# TODO(migrate): `pelvis_frame` and `hip_center` are marker-set geometry and
# belong in io_v3d next to `joint_centers`. They live here only because
# io_v3d.py is owned elsewhere while this report is being written; move them
# and re-export when the two land together.

HJC_METHODS = ("landmark", "bell", "harrington")

HJC_LABELS = {
    "landmark": "landmark (Visual3D HHL/HHR)",
    "bell": "Bell, Pedersen & Brand (1990)",
    "harrington": "Harrington et al. (2007)",
}


def pelvis_frame(trial):
    """The pelvis anatomical frame both HJC regressions are written in.

    Returns `(mid_asis [T,3], e_ant [T,3], e_left [T,3], e_up [T,3], pw, pd)`:

    * origin  = mid-ASIS, the midpoint of RASI and LASI.
    * e_left  = unit(LASI − RASI), the medio-lateral axis (subject's left).
    * e_ant   = the mid-ASIS-minus-SACR direction orthogonalised against
                e_left — anterior.
    * e_up    = e_ant × e_left, superior (right-handed with x = anterior,
                y = left, z = up).
    * pw      = pelvic width, the trial-mean inter-ASIS distance [m].
    * pd      = pelvic depth, the trial-mean mid-ASIS-to-SACR distance
                measured along e_ant [m].

    APPROXIMATION, stated once and repeated in the report: Harrington's PD is
    mid-ASIS to MID-PSIS. This marker set has a single sacral marker (SACR) and
    no PSIS pair, so SACR stands in for mid-PSIS. SACR sits slightly caudal and
    posterior to the PSIS midpoint, so PD is over-estimated by roughly a
    centimetre, which moves the predicted hip centre about 2–3 mm posteriorly
    (0.24 × 10 mm). That is small next to the between-convention spread the
    section reports, but it is an approximation, not an equivalence.
    """
    m = trial.markers
    mid_asis = 0.5 * (m["RASI"] + m["LASI"])
    e_left = _unit_rows(m["LASI"] - m["RASI"])
    ant = mid_asis - m["SACR"]
    e_ant = _unit_rows(ant - e_left * np.sum(ant * e_left, axis=-1,
                                             keepdims=True))
    e_up = np.cross(e_ant, e_left)
    pw = float(np.linalg.norm(m["LASI"] - m["RASI"], axis=1).mean())
    pd = float(np.mean(np.sum(ant * e_ant, axis=1)))
    return mid_asis, e_ant, e_left, e_up, pw, pd


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hip_center(trial, side: str, method: str = "landmark") -> np.ndarray:
    """Hip joint centre [T,3] for one side under one published convention.

    Only conventions this marker set can actually support are offered; nothing
    is invented. `method` is one of `HJC_METHODS`:

    **"landmark"** (the default, and what every other section of this report
    uses) — Visual3D's own exported `HHR`/`HHL` landmark, per frame.

    **"bell"** — Bell, Pedersen & Brand, *A comparison of the accuracy of
    several hip center location prediction methods*, J Biomech 23(6):617–621
    (1990), refining Bell, Brand & Pedersen, *Prediction of hip joint centre
    location from external landmarks*, Hum Mov Sci 8:3–16 (1989). Fixed
    fractions of the inter-ASIS distance PW, no intercepts:

        posterior  0.19·PW      distal  0.30·PW      lateral  0.36·PW

    measured from MID-ASIS, which is how Visual3D's CODA pelvis implements it.
    The 1990 paper writes the medio-lateral term as 0.14·PW MEDIAL to the
    ipsilateral ASIS; since that ASIS is 0.50·PW lateral of mid-ASIS, 0.50 −
    0.14 = 0.36 lateral of mid-ASIS is the same point, and only the origin
    differs. (The 1989 paper gives 0.22·PW posterior; 0.19 is the 1990
    refinement and the value every implementation uses. Only the medio-lateral
    term changes sign between the right and the left hip.)

    **"harrington"** — Harrington, Zavatsky, Lawson, Yuan & Theologis,
    *Prediction of the hip joint centre in adults, children, and patients with
    cerebral palsy based on magnetic resonance imaging*, J Biomech 40:595–602
    (2007), the all-subjects regression, in mm from mid-ASIS in the pelvis
    frame above:

        anterior   x = −0.24·PD − 9.9      (i.e. posterior)
        superior   y = −0.30·PW − 10.9     (i.e. inferior)
        lateral    z = +0.33·PW + 7.3

    with PW the inter-ASIS width and PD the mid-ASIS-to-mid-PSIS depth (see
    `pelvis_frame` for the SACR-for-mid-PSIS approximation).

    This is Harrington's ALL-SUBJECTS SINGLE-PREDICTOR set (the paper also
    gives a "best predictor" set that needs leg length, which this export does
    not carry, so it is not offered).

    Both regressions are static offsets in a per-frame pelvis frame, so the
    returned trajectory tracks the pelvis exactly as the landmark does.

    TWO APPROXIMATIONS, neither of them corrected here: SACR stands in for
    mid-PSIS in Harrington's PD (see `pelvis_frame`), and both regressions were
    fitted to BONY landmarks (MRI for Harrington, radiographs for Bell) while
    our ASIS coordinates are skin markers sitting ~1 cm anterior to the bone.
    Visual3D and pyCGM2 subtract a marker-radius term for exactly this; we do
    not, because the marker diameter is not in the export. Both effects push
    the predicted centre anteriorly by a few mm.
    """
    from . import io_v3d as io
    method = method.lower()
    s = side.upper()[0]
    if method == "landmark":
        return io.joint_centers(trial, side)["hip"]
    mid_asis, e_ant, e_left, e_up, pw, pd = pelvis_frame(trial)
    e_lat = e_left * (-1.0 if s == "R" else 1.0)
    if method == "bell":
        d_ant, d_up, d_lat = -0.19 * pw, -0.30 * pw, 0.36 * pw
    elif method == "harrington":
        d_ant = (-0.24 * (pd * 1000.0) - 9.9) / 1000.0
        d_up = (-0.30 * (pw * 1000.0) - 10.9) / 1000.0
        d_lat = (0.33 * (pw * 1000.0) + 7.3) / 1000.0
    else:
        raise ValueError(f"unknown hip-centre method {method!r}; "
                         f"choose from {HJC_METHODS}")
    return (mid_asis + d_ant * e_ant + d_up * e_up + d_lat * e_lat)


def build_chain_hjc(trial, side: str, method: str = "landmark", params=None):
    """`io_v3d.build_chain` with the hip centre replaced by `hip_center`.

    `build_chain` derives everything from `io_v3d.joint_centers`, which hard-
    codes the HHR/HHL landmark, and takes no hip override — so the chain is
    rebuilt here with the same conventions and one substitution. Changing the
    hip centre changes FOUR things, all of them handled below and none of them
    optional:

      * the thigh's proximal end (`prox_pos[:, 2]`),
      * the pelvis' distal end (mid-hip, the mean of the two new centres),
      * the thigh's long axis, hence its whole segment frame,
      * the measured thigh and pelvis LENGTHS, hence their de Leva masses,
        COM offsets and inertias.

    Everything else — foot, shank, medio-lateral axes, ground wrench, plate
    assignment — is untouched, so the comparison isolates the hip convention.
    """
    from . import io_v3d as io
    from .core import SegmentKinematics
    params = params or p1_params()
    thr = params.contact_threshold_n
    if method == "landmark":
        return io.build_chain(trial, side=side, threshold_n=thr)

    s = side.upper()[0]
    m = trial.markers
    jc = io.joint_centers(trial, side)
    hip = hip_center(trial, side, method)
    mid_hip = 0.5 * (hip_center(trial, "l", method)
                     + hip_center(trial, "r", method))
    sign = 1.0 if s == "R" else -1.0
    ml_ankle = sign * (m[s + "MML"] - m[s + "LML"])
    ml_knee = sign * (m[s + "MEP"] - m[s + "LEP"])
    ml_pelvis = m["LASI"] - m["RASI"]
    r_world = np.stack([
        io.orthonormal_frame(jc["toe"] - jc["heel"], ml_ankle),
        io.orthonormal_frame(jc["knee"] - jc["ankle"], ml_ankle),
        io.orthonormal_frame(hip - jc["knee"], ml_knee),
        io.orthonormal_frame(jc["l5s1"] - mid_hip, ml_pelvis)], axis=1)
    prox = np.stack([jc["ankle"], jc["knee"], hip, jc["l5s1"]], axis=1)
    dist = np.stack([jc["toe"], jc["ankle"], jc["knee"], mid_hip], axis=1)

    heel_local = np.einsum("tji,tj->ti", r_world[:, 0],
                           jc["heel"] - jc["ankle"]).mean(axis=0)
    lengths = np.linalg.norm(dist - prox, axis=2).mean(axis=0)
    skeleton = io.deleva_skeleton(
        body_mass=io.estimate_body_mass(trial, threshold_n=thr),
        foot_length=float(np.linalg.norm(jc["toe"] - jc["heel"],
                                         axis=1).mean()),
        shank_length=float(lengths[1]), thigh_length=float(lengths[2]),
        pelvis_length=float(lengths[3]), foot_heel_local=heel_local)
    kin = SegmentKinematics(t=trial.t.copy(), rate=trial.rate,
                            r_world=r_world, prox_pos=prox, dist_pos=dist)
    ground = io.ground_wrench(trial, io.plate_for_side(trial, side, thr),
                              threshold_n=thr)
    return skeleton, kin, ground


def hjc_offset_mm(trial, method: str, side: str, edge: int = EDGE) -> float:
    """Mean |Δ| from the landmark hip centre, in mm."""
    sl = slice(edge, -edge)
    d = hip_center(trial, side, method) - hip_center(trial, side, "landmark")
    return float(1000.0 * np.linalg.norm(d[sl], axis=1).mean())


def joint_power_tau_omega(skeleton, kin, idres, params=None) -> np.ndarray:
    """[T,3] joint power τ·ω_rel for [ankle, knee, hip] of one leg chain.

    The pure torque·relative-angular-velocity form the literature reports, not
    the full wrench power `core.energy_audit*` uses (which adds the joint-force
    term that vanishes for an ideal joint). Joint j joins segment j (distal) to
    segment j+1 (proximal) of the 4-segment chain, so the hip's proximal
    segment is the pelvis.
    """
    from .core import chain_kinematics
    params = params or p1_params()
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    return np.stack([
        np.sum(idres.joint_torque[:, j]
               * (ck["omega"][:, j] - ck["omega"][:, j + 1]), axis=1)
        for j in range(3)], axis=1)


def hjc_comparison(trial, params=None, edge: int = EDGE) -> dict:
    """Run the whole-body pipeline once per hip-centre convention.

    Returns `{method: {...}}` with, per method and per side, the joint powers
    (τ·ω_rel), the joint reaction force magnitudes, their integrated
    positive/negative work and peaks, plus the offset from the landmark centre.
    """
    from .core import inverse_dynamics_whole_body
    params = params or p1_params()
    sl = slice(edge, -edge)
    out = {}
    for method in HJC_METHODS:
        chain, skel_u, kin_u = whole_body_chain(trial, params, method)
        whole = inverse_dynamics_whole_body(*chain, skel_u, kin_u, params)
        entry = {"offset_mm": {}, "power": {}, "force": {}, "work": {},
                 "peak_power": {}, "peak_force": {}}
        for k, side in enumerate(SIDES):
            skel, kin = chain[3 * k], chain[3 * k + 1]
            res = whole.right if side == "r" else whole.left
            p = joint_power_tau_omega(skel, kin, res, params)
            f = np.linalg.norm(res.joint_force, axis=2)          # [T,3]
            entry["offset_mm"][side] = hjc_offset_mm(trial, method, side, edge)
            entry["power"][side] = p
            entry["force"][side] = f
            entry["peak_power"][side] = np.abs(p[sl]).max(axis=0)
            entry["peak_force"][side] = f[sl].max(axis=0)
            entry["work"][side] = np.stack([
                np.trapezoid(np.clip(p[sl], 0, None), trial.t[sl], axis=0),
                np.trapezoid(np.clip(p[sl], None, 0), trial.t[sl], axis=0)])
        out[method] = entry
    return out


def power_decomposition(trial, chain, skel_u, kin_u, params=None,
                        edge: int = EDGE) -> dict:
    """COM, peripheral and distal-to-foot powers for one trial.

    * `v_com` — whole-body COM velocity by integrating the summed GRF
      (Donelan, Kram & Kuo 2002), and the kinematic COM velocity of the
      12-segment model as an independent check.
    * `com_r`/`com_l` — individual-limbs COM power, one per belt.
    * `peripheral` — Zelik & Kuo's (2010) COM-relative power, here over the
      WHOLE body (all 12 segments) rather than the legs alone.
    * `ud_lab`/`ud_belt` — Takahashi et al.'s (2012) unified-deformable
      distal-to-foot power per side, in the lab frame and in the belt frame.
    """
    from . import io_v3d as io
    from .core import (chain_kinematics, com_power, com_velocity_from_grf,
                       foot_power_ud, peripheral_power)
    params = params or p1_params()
    skel_r, kin_r, ground_r, skel_l, kin_l, ground_l = chain
    body_mass = io.estimate_body_mass(
        trial, threshold_n=params.contact_threshold_n)
    v_com = com_velocity_from_grf([ground_r.force, ground_l.force],
                                  body_mass, kin_r.rate)
    skel_w, kin_w = whole_body_segments(chain, skel_u, kin_u)
    ck = chain_kinematics(skel_w, kin_w, params.lowpass_hz, params.filter_order)
    v_kin = (np.einsum("s,tsi->ti", skel_w.mass, ck["v_com"])
             / skel_w.mass.sum())
    v_kin = v_kin - v_kin.mean(axis=0, keepdims=True)
    belt = belt_velocity(trial, params.contact_threshold_n)
    plane = float(trial.plates[0].corners[:, 2].mean())
    out = {
        "body_mass": body_mass, "v_com": v_com, "v_kin": v_kin, "belt": belt,
        "plane_height": plane,
        "com_r": com_power(ground_r.force, v_com),
        "com_l": com_power(ground_l.force, v_com),
        "peripheral": peripheral_power(skel_w, kin_w, v_com, params),
    }
    for side, skel, kin, ground in (("r", skel_r, kin_r, ground_r),
                                    ("l", skel_l, kin_l, ground_l)):
        out[f"ud_lab_{side}"] = foot_power_ud(skel, kin, ground, params,
                                              plane_height=plane)
        out[f"ud_belt_{side}"] = foot_power_ud(skel, kin, ground, params,
                                               plane_height=plane,
                                               surface_velocity=belt)
    sl = slice(edge, -edge)
    out["work"] = {k: (float(np.trapezoid(np.clip(v[sl], 0, None), trial.t[sl])),
                       float(np.trapezoid(np.clip(v[sl], None, 0), trial.t[sl])))
                   for k, v in out.items()
                   if isinstance(v, np.ndarray) and v.shape == trial.t.shape}
    return out


def trial_sweep_stats(trials, threshold_n: float = 20.0) -> dict:
    """Per-trial gait descriptors over a whole subject file.

    Returns arrays keyed `index`, `speed`, `stride_time`, `stance_time`,
    `duty`, `peak_fz`, `flight` (fraction of frames with neither plate loaded)
    and `body_mass`, all [N] and ordered as `trials`.
    """
    from . import io_v3d as io
    from .core import detect_contact
    out = {k: [] for k in ("index", "speed", "stride_time", "stance_time",
                           "duty", "peak_fz", "flight", "body_mass")}
    for tr in trials:
        p = io.plate_for_side(tr, "r", threshold_n)
        fz = tr.force[:, p, 2]
        _, events = detect_contact(fz, threshold_n, min_gap=60)
        events = [e for e in events if e[1] - e[0] > 30]
        stance = np.mean([(e[1] - e[0]) / tr.analog_rate for e in events])
        stride = ((events[-1][0] - events[0][0]) / (len(events) - 1)
                  / tr.analog_rate)
        loaded = np.zeros(len(fz), dtype=bool)
        for q in range(tr.force.shape[1]):
            loaded |= detect_contact(tr.force[:, q, 2], threshold_n,
                                     min_gap=60)[0]
        out["index"].append(tr.index)
        out["speed"].append(belt_speed(tr, threshold_n))
        out["stride_time"].append(float(stride))
        out["stance_time"].append(float(stance))
        out["duty"].append(float(stance / stride))
        out["peak_fz"].append(float(fz.max()))
        out["flight"].append(float((~loaded).mean()))
        out["body_mass"].append(io.estimate_body_mass(
            tr, threshold_n=threshold_n))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def torque_rms_vs_v3d(trial, idres, side: str = "r",
                      edge: int = EDGE) -> np.ndarray:
    """[3] RMS of the lab-frame torque VECTOR difference vs Visual3D, N m.

    ankle/knee/hip of one leg against `<side>{Ft,Sk,Th}ProxEndTorque`. The RMS
    is over the 3-vector difference, i.e. sqrt(mean_t |tau_ours - tau_v3d|^2),
    so no component is hidden.
    """
    from . import io_v3d as io
    sl = slice(edge, -edge)
    err = []
    for j, name in enumerate(v3d_ref_torque(side)):
        d = idres.joint_torque[sl, j] - io.reference_series(trial, name)[sl]
        err.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))
    return np.asarray(err)


def reference_torques(trial, side: str) -> list[np.ndarray]:
    """[3] Visual3D reference torque arrays [T,3] for one leg."""
    from . import io_v3d as io
    return [io.reference_series(trial, n) for n in v3d_ref_torque(side)]


def reference_peaks(trial, side: str, edge: int = EDGE) -> np.ndarray:
    """[3] peak |tau| of the Visual3D reference, for scaling the RMS."""
    sl = slice(edge, -edge)
    return np.array([float(np.sqrt((r[sl] ** 2).sum(axis=1)).max())
                     for r in reference_torques(trial, side)])


def two_leg_chain(trial, params=None):
    """(skel_r, kin_r, ground_r, skel_l, kin_l, ground_l) for one trial.

    Both legs come from `io_v3d.build_chain`, which re-derives which plate
    carries which foot per side and per trial. The two 4-segment chains share
    the same pelvis segment (same L5S1 and mid-hip points, same de Leva
    pelvis parameters); `core.inverse_dynamics_two_legs` takes the pelvis from
    the right chain and each hip point from its own chain.
    """
    from . import io_v3d as io
    params = params or p1_params()
    out = []
    for side in SIDES:
        out.extend(io.build_chain(trial, side=side,
                                  threshold_n=params.contact_threshold_n))
    return tuple(out)


def two_leg_rms(trial, two, edge: int = EDGE) -> np.ndarray:
    """[6] RMS torque error vs Visual3D, ordered as `JOINT_LABELS`."""
    return np.concatenate([torque_rms_vs_v3d(trial, two.right, "r", edge),
                           torque_rms_vs_v3d(trial, two.left, "l", edge)])


def cross_trial_rms(trials, params=None, edge: int = EDGE) -> dict:
    """Run the two-leg pipeline on every trial; RMS torque error vs Visual3D.

    Returns `index` [N], `speed` [N], `rms` [N,6] and `peak` [N,6], both
    ordered as `JOINT_LABELS` (r ankle/knee/hip then l ankle/knee/hip), plus
    `residual_fz` [N] (the mean vertical L5S1 residual) and `unmodelled_n` [N]
    (the weight of the body mass this model omits) so the pelvis-residual
    check can be shown across the whole file, not just one trial.
    """
    from . import io_v3d as io
    from .core import inverse_dynamics_two_legs
    params = params or p1_params()
    sl = slice(edge, -edge)
    idx, speed, rms, peak, res_fz, unmod = [], [], [], [], [], []
    for tr in trials:
        chain = two_leg_chain(tr, params)
        two = inverse_dynamics_two_legs(*chain, params)
        idx.append(tr.index)
        speed.append(belt_speed(tr, params.contact_threshold_n))
        rms.append(two_leg_rms(tr, two, edge))
        peak.append(np.concatenate([reference_peaks(tr, s, edge)
                                    for s in SIDES]))
        res_fz.append(float(two.residual_force[sl, 2].mean()))
        unmod.append(unmodelled_weight(tr, chain[0], chain[3], params)[2])
    return {"index": np.asarray(idx, dtype=float),
            "speed": np.asarray(speed), "rms": np.asarray(rms),
            "peak": np.asarray(peak),
            "residual_fz": np.asarray(res_fz),
            "unmodelled_n": np.asarray(unmod)}


def unmodelled_weight(trial, skel_r, skel_l, params=None):
    """(body_mass, modelled_mass, unmodelled_newtons) for the two-leg model.

    The model carries both legs (foot, shank, thigh each side) plus ONE
    pelvis; `skel_l`'s pelvis entry is the same segment as `skel_r`'s and must
    not be counted twice. Everything above L5S1 — trunk, head, both arms — is
    absent, and its weight is what the pelvis residual has to carry.
    """
    from . import io_v3d as io
    params = params or p1_params()
    body_mass = io.estimate_body_mass(
        trial, threshold_n=params.contact_threshold_n)
    modelled = float(skel_r.mass.sum() + skel_l.mass[:3].sum())
    g = float(abs(params.gravity[2]))
    return body_mass, modelled, (body_mass - modelled) * g


# ---------------------------------------------------------------------------
# Whole body: two legs + pelvis + torso(+head) + both arms
# ---------------------------------------------------------------------------

def share_ylim(*axes) -> None:
    """RULE (CLAUDE.md): panels of the SAME quantity share identical y-limits.

    Pass every axis that plots the same joint/quantity for the left and the
    right side (or the same quantity under several conventions); all of them
    end up on the union of their limits. Unequal axes make a bilateral
    comparison meaningless, so this is applied to every such figure in this
    module, not just the torque grid.
    """
    lims = [v for ax in axes for v in ax.get_ylim()]
    for ax in axes:
        ax.set_ylim(min(lims), max(lims))


def whole_body_chain(trial, params=None, hip_method: str = "landmark"):
    """((skel/kin/ground x2), skel_upper, kin_upper) — the 12-segment model.

    The two leg chains are exactly `two_leg_chain`'s (or the hip-convention
    variant, see `build_chain_hjc`); the upper body is one extra call,
    `io_v3d.build_upper_body`, giving [r_forearm+hand, r_upper_arm,
    l_forearm+hand, l_upper_arm, torso(+head)].
    """
    from . import io_v3d as io
    params = params or p1_params()
    if hip_method == "landmark":
        chain = two_leg_chain(trial, params)
    else:
        chain = tuple(x for side in SIDES
                      for x in build_chain_hjc(trial, side, hip_method, params))
    skel_u, kin_u = io.build_upper_body(
        trial, threshold_n=params.contact_threshold_n)
    return chain, skel_u, kin_u


def whole_body_run(trial, params=None, hip_method: str = "landmark"):
    """(chain, skel_u, kin_u, whole, audit) — the whole pipeline, one call."""
    from .core import energy_audit_whole_body, inverse_dynamics_whole_body
    params = params or p1_params()
    chain, skel_u, kin_u = whole_body_chain(trial, params, hip_method)
    whole = inverse_dynamics_whole_body(*chain, skel_u, kin_u, params)
    audit = energy_audit_whole_body(*chain, skel_u, kin_u, whole, params)
    return chain, skel_u, kin_u, whole, audit


def whole_body_segments(chain, skel_u, kin_u):
    """(Skeleton, SegmentKinematics) over all 12 segments, in one flat chain.

    Used for genuinely whole-body quantities — the kinematic COM and
    `core.peripheral_power` — which sum over segments and do not care about the
    joint graph. `joint_names` is therefore empty: this is a BAG of segments,
    not a chain, and must never be handed to `core.inverse_dynamics`. The left
    pelvis entry is dropped (it is the same physical segment as the right's).
    """
    from .core import SegmentKinematics, Skeleton, slice_chain
    skel_r, kin_r, _, skel_l, kin_l, _ = chain
    skel_l3, kin_l3 = slice_chain(skel_l, kin_l, [0, 1, 2])
    parts = [("r_", skel_r, kin_r), ("l_", skel_l3, kin_l3),
             ("", skel_u, kin_u)]
    skel = Skeleton(
        segment_names=[tag + n for tag, s, _ in parts
                       for n in s.segment_names],
        joint_names=[],
        mass=np.concatenate([s.mass for _, s, _ in parts]),
        com_local=np.concatenate([s.com_local for _, s, _ in parts]),
        inertia_local=np.concatenate([s.inertia_local for _, s, _ in parts]),
        length=np.concatenate([s.length for _, s, _ in parts]))
    kin = SegmentKinematics(
        t=kin_r.t, rate=kin_r.rate,
        r_world=np.concatenate([k.r_world for _, _, k in parts], axis=1),
        prox_pos=np.concatenate([k.prox_pos for _, _, k in parts], axis=1),
        dist_pos=np.concatenate([k.dist_pos for _, _, k in parts], axis=1))
    return skel, kin


def contact_spans(mask: np.ndarray, t: np.ndarray):
    """[(t_on, t_off), ...] for the True runs of a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return [(t[a], t[min(b, len(t) - 1)])
            for a, b in zip(np.flatnonzero(edges == 1),
                            np.flatnonzero(edges == -1) - 1)]


def stride_window(mask: np.ndarray, which: int = 1):
    """(start, stop) frame indices of one right-foot stride (onset to onset)."""
    m = np.asarray(mask, dtype=bool)
    onsets = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int)])) == 1)
    if len(onsets) < which + 2:
        which = 0
    return int(onsets[which]), int(onsets[which + 1])


def deleva_table_html(skeleton, body_mass: float, sex: str = "male") -> str:
    """The de Leva (1996) table actually used, plus the scaled values."""
    from .io_v3d import DELEVA_FEMALE, DELEVA_MALE
    table = DELEVA_MALE if sex.lower().startswith("m") else DELEVA_FEMALE
    rows = []
    for s, name in enumerate(skeleton.segment_names):
        # lumped segments (torso = upper+mid trunk+head, forearm+hand) have no
        # single de Leva row; report the realised fraction instead.
        if name in table:
            m_frac, c_frac, rg_s, rg_t, rg_l = table[name]
        else:
            m_frac = skeleton.mass[s] / body_mass
            c_frac = float(np.linalg.norm(skeleton.com_local[s])
                           / max(skeleton.length[s], 1e-9))
            rg_s, rg_t, rg_l = np.sqrt(
                np.diag(skeleton.inertia_local[s])
                / (skeleton.mass[s] * skeleton.length[s] ** 2))
        i_diag = np.diag(skeleton.inertia_local[s])
        rows.append(
            f"<tr><td>{name}</td><td>{m_frac:.4f}</td>"
            f"<td>{skeleton.mass[s]:.2f}</td><td>{skeleton.length[s] * 100:.1f}</td>"
            f"<td>{c_frac:.4f}</td>"
            f"<td>{np.linalg.norm(skeleton.com_local[s]) * 100:.1f}</td>"
            f"<td>{rg_s:.3f} / {rg_t:.3f} / {rg_l:.3f}</td>"
            f"<td>{i_diag[0]:.4f} / {i_diag[1]:.4f} / {i_diag[2]:.4f}</td></tr>")
    return (
        '<div class="overflow"><table><thead><tr>'
        "<th>segment</th><th>mass frac</th><th>mass (kg)</th>"
        "<th>length (cm)</th><th>COM frac</th><th>|COM| from origin (cm)</th>"
        "<th>r<sub>g</sub> sag / tra / long</th>"
        "<th>I<sub>xx</sub> / I<sub>yy</sub> / I<sub>zz</sub> (kg m²)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f"<p>Fractions are de Leva (1996) Table 4 (male) where a segment maps "
        f"to one of its rows; for the lumped segments (torso = upper trunk + "
        f"mid trunk + head, forearm + hand) the REALISED fraction, COM "
        f"position and radii of gyration of the combination are shown instead. "
        f"They multiply the "
        f"measured body mass of {body_mass:.1f} kg and the trial-mean segment "
        f"lengths in column 4. Radii of gyration are fractions of segment "
        f"length; the inertia tensor is diagonal in the segment frame "
        f"(local z = long axis, local y = medio-lateral).</p>")


def p1_report_html(trial, chain, skel_u, kin_u, whole, audit, params, sweep,
                   cross, powers, hjc, viewer_html: str | None = None,
                   urdf_html: str | None = None) -> str:
    """Assemble the step-by-step real-data walkthrough for one V3D trial.

    `chain` is `(skel_r, kin_r, ground_r, skel_l, kin_l, ground_l)`,
    `skel_u`/`kin_u` the upper body, `whole` is
    `core.inverse_dynamics_whole_body` over all twelve segments and `audit` the
    matching `core.energy_audit_whole_body`; `powers` comes from
    `power_decomposition` and `hjc` from `hjc_comparison`.
    """
    from . import io_v3d as io
    from .core import UPPER_BODY_TORSO, chain_kinematics, detect_contact

    skel_r, kin_r, ground_r, skel_l, kin_l, ground_l = chain
    skeleton, kin, ground = skel_r, kin_r, ground_r
    legs = {"r": whole.right, "l": whole.left}
    grounds = {"r": ground_r, "l": ground_l}
    t = kin.t
    joints = skeleton.joint_names            # [ankle, knee, hip]
    edge = EDGE
    sl = slice(edge, -edge)
    body_mass, leg_mass, unmod_n = unmodelled_weight(
        trial, skel_r, skel_l, params)
    upper_mass = float(skel_u.mass.sum())
    upper_n = upper_mass * 9.81
    torso_mass = float(skel_u.mass[UPPER_BODY_TORSO])
    modelled_mass = leg_mass + upper_mass
    bw = body_mass * 9.81
    plate = {s: io.plate_for_side(trial, s, params.contact_threshold_n)
             for s in SIDES}
    fz = io.analog_to_mocap(trial, trial.force[:, :, 2])          # [T,P]
    mask = {s: detect_contact(fz[:, plate[s]], params.contact_threshold_n,
                              min_gap=12)[0] for s in SIDES}
    xover = io.crossover_flags(trial, "r", params.contact_threshold_n)
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz, params.filter_order)
    rms = two_leg_rms(trial, whole, edge)                          # [6]
    refs = {s: reference_torques(trial, s) for s in SIDES}
    ref_peak = np.concatenate([reference_peaks(trial, s, edge) for s in SIDES])
    med = np.median(cross["rms"], axis=0)
    here = int(np.flatnonzero(cross["index"] == trial.index)[0])
    speed = float(cross["speed"][here])
    imb = float(np.abs(audit.imbalance[sl]).max())
    scale = float(np.abs(audit.de_dt[sl]).max())
    # the L5S1 JOINT wrench is what the two-leg model called its residual
    l5_mean = whole.l5s1_force[sl].mean(axis=0)
    l5_gap = float(l5_mean[2] + upper_n)      # ideally zero, see step 7
    res_mean = whole.residual_force[sl].mean(axis=0)
    res_mag = np.linalg.norm(whole.residual_force[sl], axis=1)
    res_peak = float(res_mag.max())
    unmod_frac = 100.0 * (body_mass - modelled_mass) / body_mass
    leg_frac = 100.0 * leg_mass / body_mass

    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    def shade(ax, spans, color, alpha=0.12, label=None):
        for k, (a, b) in enumerate(spans):
            ax.axvspan(a, b, color=color, alpha=alpha, lw=0,
                       label=label if k == 0 else None)

    # ---- 1. the sweep -----------------------------------------------------
    fig, axes = new_fig(2, height=3.0)
    order = np.argsort(sweep["index"])
    x = np.arange(len(order))
    axes[0].plot(x, sweep["speed"][order], "o-", color=JOINT_COLORS[0], lw=1.2,
                 ms=4, label="belt speed (m/s)")
    axes[0].plot(x, sweep["stride_time"][order], "s-", color=JOINT_COLORS[1],
                 lw=1.2, ms=4, label="stride time (s)")
    hit = int(np.flatnonzero(sweep["index"][order] == trial.index)[0])
    axes[0].axvline(hit, color=NEUTRAL, lw=1.0, ls="--")
    axes[0].annotate(f"trial {trial.index}", (hit, axes[0].get_ylim()[1]),
                     ha="center", va="top", fontsize=8.5, color=NEUTRAL)
    axes[0].set_xticks(x[::4])
    axes[0].set_xticklabels([f"{int(i)}" for i in sweep["index"][order][::4]])
    style_axes(axes[0], "trial slot in file", "")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].plot(sweep["speed"], sweep["peak_fz"] / bw, "o", ms=5,
                 color=JOINT_COLORS[2], alpha=0.8)
    axes[1].plot(speed, sweep["peak_fz"][np.flatnonzero(
        sweep["index"] == trial.index)[0]] / bw, "o", ms=10, mfc="none",
        mec=NEUTRAL, mew=1.6)
    style_axes(axes[1], "belt speed (m/s)", "peak vertical GRF (BW)")
    add(fig, f"The speed sweep. {len(sweep['index'])} populated trials of "
             f"subject P1 (120 Hz mocap / 1200 Hz analog, dual-belt "
             f"instrumented treadmill; "
             f'<a href="https://www.nature.com/articles/s41597-022-01817-1">'
             f"dataset description</a>). Belt speed is measured from the "
             f"stance-phase heel marker, not read from a protocol sheet: it "
             f"rises from {sweep['speed'].min():.2f} to "
             f"{sweep['speed'].max():.2f} m/s while stride time falls from "
             f"{sweep['stride_time'].max():.2f} to "
             f"{sweep['stride_time'].min():.2f} s, and peak vertical GRF "
             f"climbs from {sweep['peak_fz'].min() / bw:.2f} to "
             f"{sweep['peak_fz'].max() / bw:.2f} body weights. Trial "
             f"{trial.index} (circled, {speed:.2f} m/s) is the one analysed "
             f"below.")

    # ---- 2. raw forces and contact ---------------------------------------
    span = slice(0, min(len(t), int(4.0 * kin.rate)))
    fig, ax = new_fig(height=3.1)
    shade(ax, contact_spans(mask["r"][span], t[span]), JOINT_COLORS[0], 0.10,
          "right-foot contact")
    shade(ax, contact_spans(mask["l"][span], t[span]), JOINT_COLORS[2], 0.10,
          "left-foot contact")
    ax.plot(t[span], fz[span, plate["r"]], color=JOINT_COLORS[0], lw=1.6,
            label=f"plate {plate['r'] + 1} (right foot)")
    ax.plot(t[span], fz[span, plate["l"]], color=JOINT_COLORS[2], lw=1.6,
            label=f"plate {plate['l'] + 1} (left foot)")
    ax.axhline(params.contact_threshold_n, color=NEUTRAL, lw=1.0, ls="--")
    ax.annotate(f"{params.contact_threshold_n:.0f} N contact threshold",
                (t[span][-1], params.contact_threshold_n), ha="right",
                va="bottom", fontsize=8.5, color=NEUTRAL)
    style_axes(ax, "time (s)", "vertical force (N)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    add(fig, f"Both belts' vertical force over the first "
             f"{t[span][-1]:.1f} s, with the {params.contact_threshold_n:.0f} N "
             f"contact threshold and both feet's detected contacts shaded. "
             f"The double-humped profile and the "
             f"{100 * xover.mean():.0f}% of frames with both belts loaded are "
             f"the signature of walking double support. With <em>both</em> "
             f"legs in the model those frames are no longer a hole in the "
             f"mechanics: each belt's wrench drives its own leg, and the two "
             f"hip wrenches meet at the shared pelvis. No frame in any P1 "
             f"trial has both belts unloaded, i.e. there is no aerial phase "
             f"anywhere in this file.")

    # ---- 3. skeleton from the data ---------------------------------------
    fig, ax = new_fig(height=2.7)
    fz_tot = trial.force[:, :, 2].sum(axis=1)
    ax.plot(trial.t_analog, fz_tot, color=FAINT, lw=0.8, label="Fz plate 1 + 2")
    ax.axhline(bw, color=JOINT_COLORS[0], lw=1.6,
               label=f"mean = {bw:.0f} N → {body_mass:.1f} kg")
    style_axes(ax, "time (s)", "total vertical force (N)")
    ax.legend(frameon=False, fontsize=8.5)
    add(fig, f"Body mass from the data: the summed vertical force of both "
             f"belts averaged over a whole number of strides is "
             f"{bw:.0f} N, i.e. {body_mass:.1f} kg. Across all "
             f"{len(sweep['index'])} trials of this subject the same estimate "
             f"spans {sweep['body_mass'].min():.1f}–"
             f"{sweep['body_mass'].max():.1f} kg, a spread of "
             f"{100 * np.ptp(sweep['body_mass']) / sweep['body_mass'].mean():.1f}% "
             f"— it is a force-plate calibration check as much as an "
             f"anthropometric measurement. All TWELVE modelled segments — two "
             f"legs of three, one pelvis, the torso+head, and two arms — "
             f"account for {modelled_mass:.1f} kg of it "
             f"({100 - unmod_frac:.1f}%); the legs and pelvis alone are only "
             f"{leg_mass:.1f} kg ({leg_frac:.0f}%), and the difference is the "
             f"{upper_mass:.1f} kg of upper body that the previous version of "
             f"this pipeline had to leave as a residual.")

    # ---- 4. kinematics: joint centres -------------------------------------
    fig, ax = new_fig(height=3.0)
    for j, s_i in enumerate((0, 1, 2)):
        ax.plot(t, kin_r.prox_pos[:, s_i, 2], color=JOINT_COLORS[j], lw=1.5,
                label=f"R {joints[j]}")
        ax.plot(t, kin_l.prox_pos[:, s_i, 2], color=JOINT_COLORS[j], lw=1.1,
                ls="--", alpha=0.85, label=f"L {joints[j]}")
    shade(ax, contact_spans(mask["r"], t), NEUTRAL, 0.07, "right stance")
    style_axes(ax, "time (s)", "joint-centre height (m)")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.22 * float(np.ptp(ax.get_ylim())))
    ax.legend(frameon=False, fontsize=8.0, ncol=4, loc="upper left")
    add(fig, "Both legs' joint centres over the whole trial (right solid, left "
             "dashed): ankle = mid-malleoli, knee = mid-epicondyle, hip = "
             "Visual3D's HHR/HHL landmark. Right stance shaded — the left "
             "traces run half a cycle out of phase, which is the cleanest "
             "confirmation that the two chains really are two different legs "
             "and not the same one built twice. These are the segment "
             "endpoints the recursion uses; no cluster fitting is needed "
             "because the export already carries the anatomical markers each "
             "frame.")

    # ---- 5. kinematics vs Visual3D ---------------------------------------
    w_ref = io.reference_series(trial, "rSkAngVel")
    w_ours = ck["omega"][:, 1]
    fig, axes = new_fig(2, height=3.0)
    axes[0].plot(t, w_ref[:, 0], color=NEUTRAL, lw=2.6, alpha=0.4,
                 label="Visual3D rSkAngVel")
    axes[0].plot(t, w_ours[:, 0], color=JOINT_COLORS[1], lw=1.2, label="ours")
    style_axes(axes[0], "time (s)", "shank ω, lab x (rad/s)")
    axes[0].legend(frameon=False, fontsize=8.5)
    for k, (s_i, key, c) in enumerate(((1, "rSkCGPos", 0), (2, "rThCGPos", 2))):
        ref = io.reference_series(trial, key)
        axes[1].plot(t, ref[:, 2], color=NEUTRAL, lw=2.6, alpha=0.35)
        axes[1].plot(t, ck["com"][:, s_i, 2], color=JOINT_COLORS[c], lw=1.2,
                     label=f"{skeleton.segment_names[s_i]} COM (ours)")
    axes[1].plot([], [], color=NEUTRAL, lw=2.6, alpha=0.35, label="Visual3D CGPos")
    style_axes(axes[1], "time (s)", "COM height (m)")
    axes[1].legend(frameon=False, fontsize=8.5)
    w_rms = float(np.sqrt(((w_ours[sl, 0] - w_ref[sl, 0]) ** 2).mean()))
    w_corr = float(np.corrcoef(w_ours[sl, 0], w_ref[sl, 0])[0, 1])
    com_rms = [1000 * float(np.sqrt(((ck["com"][sl, s_i] -
                                      io.reference_series(trial, key)[sl])
                                     ** 2).sum(axis=1).mean()))
               for s_i, key in ((1, "rSkCGPos"), (2, "rThCGPos"))]
    add(fig, f"Our kinematics against Visual3D's. Left: sagittal (lab-x) "
             f"angular velocity of the right shank — r = {w_corr:.4f}, "
             f"{w_rms:.2f} rad/s RMS against a "
             f"{np.abs(w_ref[sl, 0]).max():.1f} rad/s peak. Right: segment COM "
             f"height from our de Leva model versus Visual3D's own "
             f"<code>CGPos</code> — {com_rms[0]:.0f} mm RMS for the shank and "
             f"{com_rms[1]:.0f} mm for the thigh, the thigh being larger "
             f"because the hip-centre and COM-fraction conventions differ "
             f"slightly. Neither series is used to build the other; the "
             f"agreement is what licenses the kinetic comparison below.")

    # ---- 6. the ground wrenches ------------------------------------------
    fig, axes = new_fig(2, height=2.9)
    lab = ("x", "y", "z")
    for c in range(3):
        axes[0].plot(t, ground_r.force[:, c], color=JOINT_COLORS[c], lw=1.3,
                     label=f"F{lab[c]}")
        axes[0].plot(t, ground_l.force[:, c], color=JOINT_COLORS[c], lw=1.0,
                     ls="--", alpha=0.85)
        axes[1].plot(t, ground_r.moment[:, c], color=JOINT_COLORS[c], lw=1.3,
                     label=f"M{lab[c]}")
        axes[1].plot(t, ground_l.moment[:, c], color=JOINT_COLORS[c], lw=1.0,
                     ls="--", alpha=0.85)
    style_axes(axes[0], "time (s)", "ground force (N)")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=3)
    style_axes(axes[1], "time (s)", "moment about lab origin (N m)")
    axes[1].legend(frameon=False, fontsize=8.5, ncol=3)
    add(fig, "The external load as this pipeline carries it: for each foot a "
             "force and a moment <em>about the fixed lab origin</em> (right "
             "solid, left dashed). There is no centre of pressure anywhere in "
             "the recursion — the COP is a division by vertical force and "
             "blows up at every touchdown and toe-off, which is exactly where "
             "joint torques matter. The COP from the export is used once, to "
             "form these moments at the analog rate while each plate is "
             "loaded, and never again.")

    # ---- 7. joint torques vs Visual3D, both legs -------------------------
    for zoom in (False, True):
        if zoom:
            a, b = stride_window(mask["r"], 1)
            win = slice(a, b + 1)
        else:
            win = slice(None)
        fig, axes = new_grid(2, 3, height=5.0, width=8.6)
        for row, side in enumerate(SIDES):
            for j, name in enumerate(joints):
                ax = axes[row, j]
                ax.plot(t[win], refs[side][j][win, 0], color=NEUTRAL, lw=2.8,
                        alpha=0.4, label="Visual3D")
                ax.plot(t[win], legs[side].joint_torque[win, j, 0],
                        color=JOINT_COLORS[j], lw=1.2, label="ours")
                style_axes(ax, "time (s)" if row == 1 else "",
                           "torque, lab x (N m)" if j == 0 else "")
                ax.set_title(f"{JOINT_LABELS[row * 3 + j]}", color=NEUTRAL,
                             fontsize=10)
                if row == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=8.5)
        # RULE: the same joint on the two sides shares identical y-limits,
        # otherwise the L/R comparison is meaningless.
        for j in range(3):
            lims = axes[0, j].get_ylim() + axes[1, j].get_ylim()
            for row in range(2):
                axes[row, j].set_ylim(min(lims), max(lims))
        if zoom:
            add(fig, f"One right-foot stride ({t[a]:.2f}–{t[b]:.2f} s, "
                     f"{(t[b] - t[a]):.2f} s) at full resolution, both legs. "
                     f"The touchdown and push-off transients — the parts a COP "
                     f"formulation handles worst — track without any special "
                     f"casing, on the swinging leg as well as the stance one.")
        else:
            add(fig, "The result. Our lab-frame net joint torque (dominant "
                     "sagittal component, about lab x) over the whole trial "
                     "against Visual3D's own "
                     "<code>Kinetic_Kinematic.*ProxEndTorque</code>, for all "
                     "six joints. Top row right leg, bottom row left. Vector "
                     "RMS differences: "
                     + ", ".join(f"{JOINT_LABELS[k]} {rms[k]:.2f}"
                                 for k in range(6))
                     + " N m, against reference peaks of "
                     + ", ".join(f"{ref_peak[k]:.0f}" for k in range(6))
                     + " N m. Two independent implementations, two "
                       "anthropometric tables, one physics — and the left leg "
                       "is scored by exactly the same code as the right, with "
                       "nothing but the side letter changed.")

    # ---- 8a. the residual, before and after ------------------------------
    fig, axes = new_grid(1, 2, height=3.0, width=8.8, sharey=True)
    for c in range(3):
        axes[0, 0].plot(t, whole.l5s1_force[:, c], color=JOINT_COLORS[c],
                        lw=1.2, label=f"F{lab[c]}")
        axes[0, 1].plot(t, whole.residual_force[:, c], color=JOINT_COLORS[c],
                        lw=1.2, label=f"F{lab[c]}")
    axes[0, 0].axhline(l5_mean[2], color=JOINT_COLORS[3], lw=1.4)
    axes[0, 0].annotate(f"mean Fz = {l5_mean[2]:.0f} N", (t[0], l5_mean[2]),
                        ha="left", va="top", fontsize=8.5,
                        color=JOINT_COLORS[3])
    axes[0, 1].axhline(res_mean[2], color=JOINT_COLORS[3], lw=1.4)
    axes[0, 1].annotate(f"mean Fz = {res_mean[2]:.1f} N", (t[0], res_mean[2]),
                        ha="left", va="bottom", fontsize=8.5,
                        color=JOINT_COLORS[3])
    axes[0, 0].set_title("legs + pelvis only: residual at L5/S1",
                         color=NEUTRAL, fontsize=10)
    axes[0, 1].set_title("whole body: residual at the torso COM",
                         color=NEUTRAL, fontsize=10)
    for a in (axes[0, 0], axes[0, 1]):
        shade(a, contact_spans(mask["r"], t), NEUTRAL, 0.06)
        style_axes(a, "time (s)", "residual force (N)")
        a.legend(frameon=False, fontsize=8.5, ncol=3)
    # RULE: the same quantity in two panels shares one y-scale — the whole
    # point of the figure is that the right panel is flat by comparison.
    share_ylim(axes[0, 0], axes[0, 1])
    add(fig, f"<b>The headline result.</b> Left: the residual of the previous, "
             f"legs-and-pelvis model — the wrench left over at L5/S1, averaging "
             f"{l5_mean[2]:.0f} N downward because the trunk, head and arms "
             f"were simply absent. Right: the residual of the whole-body model, "
             f"at the torso's centre of mass, on <em>the same axes</em>: it "
             f"averages {res_mean[2]:.1f} N. The missing weight has not been "
             f"cancelled or fitted away — it has been given a body to belong "
             f"to, and the left panel's offset has become a named output, the "
             f"L5/S1 joint wrench of Figure {n[0] + 2}.")

    # ---- 8b. the L5S1 joint wrench, and the cross-trial check ------------
    fig, axes = new_grid(1, 3, height=3.0, width=9.2)
    for c in range(3):
        axes[0, 0].plot(t, whole.l5s1_force[:, c], color=JOINT_COLORS[c],
                        lw=1.2, label=f"F{lab[c]}")
        axes[0, 1].plot(t, whole.l5s1_torque[:, c], color=JOINT_COLORS[c],
                        lw=1.2, label=f"M{lab[c]}")
    axes[0, 0].axhline(-upper_n, color=NEUTRAL, lw=1.4, ls="--")
    axes[0, 0].annotate(f"−(upper-body weight) = {-upper_n:.0f} N",
                        (t[-1], -upper_n), ha="right", va="bottom",
                        fontsize=8.5, color=NEUTRAL)
    style_axes(axes[0, 0], "time (s)", "L5/S1 joint force (N)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, ncol=3)
    style_axes(axes[0, 1], "time (s)", "L5/S1 joint torque (N m)")
    axes[0, 1].legend(frameon=False, fontsize=8.5, ncol=3)
    lo = min(cross["residual_fz"].min(), (-cross["unmodelled_n"]).min())
    hi = max(cross["residual_fz"].max(), (-cross["unmodelled_n"]).max())
    pad = 0.05 * (hi - lo)
    axes[0, 2].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=FAINT,
                    lw=1.2, ls="--", label="y = x")
    axes[0, 2].plot(-cross["unmodelled_n"], cross["residual_fz"], "o", ms=5,
                    color=JOINT_COLORS[2], alpha=0.85)
    axes[0, 2].plot(-unmod_n, l5_mean[2], "o", ms=10, mfc="none", mec=NEUTRAL,
                    mew=1.6)
    style_axes(axes[0, 2], "−(upper-body weight) (N)",
               "mean L5/S1 F$_z$ (N)")
    axes[0, 2].legend(frameon=False, fontsize=8.5)
    add(fig, f"The L5/S1 joint wrench — a new first-class output, not an error "
             f"term. Sign convention: this is the wrench the TORSO exerts ON "
             f"THE PELVIS, so the pelvis is the distal side, mirroring the "
             f"hips. Left: its force, mean F<sub>z</sub> = {l5_mean[2]:.0f} N "
             f"against minus the de Leva weight of the segments above it "
             f"({-upper_n:.0f} N) — they agree to {abs(l5_gap):.0f} N, "
             f"{100 * abs(l5_gap) / abs(upper_n):.1f}%. Middle: the torque, "
             f"swinging with the trunk within each stride but averaging "
             f"({whole.l5s1_torque[sl].mean(axis=0)[0]:.0f}, "
             f"{whole.l5s1_torque[sl].mean(axis=0)[1]:.0f}, "
             f"{whole.l5s1_torque[sl].mean(axis=0)[2]:.0f}) N m. Right: the "
             f"same balance over all {len(cross['index'])} trials (this one "
             f"circled) — every trial within "
             f"{np.abs(cross['residual_fz'] + cross['unmodelled_n']).max():.0f} N "
             f"of the identity line. Nothing in the recursion is told the "
             f"subject's mass; the wrench arriving at the top of the leg "
             f"chains reproduces the weight of everything above it anyway.")

    # ---- 9. energy audit --------------------------------------------------
    fig, axes = new_fig(2, height=2.9)
    axes[0].plot(t[sl], audit.de_dt[sl], color=NEUTRAL, lw=2.8, alpha=0.45,
                 label="d(KE+PE)/dt")
    axes[0].plot(t[sl], audit.power_total[sl], color=JOINT_COLORS[0], lw=1.1,
                 label="11 joints + 2 ground + residual")
    style_axes(axes[0], "time (s)", "power (W)")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].plot(t[sl], audit.imbalance[sl], color=JOINT_COLORS[1], lw=1.2)
    style_axes(axes[1], "time (s)", "imbalance (W)")
    add(fig, f"The energy audit of the whole twelve-segment model. Left: the "
             f"rate of change of its mechanical energy against the summed "
             f"power of every wrench acting on it — eleven joints (six leg "
             f"joints, L5/S1, two shoulders, two elbows), <em>both</em> ground "
             f"wrenches, and the torso residual. Right: the difference. Peak "
             f"imbalance is {imb:.1f} W against a {scale:.0f} W peak in "
             f"d(KE+PE)/dt, {100 * imb / scale:.1f}%. This does not test the "
             f"anthropometrics or the marker set: it tests that the branched "
             f"recursion and the audit describe the same mechanical system.")

    fig, axes = new_grid(1, 2, height=3.0, width=9.0)
    for k in range(6):
        axes[0, 0].plot(t, audit.joint_power[:, k], color=JOINT_COLORS[k % 3],
                        lw=1.5 if k < 3 else 1.1, ls="-" if k < 3 else "--",
                        alpha=0.9, label=JOINT_LABELS[k])
    axes[0, 0].plot(t, audit.ground_power, color=FAINT, lw=1.2, label="ground")
    style_axes(axes[0, 0], "time (s)", "power (W)")
    axes[0, 0].legend(frameon=False, fontsize=8.0, ncol=3)
    upper_labels = ("L5/S1", "R shoulder", "R elbow", "L shoulder", "L elbow")
    for k, name in enumerate(upper_labels):
        axes[0, 1].plot(t, audit.joint_power[:, 6 + k],
                        color=JOINT_COLORS[k % 4],
                        ls="-" if k < 3 else "--", lw=1.2, label=name)
    axes[0, 1].plot(t, audit.residual_power, color=FAINT, lw=1.2,
                    label="torso residual")
    style_axes(axes[0, 1], "time (s)", "power (W)")
    axes[0, 1].legend(frameon=False, fontsize=8.0, ncol=3)
    add(fig, "All eleven joint powers (τ·ω_rel plus the joint-force term). "
             "Left: the six leg joints, right leg solid and left dashed, with "
             "the ground power for scale — the two legs trace the same curve "
             "half a stride apart, which is what an alternating gait should "
             "look like and is not enforced anywhere in the code. Right: the "
             "five upper-body joints, on their own axis because they are an "
             "order of magnitude smaller. Note that the two panels are "
             "different quantities and deliberately do NOT share a scale; the "
             "bilateral pairs within each panel do, by construction.")

    # ---- 10. cross-trial --------------------------------------------------
    fig, axes = new_fig(2, height=3.4)
    order = np.argsort(cross["index"])
    xs = np.arange(len(order))
    for k in range(6):
        right = k < 3
        style = dict(ms=5, color=JOINT_COLORS[k % 3], alpha=0.85,
                     mfc=JOINT_COLORS[k % 3] if right else "none",
                     mew=1.3, ls="none", marker="o" if right else "s")
        axes[0].plot(xs, cross["rms"][order, k], label=JOINT_LABELS[k], **style)
        axes[1].plot(cross["speed"], cross["rms"][:, k], **style)
        axes[0].axhline(med[k], color=JOINT_COLORS[k % 3], lw=0.9,
                        ls="-" if right else ":", alpha=0.55)
    axes[0].axvline(hit, color=NEUTRAL, lw=1.0, ls=":")
    axes[0].set_xticks(xs[::4])
    axes[0].set_xticklabels([f"{int(i)}" for i in cross["index"][order][::4]])
    style_axes(axes[0], "trial slot in file", "torque RMS vs Visual3D (N m)")
    for a in axes:
        a.set_ylim(0.0, 1.25 * float(cross["rms"].max()))
    axes[0].legend(frameon=False, fontsize=8.0, ncol=3, loc="upper left")
    style_axes(axes[1], "belt speed (m/s)", "")
    add(fig, f"The whole two-leg pipeline run over every one of the "
             f"{len(cross['index'])} populated trials — "
             f"{6 * len(cross['index'])} joint comparisons — unattended and "
             f"with identical parameters. Right leg filled circles, left leg "
             f"open squares. Medians: "
             + ", ".join(f"{JOINT_LABELS[k]} {med[k]:.2f}" for k in range(6))
             + f" N m. Left by file slot (the dotted line is the analysed "
               f"trial), right against measured belt speed — error grows with "
               f"speed roughly in proportion to the torques themselves. Ankle "
               f"and knee stay near "
               f"{100 * med[0] / np.median(cross['peak'][:, 0]):.0f}–"
               f"{100 * med[1] / np.median(cross['peak'][:, 1]):.0f}% of peak "
               f"torque; the hips carry the accumulated difference in "
               f"hip-centre definition and thigh inertia, which is where two "
               f"implementations always diverge most. The left leg scores like "
               f"the right, which is the point: no side is special-cased.")

    # ---- 11. whole-body powers -------------------------------------------
    a, b = stride_window(mask["r"], 1)
    win = slice(a, b + 1)
    pw_work = powers["work"]
    fig, axes = new_grid(1, 2, height=3.0, width=9.0)
    for c, name in enumerate(("x (medio-lateral)", "y (fore-aft)",
                              "z (vertical)")):
        axes[0, 0].plot(t[win], powers["v_kin"][win, c], color=NEUTRAL,
                        lw=2.6, alpha=0.35)
        axes[0, 0].plot(t[win], powers["v_com"][win, c], color=JOINT_COLORS[c],
                        lw=1.3, label=name)
    axes[0, 0].plot([], [], color=NEUTRAL, lw=2.6, alpha=0.35,
                    label="kinematic (12 segments)")
    style_axes(axes[0, 0], "time (s)", "COM velocity (m/s)")
    axes[0, 0].legend(frameon=False, fontsize=8.0)
    axes[0, 1].plot(t[win], powers["com_r"][win], color=JOINT_COLORS[0],
                    lw=1.5, label="right limb")
    axes[0, 1].plot(t[win], powers["com_l"][win], color=JOINT_COLORS[2],
                    lw=1.5, label="left limb")
    axes[0, 1].plot(t[win], (powers["com_r"] + powers["com_l"])[win],
                    color=NEUTRAL, lw=1.1, ls="--", label="sum")
    axes[0, 1].plot(t[win], powers["peripheral"][win], color=JOINT_COLORS[3],
                    lw=1.2, label="peripheral (whole body)")
    style_axes(axes[0, 1], "time (s)", "power (W)")
    axes[0, 1].legend(frameon=False, fontsize=8.0)
    v_rms = 100.0 * np.sqrt(((powers["v_com"][sl] - powers["v_kin"][sl]) ** 2)
                            .mean(axis=0))
    add(fig, f"Whole-body power, one stride ({t[a]:.2f}–{t[b]:.2f} s). Left: "
             f"the COM velocity used throughout, obtained by integrating the "
             f"summed ground reaction force and removing the mean "
             f"(<code>core.com_velocity_from_grf</code>; legitimate only over a "
             f"whole number of strides of steady treadmill locomotion, which is "
             f"what this Visual3D crop is), against the independent kinematic "
             f"COM velocity of all twelve segments in grey — "
             f"{v_rms[0]:.1f}/{v_rms[1]:.1f}/{v_rms[2]:.1f} cm/s RMS apart. "
             f"Right: the individual-limbs external power, one force plate each "
             f"(Donelan, Kram &amp; Kuo 2002). The two limbs do simultaneous "
             f"positive and negative work through double support — summing the "
             f"forces first would hide exactly that. Over the trial the right "
             f"limb does {pw_work['com_r'][0]:.0f} J positive and "
             f"{pw_work['com_r'][1]:.0f} J negative external work, the left "
             f"{pw_work['com_l'][0]:.0f} / {pw_work['com_l'][1]:.0f} J. The "
             f"yellow trace is the peripheral (COM-relative) power of Zelik "
             f"&amp; Kuo (2010), here summed over the <b>whole</b> body — arms, "
             f"torso and head included, not just the legs — "
             f"{pw_work['peripheral'][0]:.0f} / "
             f"{pw_work['peripheral'][1]:.0f} J.")

    fig, axes = new_grid(1, 2, height=3.0, width=9.0, sharey=True)
    for k, frame_tag in enumerate(("lab", "belt")):
        for c, side in enumerate(SIDES):
            axes[0, k].plot(t[win], powers[f"ud_{frame_tag}_{side}"][win],
                            color=JOINT_COLORS[0 if side == "r" else 2],
                            lw=1.4, ls="-" if side == "r" else "--",
                            label=f"{side.upper()} foot")
        style_axes(axes[0, k], "time (s)", "distal-to-foot power (W)")
        axes[0, k].legend(frameon=False, fontsize=8.5)
    axes[0, 0].set_title("lab frame", color=NEUTRAL, fontsize=10)
    axes[0, 1].set_title(f"belt frame (belt moves +y at "
                         f"{np.linalg.norm(powers['belt']):.2f} m/s)",
                         color=NEUTRAL, fontsize=10)
    add(fig, f"Distal-to-foot (\"unified deformable\") power — everything below "
             f"the rigid foot segment lumped together: shoe, heel pad, arch and "
             f"every foot joint (Takahashi, Kepple &amp; Stanhope 2012; Zelik "
             f"&amp; Honert 2018). Both panels share one y-axis, and so do the "
             f"two feet within each panel. <b>The frame matters and the lab "
             f"frame is the wrong one on a treadmill.</b> Wrench power is "
             f"Galilean frame-dependent: in the lab the stance foot is dragged "
             f"backwards at belt speed, so the GRF does a large spurious "
             f"F·v<sub>belt</sub> amount of work "
             f"({pw_work['ud_lab_r'][0]:.0f} / {pw_work['ud_lab_r'][1]:.0f} J "
             f"right, {pw_work['ud_lab_l'][0]:.0f} / "
             f"{pw_work['ud_lab_l'][1]:.0f} J left) that overground data simply "
             f"does not have. In the belt frame — the overground-equivalent "
             f"quantity the literature reports — the same measure gives "
             f"{pw_work['ud_belt_r'][0]:.0f} / {pw_work['ud_belt_r'][1]:.0f} J "
             f"and {pw_work['ud_belt_l'][0]:.0f} / "
             f"{pw_work['ud_belt_l'][1]:.0f} J: net negative, the familiar "
             f"soft-tissue and shoe dissipation. The belt velocity is not "
             f"assumed — it is the median stance-phase heel velocity, "
             f"({powers['belt'][0]:+.3f}, {powers['belt'][1]:+.3f}, 0.000) m/s, "
             f"i.e. +y at {np.linalg.norm(powers['belt']):.2f} m/s with a "
             f"medio-lateral component "
             f"{100 * abs(powers['belt'][0]) / np.linalg.norm(powers['belt']):.1f}% "
             f"of it. <b>No MTP split is offered:</b> this marker set has CAL, "
             f"5TH and the malleoli and no forefoot or hallux markers, so a "
             f"hindfoot/forefoot division would be invented, not measured.")

    # ---- 12. hip joint centre conventions --------------------------------
    fig, axes = new_grid(2, 3, height=5.0, width=8.8)
    for row, side in enumerate(SIDES):
        for j, name in enumerate(joints):
            ax = axes[row, j]
            for m_i, method in enumerate(HJC_METHODS):
                ax.plot(t[win], hjc[method]["power"][side][win, j],
                        color=JOINT_COLORS[m_i], lw=1.4 if m_i == 0 else 1.1,
                        ls="-" if m_i == 0 else ("--", ":")[m_i - 1],
                        label=HJC_LABELS[method] if (row == 0 and j == 0)
                        else None)
            ax.set_title(f"{JOINT_LABELS[row * 3 + j]}", color=NEUTRAL,
                         fontsize=10)
            style_axes(ax, "time (s)" if row == 1 else "",
                       "joint power τ·ω (W)" if j == 0 else "")
    for j in range(3):        # RULE: same joint, both sides, identical axes
        share_ylim(axes[0, j], axes[1, j])
    axes[0, 0].legend(frameon=False, fontsize=8.0)
    add(fig, f"Joint power (τ·ω<sub>rel</sub>) over one stride under three hip "
             f"joint-centre conventions, with the whole 12-segment pipeline "
             f"re-run from scratch for each. Top row right leg, bottom left; "
             f"the same joint shares y-limits across sides. The ankle curves "
             f"are identical by construction — the bottom-up recursion reaches "
             f"the ankle before it ever meets a hip — the knee moves a little "
             f"(its torque is unchanged but the thigh's angular velocity is "
             f"not), and the hip moves most.")

    fig, axes = new_grid(1, 2, height=3.0, width=8.8, sharey=True)
    width = 0.26
    for k, side in enumerate(SIDES):
        for m_i, method in enumerate(HJC_METHODS):
            axes[0, k].bar(np.arange(3) + (m_i - 1) * width,
                           hjc[method]["peak_force"][side], width * 0.9,
                           color=JOINT_COLORS[m_i], alpha=0.9,
                           label=HJC_LABELS[method] if k == 0 else None)
        axes[0, k].set_xticks(np.arange(3))
        axes[0, k].set_xticklabels(joints)
        axes[0, k].set_title(f"{'right' if side == 'r' else 'left'} leg",
                             color=NEUTRAL, fontsize=10)
        style_axes(axes[0, k], "", "peak |joint reaction force| (N)"
                   if k == 0 else "")
        axes[0, k].axhline(bw, color=NEUTRAL, lw=1.0, ls="--")
    axes[0, 0].annotate("body weight", (2.4, bw), ha="right", va="bottom",
                        fontsize=8.5, color=NEUTRAL)
    axes[0, 0].legend(frameon=False, fontsize=8.0)
    add(fig, f"Peak net intersegmental (joint reaction) force per joint, per "
             f"convention, both legs on one shared axis. <b>Read the knee row "
             f"carefully.</b> This is the NET force one segment transmits to "
             f"the next — the resultant of bone contact, ligament and every "
             f"muscle crossing the joint, all cancelled into a single vector. "
             f"It is not the bone-on-bone contact force: this model contains no "
             f"muscles, and muscle co-contraction typically makes the true "
             f"tibiofemoral contact force two to three times larger than the "
             f"net force plotted here. For osteoarthritis or atypical "
             f"tibial-fracture work the net force is a lower bound and an "
             f"input to a muscle model, never the joint load itself. The ankle "
             f"and knee bars are identical across conventions to the digit — "
             f"they are computed before the hip enters the recursion — while "
             f"the hip peak moves by up to "
             f"{max(abs(100 * (hjc[m]['peak_force'][s][2] / hjc['landmark']['peak_force'][s][2] - 1)) for m in HJC_METHODS for s in SIDES):.1f}%.")

    stats_rms = "".join([
        stat(f"{JOINT_LABELS[k]} RMS", f"{rms[k]:.2f}",
             f"N m (peak {ref_peak[k]:.0f})",
             "pass" if rms[k] < (0.20 if k % 3 == 2 else 0.08) * ref_peak[k]
             else "fail")
        for k in range(6)])
    stats = "".join([
        stat("whole-body residual Fz", f"{res_mean[2]:.1f}",
             f"N mean (was {l5_mean[2]:.0f} at the pelvis)",
             "pass" if abs(res_mean[2]) < 0.01 * bw else "fail"),
        stat("peak |residual force|", f"{res_peak / bw:.2f}",
             f"BW ({res_peak:.0f} N)"),
        stat("energy imbalance", f"{100 * imb / scale:.1f}",
             f"% of peak dE/dt ({imb:.1f} W)",
             "pass" if imb < 0.05 * scale else "fail"),
        stat("mean L5/S1 joint Fz", f"{l5_mean[2]:.0f}",
             f"N vs {-upper_n:.0f} N upper-body weight",
             "pass" if abs(l5_gap) < 0.1 * abs(upper_n) else "fail"),
    ])

    # --- hip-centre tables
    hjc_offset_rows = "".join(
        f"<tr><td>{HJC_LABELS[m]}</td>"
        f"<td>{hjc[m]['offset_mm']['r']:.1f}</td>"
        f"<td>{hjc[m]['offset_mm']['l']:.1f}</td></tr>"
        for m in HJC_METHODS)
    hjc_offset_table = (
        '<div class="overflow"><table><thead><tr><th>convention</th>'
        "<th>right hip, mean |Δ| (mm)</th>"
        "<th>left hip, mean |Δ| (mm)</th></tr></thead>"
        f"<tbody>{hjc_offset_rows}</tbody></table></div>")

    hjc_work_rows = "".join(
        f"<tr><td>{HJC_LABELS[m]}</td><td>{'RL'[k]}</td>"
        + "".join(f"<td>{hjc[m]['work'][s][0, j]:.0f} / "
                  f"{hjc[m]['work'][s][1, j]:.0f}</td>" for j in range(3))
        + "</tr>"
        for m in HJC_METHODS for k, s in enumerate(SIDES))
    hjc_work_table = (
        '<div class="overflow"><table><thead><tr><th>convention</th>'
        "<th>side</th><th>ankle W+ / W− (J)</th><th>knee W+ / W− (J)</th>"
        f"<th>hip W+ / W− (J)</th></tr></thead>"
        f"<tbody>{hjc_work_rows}</tbody></table></div>")

    # --- viewer, strips, urdf
    n_key, fps, period = viewer_period(kin, VIEWER_DECIMATE)
    strip_stop = (n_key - 1) * VIEWER_DECIMATE
    strip_slice = slice(0, strip_stop + 1)
    strips = strip_charts(
        t[strip_slice],
        [("joint-centre\nheight (m)",
          [(f"{'RL'[k]} {joints[j]}", JOINT_COLORS[j],
            None if k == 0 else "4 3",
            (kin_r if k == 0 else kin_l).prox_pos[strip_slice, j, 2])
           for k in range(2) for j in range(3)]),
         ("sagittal joint\ntorque (N m)",
          [(f"{'RL'[k]} {joints[j]}", JOINT_COLORS[j],
            None if k == 0 else "4 3",
            legs[SIDES[k]].joint_torque[strip_slice, j, 0])
           for k in range(2) for j in range(3)]),
         ("vertical GRF (N)",
          [("R belt", JOINT_COLORS[0], None, ground_r.force[strip_slice, 2]),
           ("L belt", JOINT_COLORS[2], "4 3",
            ground_l.force[strip_slice, 2])])],
        period_s=period, fps=fps, n_frames=n_key)

    viewer = ""
    if viewer_html is not None:
        esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
        viewer = (
            "<h2>Step 9 — the trial, moving</h2>"
            "<p>All twelve modelled segments — both legs (foot, shank, thigh), "
            "the pelvis, the torso+head and both arms — posed from the measured "
            "joint centres, the markers they were built from in red, and "
            "<em>both</em> ground reactions drawn as green arrows from their "
            "equivalent centres of pressure (1 kN per metre). Each arrow "
            "collapses to a stub parked at the treadmill origin while its "
            "belt is unloaded — the centre of pressure is a division by "
            "vertical force and is undefined there — so the "
            f"alternating stance is visible directly. The pelvis is drawn at "
            f"its real width, spanning the two hip centres, and the torso as a "
            f"cylinder sized to displace its actual {torso_mass:.0f} kg at "
            f"1000 kg/m³ over the measured L5/S1-to-acromion distance — it is "
            f"not decoration, it is what that mass looks like. Decimated to "
            f"{fps:.0f} fps, looping. Drag to orbit.</p>"
            f'<iframe class="viewer" srcdoc="{esc}"></iframe>'
            "<h3>Time-locked strips</h3>"
            "<p>Three compact strips on the animation's own time axis: joint-"
            "centre heights (both legs, one panel so the axes are necessarily "
            "identical), sagittal joint torques, and both belts' vertical "
            "force. The orange cursor is <b>synchronised to the looping "
            "animation — the same period, and both start on page load</b>; it "
            "is an independent "
            f"<code>requestAnimationFrame</code> loop of exactly "
            f"{period:.3f} s ({n_key} keyframes at {fps:.0f} fps), not a read "
            "of the viewer's clock, which sits in a sandboxed iframe.</p>"
            f"{strips}")
        if urdf_html is not None:
            esc_u = urdf_html.replace("&", "&amp;").replace('"', "&quot;")
            viewer += (
                "<h3>The same pose, as a skeleton</h3>"
                "<p>An optional alternative look: the vendored "
                "<code>humanSubject01_66dof.urdf</code> (LGPL-2.1) with its "
                "leg, pelvis and trunk links placed at our own segments' "
                "endpoints and uniformly scaled to this subject's measured "
                "segment lengths (<code>viz.draw_urdf_skeleton</code>), in one "
                "posed mid-trial frame. Three things it is honest to say about "
                "it. The limb cylinders in this URDF grow along <b>−Z</b> from "
                "their link frame while the drawing routine aims +Z at the "
                "distal end, so the shanks and thighs are fed a re-anchored "
                "kinematics (<code>report.urdf_kinematics</code>) that puts "
                "the node on the distal joint — otherwise they are drawn on "
                "the wrong side of the knee. The box links (feet, pelvis, "
                "trunk) get explicit <code>scale_map</code> entries, because "
                "the automatic \"scale by the link's own +Z extent\" rule "
                "would divide the feet by the sole thickness. And the foot "
                "boxes are <em>mis-oriented</em>: their long axis is the "
                "link's local X, which nothing in this API can aim, so they "
                "sit crossways. The arms are left out for the same reason "
                "rather than drawn wrong. This is a rendering flourish, not a "
                "model — every number in this report comes from the segment "
                "geometry above, not from this URDF.</p>"
                f'<iframe class="viewer" srcdoc="{esc_u}"></iframe>')

    body = f"""
<p class="eyebrow">Report 2 · boneid · REAL DATA · subject P1</p>
<h1>Subject P1 Inverse Dynamics</h1>
<p class="subtitle">The full pipeline on measured data, <b>whole body: two
legs, a pelvis, the torso and head, and both arms — twelve segments, eleven
joints</b>. {len(sweep['index'])} treadmill trials of one subject
spanning {sweep['speed'].min():.2f}–{sweep['speed'].max():.2f} m/s, 120 Hz
optical mocap and 1200 Hz dual-belt force plates, validated against Visual3D's
own computed kinetics at all six leg joints.</p>

<div class="stat-row">{stats_rms}</div>
<div class="stat-row">{stats}</div>

<h2>What this report is</h2>
<p>The squat-to-stand report checks the mechanics against physics we invented.
This one checks it against reality: real markers with real soft-tissue
artefact, real force plates, a body that is only partly in the model, and a
completely independent implementation — Visual3D — to disagree with. Each
section below is one step of the analysis, in the order the code performs
them.</p>
<p><b>The model is branched, not serial, and it is now the whole body.</b> Each
leg is the four-segment chain foot→shank→thigh→pelvis driven by its own belt's
wrench; the two recursions stop at the hips and the pelvis receives
<em>both</em> hip reactions. Each arm is a two-segment chain with nothing in
the hand. The torso+head is then balanced against all three reactions it
receives — L5/S1 and the two shoulders
(<code>core.inverse_dynamics_whole_body</code>). Two consequences run through
the whole report:</p>
<ul>
<li>The wrench arriving at L5/S1 is no longer a residual but a <b>joint
wrench</b> with a name and a sign convention — the wrench the torso exerts
<em>on the pelvis</em>, mirroring the hips — and it should equal minus the
weight of everything above it on average. Step 7 checks that.</li>
<li>The <b>residual has moved to the torso's centre of mass</b>, and with
every segment of the body now modelled it has nothing structural left to
carry: its mean vertical component drops from {l5_mean[2]:.0f} N to
{res_mean[2]:.1f} N. What is left is the honest whole-model error term —
anthropometry, soft tissue, marker artefact, plate calibration.</li>
</ul>
<p>Visual3D exports no arm or trunk kinetics, so the upper body has <em>no</em>
independent reference to be scored against; the leg comparison in Step 6 is
unchanged, and adding the upper body cannot change a single leg number, because
a bottom-up recursion never looks up.</p>
<p><b>Filtering, stated once.</b> The 1200 Hz force and COP are low-passed at
50 Hz (4th-order zero-lag Butterworth) as an anti-alias step and interpolated
onto the 120 Hz mocap clock inside <code>io_v3d.ground_wrench</code>; because
that has already happened, <code>AnalysisParams.force_lowpass_hz</code> is set
to <b>0</b> here so <code>core.inverse_dynamics</code> does not filter the
wrench a second time. Markers are the raw <code>_pos</code> targets, not
Visual3D's pre-processed <code>_pos_proc</code>, and the only kinematic filter
is the {params.lowpass_hz:.0f} Hz low-pass applied to segment poses inside
<code>core.chain_kinematics</code> before they are differentiated.</p>

<h2>Step 1 — the dataset and the trial</h2>
<p>Subject P1 of the van der Zee, Mundinger &amp; Kuo treadmill dataset
(<a href="https://www.nature.com/articles/s41597-022-01817-1">"A biomechanics
dataset of healthy human walking at various speeds, step lengths and step
widths", Scientific Data, 2022</a>) has 28 populated trial slots out of 33,
each cropped by Visual3D to about five strides. The paper's protocol is 33
experimentally controlled combinations of walking speed (0.7–2.0 m/s), step
length, and step width; the condition labels are not in the export, so the
protocol is re-derived from the data itself (Figure 1).</p>
<p><b>Trial {trial.index} is the one analysed here.</b> It sits in the fast
block of the sweep at {speed:.2f} m/s, holds
{len(t) / kin.rate:.1f} s of clean data with five right-foot contacts, and its
agreement with Visual3D (R ankle {rms[0]:.2f}, R knee {rms[1]:.2f} N m RMS) is
within a few percent of the 28-trial median, so it is representative rather
than flattering. One expectation from the brief did not survive contact with
the data: <b>there is no running trial in this file.</b> Every trial has a duty
factor above {sweep['duty'].min():.2f} and, at the 20 N threshold, not one
frame of any trial has both belts unloaded — the aerial phase that defines
running never occurs. That matches the source paper exactly: this dataset is
walking by design (speed × step length × step width conditions), so the
"sweep" below is read from the data, and same-speed trials that differ are
the step-length/width conditions.</p>

{figs[0]}

<h2>Step 2 — raw force and contact detection</h2>
<p>Contact is a vertical-force threshold at
{params.contact_threshold_n:.0f} N with short gaps bridged
(<code>core.detect_contact</code>). The right foot is on plate
{plate['r'] + 1} and the left on plate {plate['l'] + 1}; that assignment is
re-derived per side and per trial from COP-to-foot-marker proximity rather
than assumed. On this trial {100 * xover.mean():.0f}% of frames are double
support.</p>

{figs[1]}

<h2>Step 3 — the skeleton, measured</h2>
<p>Nothing about the subject is hardcoded. Body mass comes from the mean total
vertical force over a whole number of strides; segment lengths come from the
per-side trial-mean distances between the joint centres; de Leva's (1996)
regression table turns those two into masses, COM offsets and inertia
tensors. The first table is the right leg's; the left is built identically
from the left leg's own measured lengths. The second is the upper body
(<code>io_v3d.build_upper_body</code>): the torso row lumps de Leva's upper
trunk, mid trunk and head into one rigid segment, and each forearm row lumps
the hand into the forearm.</p>

{figs[2]}

{deleva_table_html(skeleton, body_mass)}

{deleva_table_html(skel_u, body_mass)}
<p>Three approximations in the upper body are worth naming, because they are
the price of having one at all: mid-acromion stands in for de Leva's
suprasternale and cervicale (the two real landmarks are within ~3 cm of it);
the head is placed on the upward extension of the trunk axis at de Leva's
sample-mean head length, since there are no head markers to scale it; and the
arm joint centres <em>are the markers</em> (AC, EP, WR) because this set has no
medial/lateral pair anywhere on the arm to bisect. The arms are 5% of body mass
between them, so the effect of that last one on everything proximal is small —
but arm kinetics from this marker set are indicative, not definitive.</p>

<h2>Step 4 — kinematics</h2>
<p>Segment poses are built directly from anatomical markers: the long axis from
the two joint centres, the medio-lateral axis from the malleoli, epicondyles or
ASIS pair, and a Gram–Schmidt completion. The medio-lateral axis is flipped for
the left leg so both segment frames are anatomically consistent (local y =
subject's left). Angular velocity is extracted from dR/dt·Rᵀ rather than from
Cardan-angle derivatives — the legacy MATLAB used the latter, which is only
valid for small planar rotations.</p>

{figs[3]}
{figs[4]}

<h2>Step 5 — the ground wrenches</h2>

{figs[5]}

<p>One data quirk worth recording: the exported COP plane sits at
z = −0.175 m while the lowest marker in the trial is at
z = +{trial.markers['RCAL'][:, 2].min():.3f} m, so the force-plate and mocap
vertical references in this export differ by about 0.2 m. Visual3D's own
kinetics use the exported COP as-is — forcing the COP to the marker floor
raises the ankle disagreement from {rms[0]:.1f} to about 24 N m — so this
pipeline uses it too, and reproduces Visual3D exactly. Because the load is
carried as a wrench about a fixed point rather than as a COP, that choice is a
single explicit line, not an assumption buried in a division.</p>

<h2>Step 6 — joint torques against Visual3D, both legs</h2>
<p>Visual3D exports its own inverse dynamics as
<code>Kinetic_Kinematic.&lt;seg&gt;ProxEndTorque</code>, in the lab frame, with
the same sign convention (the torque acting on the distal segment), for every
segment of both legs. The segment naming is a trap: the shanks are
<code>rSk</code>/<code>lSk</code> for kinematics but <code>rSh</code>/
<code>lSh</code> for kinetics, which <code>io_v3d.reference_series</code>
hides.</p>

{figs[6]}
{figs[7]}

<h2>Step 7 — the residual, and the L5/S1 joint wrench</h2>
<p>This is the section the whole-body model exists for. Before it, the model
stopped at the pelvis and everything above L5/S1 was missing; the recursion's
leftover wrench had to carry that missing weight, so its mean vertical
component sat at <b>{l5_mean[2]:.0f} N</b> — recognisably minus the weight of a
trunk, a head and two arms, and a perfectly good validation, but also a
permanent {abs(l5_mean[2]) / bw:.0%}-of-body-weight offset sitting in the
model's error term where real errors should live.</p>

{figs[8]}

<p>With the upper body present the same number becomes <b>{res_mean[2]:.1f} N
at the torso's centre of mass</b>. Nothing was tuned, fitted or subtracted: the
missing weight was given a body. Two distinct quantities now exist where there
was one:</p>
<table><tbody>
<tr><td>measured body mass</td><td>{body_mass:.2f} kg</td></tr>
<tr><td>legs + pelvis (7 segments)</td>
    <td>{leg_mass:.2f} kg ({leg_frac:.0f}%)</td></tr>
<tr><td>torso + head + arms (5 segments)</td>
    <td>{upper_mass:.2f} kg ({100 - leg_frac:.0f}%)</td></tr>
<tr><td>unmodelled mass</td><td>{body_mass - modelled_mass:.2f} kg</td></tr>
<tr><td>−(upper-body weight)</td><td>{-upper_n:.1f} N</td></tr>
<tr><td><b>mean L5/S1 joint F<sub>z</sub></b> (torso on pelvis)</td>
    <td><b>{l5_mean[2]:.1f} N</b> — difference {l5_gap:.1f} N,
    {100 * abs(l5_gap) / abs(upper_n):.1f}% of the upper-body weight</td></tr>
<tr><td><b>mean whole-body residual F<sub>z</sub></b> (at the torso COM)</td>
    <td><b>{res_mean[2]:.2f} N</b> = {abs(res_mean[2]) / bw:.4%} of body
    weight</td></tr>
</tbody></table>

{figs[9]}

<p>The L5/S1 wrench is a real output with a predictable mean, and the closure
test that used to justify the residual now justifies it: nothing in the
recursion is told the subject's total mass, yet the wrench arriving at the top
of the leg chains lands on the de Leva weight of the segments above it to
{100 * abs(l5_gap) / abs(upper_n):.1f}%.</p>
<p>The residual that remains is the honest one. Its mean horizontal components
are {res_mean[0]:.2f} and {res_mean[1]:.2f} N; its peak is
{res_peak:.0f} N = {res_peak / bw:.2f} BW, and that peak is not small. It is
where soft-tissue artefact, the mismatch between a {params.lowpass_hz:.0f} Hz
kinematic filter and a 50 Hz force filter, de Leva's population regressions
applied to one particular body, the marker-for-joint-centre arm model and the
rigid single-segment torso all end up. A residual that is near zero on average
and non-trivial instant to instant is exactly what an honest whole-body model
should produce; a residual that were zero everywhere would mean something had
been fitted.</p>

<h2>Step 8 — the energy audit</h2>

{figs[10]}
{figs[11]}

{viewer}

<h2>Step 10 — whole-body power: COM, peripheral, and below the foot</h2>
<p>Inverse dynamics answers "what torque"; these three measures answer "where
does the energy go". They are computed from the same data and none of them
feeds the recursion.</p>
<p><b>COM (external) power, individual limbs.</b> Donelan, Kram &amp; Kuo
(2002, <i>J Biomech</i> 35:117–124) showed that the work rate on the body's
centre of mass must be evaluated <em>per limb</em> — each force plate's own
force dotted with the whole-body COM velocity — because in double support the
two limbs do simultaneous positive and negative work on the COM that summing
the forces first would cancel. The COM velocity comes from integrating the
summed GRF (<code>core.com_velocity_from_grf</code>), with the integration
constant fixed by removing the mean, which is only legitimate over a whole
number of strides of steady locomotion — the Visual3D crop is exactly that.</p>
<p><b>Peripheral power.</b> Koenig's decomposition splits mechanical energy
into a COM part and the motion of the segments relative to the COM; the second
is Zelik &amp; Kuo's peripheral term (2010, <i>J Exp Biol</i> 213:4257–4264).
Every previous version of this figure could only sum it over the legs and
pelvis. <b>Here it runs over all twelve segments</b> — swinging arms and a
moving trunk included — which is the whole-body quantity Zelik &amp; Kuo
actually define.</p>

{figs[13]}

<p><b>Distal-to-foot power.</b> Takahashi, Kepple &amp; Stanhope (2012,
<i>J Biomech</i> 45:2662–2667) unified-deformable segment power: the ground
wrench's power on the rigid foot segment, which is everything the structures
below that segment — shoe, heel pad, arch, all foot joints — deliver or
absorb. Zelik &amp; Honert (2018, <i>J Biomech</i> 75:1–12) argue this is the
measure to report rather than ankle power alone, because "ankle power" silently
attributes foot deformation to the ankle joint.</p>

{figs[14]}

<p><b>What cannot be done here.</b> Splitting the foot at the metatarso-
phalangeal joint into hindfoot and forefoot — the multi-segment foot analysis
that would separate the arch from the toes — needs forefoot and hallux markers.
This set has the calcaneus, the 5th metatarsal head and the malleoli: one rigid
foot and nothing more. So no MTP joint is offered and the distal-to-foot power
above lumps every sub-foot structure together. Inventing an MTP axis from three
markers would produce a curve, not a measurement.</p>

<h2>Step 11 — hip joint centre conventions</h2>
<p>Every number in this report so far uses one hip joint centre: Visual3D's
exported <code>HHR</code>/<code>HHL</code> landmark. It is a choice, and it is
the choice the hip torques are most sensitive to. Two published regressions
this marker set can actually evaluate are compared against it — and only two,
because a convention needing landmarks we do not have would be invention:</p>
<ul>
<li><b>Bell, Pedersen &amp; Brand (1990)</b>, <i>J Biomech</i> 23(6):617–621,
refining Bell, Brand &amp; Pedersen, <i>Hum Mov Sci</i> 8:3–16 (1989): fixed
fractions of the inter-ASIS width PW in the pelvis anatomical frame —
0.19·PW posterior, 0.30·PW distal, 0.36·PW lateral of mid-ASIS. (The 1990 paper
writes the lateral term as 0.14·PW <em>medial to the ipsilateral ASIS</em>,
which is the same point measured from a different origin; Visual3D's CODA
pelvis uses the mid-ASIS form.) Here PW = {1000 * pelvis_frame(trial)[4]:.0f}
mm.</li>
<li><b>Harrington et al. (2007)</b>, <i>J Biomech</i> 40:595–602, all-subjects
single-predictor MRI regression, in mm from mid-ASIS: <code>x = −0.24·PD −
9.9</code> (posterior), <code>y = −0.30·PW − 10.9</code> (inferior),
<code>z = 0.33·PW + 7.3</code> (lateral), with PD the mid-ASIS-to-mid-PSIS
depth. Here PD = {1000 * pelvis_frame(trial)[5]:.0f} mm.</li>
</ul>
<p><b>Two approximations, stated rather than hidden.</b> This marker set has a
single sacral marker and no PSIS pair, so <b>SACR stands in for mid-PSIS</b> in
Harrington's PD; SACR sits slightly caudal and posterior to the PSIS midpoint,
which over-estimates PD by roughly a centimetre and moves the predicted centre
about 2–3 mm posteriorly. And both regressions were fitted to <b>bony</b>
landmarks (MRI, radiographs) while our ASIS coordinates are skin markers about
a centimetre anterior to the bone; Visual3D and pyCGM2 subtract a marker-radius
term for this, which we cannot because the marker diameter is not in the
export.</p>
<p>How far apart are they, really?</p>

{hjc_offset_table}

<p>Each convention is then run through the <em>entire</em> 12-segment pipeline
from scratch — new thigh frames, new thigh and pelvis lengths, new de Leva
inertias, new whole-body inverse dynamics — so what follows is the propagated
consequence, not a re-labelling.</p>

{figs[15]}

{hjc_work_table}

{figs[16]}

<p>The pattern is worth stating plainly. The ankle is <em>identical</em> to the
digit under all three conventions and so is the knee's reaction force: a
bottom-up recursion computes them from the ground wrench and the foot and shank
kinematics before a hip has been mentioned. What moves is the hip — up to
{max(abs(100 * (hjc[m]['peak_power'][s][2] / hjc['landmark']['peak_power'][s][2] - 1)) for m in HJC_METHODS for s in SIDES):.0f}%
in peak power and {max(abs(100 * (hjc[m]['peak_force'][s][2] / hjc['landmark']['peak_force'][s][2] - 1)) for m in HJC_METHODS for s in SIDES):.1f}%
in peak reaction force — and, more subtly, the knee <em>power</em>, because the
thigh's angular velocity depends on where its proximal end is even though its
torque does not. That asymmetry is the practical answer to "does the hip
convention matter": for ankle and knee kinetics, no; for hip kinetics and any
joint power, yes.</p>

<h2>Step 12 — every trial, both legs</h2>
<p>The previous steps prove one trial. This one proves the pipeline: identical
code, identical parameters, no per-trial tuning, all
{len(cross['index'])} trials of the subject and all six leg joints, scored
against Visual3D.</p>

{figs[12]}

<h2>How to reproduce</h2>
<p><code>uv run python -m boneid.report v3d</code> regenerates this page from
<code>{trial.path}</code>; pass a path and a trial slot to point it elsewhere,
e.g. <code>uv run python -m boneid.report v3d {trial.path} {trial.index}</code>.
<code>uv run pytest</code> runs the KEY suite and the loader tests behind it,
and <code>notebooks/p1_report.ipynb</code> is the same analysis step by step in
a notebook.</p>
"""
    return html_page("Subject P1 Inverse Dynamics", "p1", body)


def viewer_frame_shift(*kins, headroom: float = 0.0) -> np.ndarray:
    """Lab->viewer translation that puts the subject on meshcat's orbit point.

    meshcat's exported static page always orbits the world origin: setting a
    transform on ``/Cameras/default`` moves the eye but not the look-at target
    (verified by screenshot), so a subject standing at lab (x, y) ~ (0, 0.9)
    and z in [0, 1] renders in a corner. The animation is therefore drawn in a
    lab frame TRANSLATED so the subject's bounding box is centred on the
    origin. Nothing but the picture changes — no rotation, no scaling — and the
    ground wrench is transported with it (`shift_wrench`).

    Takes any number of `SegmentKinematics`; the box spans all of them, so
    both legs share one shift and stay in register.
    """
    pts = np.concatenate([a.reshape(-1, 3) for kin in kins
                          for a in (kin.prox_pos, kin.dist_pos)], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return -0.5 * (lo + hi) + np.array([0.0, 0.0, headroom])


def shift_kinematics(kin, shift: np.ndarray):
    """Copy of `kin` translated by `shift` (rotations are untouched)."""
    from dataclasses import replace
    return replace(kin, prox_pos=kin.prox_pos + shift,
                   dist_pos=kin.dist_pos + shift)


def shift_wrench(ground, shift: np.ndarray):
    """Copy of `ground` re-expressed in coordinates translated by `shift`.

    A point at lab x sits at x + shift in the new frame, so the wrench's
    reference point simply moves with it: the moment ABOUT THAT POINT is
    unchanged (a moment about a material point does not care what the
    coordinates are called), and the force is frame-independent. Only the
    stored `point` shifts.
    """
    from dataclasses import replace
    shift = np.asarray(shift, dtype=float)
    return replace(ground, point=np.asarray(ground.point, float) + shift)


VIEWER_DECIMATE = 4             #: 120 Hz / 4 = 30 fps playback
VIEWER_MARKERS = ("RCAL", "R5TH", "RLML", "RMML", "RLEP", "RMEP", "RGTR",
                  "LCAL", "L5TH", "LLML", "LMML", "LLEP", "LMEP", "LGTR",
                  "RASI", "LASI", "SACR", "RAC", "LAC", "REP", "LEP",
                  "RWR", "LWR")


def viewer_period(kin, decimate: int = VIEWER_DECIMATE) -> tuple[int, float, float]:
    """(n_keyframes, fps, period_s) of the meshcat clip `p1_viewer_html` sends.

    `viz.animate` keys frames `0, decimate, 2*decimate, ...` and stamps the
    clip at `fps = rate / decimate`. three.js converts a clip's keyframe index
    k to time k/fps and — since meshcat sends no explicit duration — takes the
    clip duration to be the LAST keyframe time, `(n-1)/fps`. `LoopRepeat` then
    cycles over `[0, duration]`. So the loop period is `(n-1)/fps` seconds
    exactly, which is what the time-locked strip cursor uses.
    """
    step = max(1, int(decimate))
    n = len(range(0, int(kin.prox_pos.shape[0]), step))
    fps = max(1.0, float(kin.rate) / step)
    return n, fps, (n - 1) / fps


def p1_viewer_html(trial, chain, skel_u, kin_u, params,
                   decimate: int = VIEWER_DECIMATE,
                   repetitions: int = 10000,
                   markers: tuple[str, ...] = VIEWER_MARKERS) -> str:
    """Meshcat snapshot of the WHOLE body: both legs, a pelvis spanning the
    hips, the mass-sized torso, both arms, both GRF arrows and the markers.

    `chain` is `two_leg_chain`'s tuple and `skel_u`/`kin_u` come from
    `io_v3d.build_upper_body`. Four meshcat roots are used —
    `boneid/{r,l,ar,al}` — because `viz` keys its paths on segment names and
    the two legs (and the two arms) share theirs; without the namespace the
    second of each pair would silently overwrite the first. `viz.animate` is
    called four times, each after the first with `animation=` so every track
    lands in ONE clip set and with `camera=False` so the first call's framing
    survives. The right leg chain carries the shared pelvis segment; the left
    is drawn as its three leg segments only (`core.slice_chain`).

    The body reads as ONE body because of two extra arguments on the first
    call: `pelvis_span=(hip_l, hip_r, l5s1)` draws a real-width pelvis both
    thighs plug into, and `torso=(l5s1, mid_acromion, torso_mass)` draws the
    torso+head at the volume its 33 kg actually occupies. The arms are ordinary
    2-segment chains and get the normal limb girth (`top_girth_fraction =
    GIRTH_FRACTION`), so the upper arm is not drawn as a trunk.

    `decimate=4` on 120 Hz data is 30 fps playback, looping (`viz` defaults to
    `LOOP_REPETITIONS`). The markers are animated by extending the returned
    `Animation` with one transform track each — `viz.draw_markers` only creates
    the spheres and poses them statically. Everything is drawn in the recentred
    frame of `viewer_frame_shift`; meshcat's own grid and axes are hidden
    because they would then sit at an arbitrary height, and our ground box is
    placed at the subject's foot level instead.
    """
    from . import viz
    from .core import UPPER_BODY_ARM, UPPER_BODY_TORSO, slice_chain

    skel_r, kin_r, ground_r, skel_l, kin_l, ground_l = chain
    shift = viewer_frame_shift(kin_r, kin_l, kin_u)
    vkin = {"r": shift_kinematics(kin_r, shift),
            "l": shift_kinematics(kin_l, shift),
            "u": shift_kinematics(kin_u, shift)}
    vground = {"r": shift_wrench(ground_r, shift),
               "l": shift_wrench(ground_l, shift)}
    # left leg without the (shared) pelvis segment
    skel_left3, kin_left3 = slice_chain(skel_l, vkin["l"], [0, 1, 2])
    traj = {name: trial.markers[name] + shift for name in markers
            if name in trial.markers}
    floor = float(min(vkin["r"].dist_pos[:, 0, 2].min(),
                      vkin["l"].dist_pos[:, 0, 2].min()))
    root = {"r": f"{viz.ROOT}/r", "l": f"{viz.ROOT}/l",
            "ar": f"{viz.ROOT}/ar", "al": f"{viz.ROOT}/al"}
    pelvis_span = (vkin["l"].prox_pos[:, 2], vkin["r"].prox_pos[:, 2],
                   vkin["r"].prox_pos[:, 3])
    torso = (vkin["u"].prox_pos[:, UPPER_BODY_TORSO],
             vkin["u"].dist_pos[:, UPPER_BODY_TORSO],
             float(skel_u.mass[UPPER_BODY_TORSO]))

    vis = viz.start_viewer()
    try:
        animation = viz.animate(vis, skel_r, vkin["r"], ground=vground["r"],
                                decimate=decimate, repetitions=1, play=False,
                                plane_height=floor, root=root["r"],
                                pelvis_span=pelvis_span, torso=torso)
        viz.animate(vis, skel_left3, kin_left3, ground=vground["l"],
                    decimate=decimate, repetitions=1, play=False,
                    plane_height=floor, root=root["l"], animation=animation,
                    camera=False,
                    top_girth_fraction=viz.GIRTH_FRACTION)
        for side in ("r", "l"):
            arm_skel, arm_kin = slice_chain(skel_u, vkin["u"],
                                            UPPER_BODY_ARM[side])
            viz.animate(vis, arm_skel, arm_kin, decimate=decimate,
                        repetitions=1, play=False, plane_height=floor,
                        root=root["a" + side], animation=animation,
                        camera=False,
                        top_girth_fraction=viz.GIRTH_FRACTION)
        viz.draw_markers(vis, traj, 0)
        step = max(1, int(decimate))
        for k, f in enumerate(range(0, len(kin_r.t), step)):
            with animation.at_frame(vis, k) as frame:
                for name, xyz in traj.items():
                    frame[f"{viz.ROOT}/markers/{name}"].set_transform(
                        viz.translation(xyz[f]))
        for tag in ("r", "l"):
            viz.draw_ground(vis, size=1.2, root=root[tag])
            vis[f"{root[tag]}/ground"].set_transform(
                viz.translation([0.0, 0.0, floor - 0.005]))
        for tag in ("ar", "al"):
            vis[f"{root[tag]}/ground"].delete()
        vis["/Grid"].set_property("visible", False)
        vis["/Axes"].set_property("visible", False)
        vis.set_animation(animation, play=True, repetitions=repetitions)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0),
                       offset=(1.55, -1.55, 0.35))
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


# ---------------------------------------------------------------------------
# Time-locked strip charts (inline SVG + a rAF cursor)
# ---------------------------------------------------------------------------

STRIP_WIDTH = 860               #: SVG user units across the whole strip block
STRIP_HEIGHT = 96               #: plot height of one strip
STRIP_GAP = 10                  #: vertical gap between strips
STRIP_PAD_L = 128               #: left gutter for the y-axis label + ticks
STRIP_PAD_R = 10


def _strip_path(t, y, x0, x1, ytop, ybot, lo, hi, max_pts: int = 900) -> str:
    """One polyline's `points` attribute, decimated to `max_pts` samples."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    step = max(1, int(np.ceil(len(t) / max_pts)))
    t, y = t[::step], y[::step]
    span = max(hi - lo, 1e-12)
    xs = x0 + (x1 - x0) * (t - t[0]) / max(t[-1] - t[0], 1e-12)
    ys = ybot - (ybot - ytop) * (y - lo) / span
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def strip_charts(t, panels, period_s: float, fps: float,
                 n_frames: int, uid: str = "strip") -> str:
    """Three stacked SVG strips sharing one time axis, with a looping cursor.

    `panels` is a list of `(ylabel, [(label, color, dash, y), ...])`; every
    series in a panel is drawn on that panel's own shared y-scale, which is how
    the bilateral rule is honoured here — the left and right trace of a joint
    live in the SAME panel and therefore literally cannot have different axes.

    The cursor is a single vertical line spanning all three strips, advanced by
    `requestAnimationFrame` with period `period_s` — the meshcat clip's exact
    period, `(n_frames − 1) / fps` (see `viewer_period`). Both the clip and this
    loop start when the page loads, so they run at the same rate from the same
    instant; nothing here reads the viewer's clock, because the viewer is in a
    sandboxed iframe. The caption says so.
    """
    t = np.asarray(t, float)
    x0, x1 = STRIP_PAD_L, STRIP_WIDTH - STRIP_PAD_R
    total_h = len(panels) * (STRIP_HEIGHT + STRIP_GAP) + 26
    out = [f'<svg id="{uid}-svg" viewBox="0 0 {STRIP_WIDTH} {total_h}" '
           f'width="100%" role="img" '
           f'aria-label="time-locked strip charts">']
    legend = []
    for p_i, (ylabel, series) in enumerate(panels):
        ytop = p_i * (STRIP_HEIGHT + STRIP_GAP)
        ybot = ytop + STRIP_HEIGHT
        lo = min(float(np.min(y)) for *_, y in series)
        hi = max(float(np.max(y)) for *_, y in series)
        pad = 0.06 * max(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        out.append(f'<rect x="{x0}" y="{ytop}" width="{x1 - x0}" '
                   f'height="{STRIP_HEIGHT}" fill="{SURFACE}" '
                   f'stroke="{FAINT}" stroke-width="0.8"/>')
        if lo < 0 < hi:
            zy = ybot - (ybot - ytop) * (0 - lo) / (hi - lo)
            out.append(f'<line x1="{x0}" y1="{zy:.1f}" x2="{x1}" '
                       f'y2="{zy:.1f}" stroke="{FAINT}" stroke-width="0.7"/>')
        for label, color, dash, y in series:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline fill="none" stroke="{color}" '
                       f'stroke-width="1.3"{d} points="'
                       f'{_strip_path(t, y, x0, x1, ytop, ybot, lo, hi)}"/>')
        fmt = "{:.2f}" if max(abs(lo), abs(hi)) < 10 else "{:.0f}"
        out.append(f'<text x="{x0 - 5}" y="{ytop + 10}" text-anchor="end" '
                   f'font-size="9.5" fill="{NEUTRAL}">{fmt.format(hi)}</text>')
        out.append(f'<text x="{x0 - 5}" y="{ybot - 1}" text-anchor="end" '
                   f'font-size="9.5" fill="{NEUTRAL}">{fmt.format(lo)}</text>')
        lines = ylabel.split("\n")
        y_mid = 0.5 * (ytop + ybot) - 5.5 * (len(lines) - 1) + 4
        for li, line in enumerate(lines):
            out.append(f'<text x="{x0 - 34}" y="{y_mid + 11 * li}" '
                       f'text-anchor="end" font-size="10.5" fill="{NEUTRAL}" '
                       f'font-weight="600">{line}</text>')
        legend.append(" &nbsp; ".join(
            f'<span style="color:{c}">&#9644;</span> {lab}'
            for lab, c, _, _ in series))
    axis_y = len(panels) * (STRIP_HEIGHT + STRIP_GAP)
    out.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" '
               f'stroke="{FAINT}"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * frac
        out.append(f'<text x="{x:.1f}" y="{axis_y + 14}" text-anchor="middle" '
                   f'font-size="10" fill="{NEUTRAL}">'
                   f'{t[0] + frac * (t[-1] - t[0]):.1f}</text>')
    out.append(f'<text x="{x1}" y="{axis_y + 25}" text-anchor="end" '
               f'font-size="10" fill="{NEUTRAL}">time (s)</text>')
    out.append(f'<line id="{uid}-cursor" x1="{x0}" y1="0" x2="{x0}" '
               f'y2="{axis_y}" stroke="{JOINT_COLORS[1]}" stroke-width="1.6"/>')
    # the cursor's time readout lives in the left gutter, clear of the ticks
    out.append(f'<text id="{uid}-time" x="{x0 - 5}" y="{axis_y + 14}" '
               f'text-anchor="end" font-size="10" '
               f'fill="{JOINT_COLORS[1]}"></text>')
    out.append("</svg>")

    script = f"""<script>
(function() {{
  var cursor = document.getElementById("{uid}-cursor");
  var label  = document.getElementById("{uid}-time");
  var X0 = {x0}, X1 = {x1}, T0 = {t[0]:.6f}, T1 = {t[-1]:.6f};
  var PERIOD = {period_s:.6f};          // (n_frames - 1) / fps, exactly
  var start = null;
  function tick(now) {{
    if (start === null) start = now;
    var u = (((now - start) / 1000.0) % PERIOD) / PERIOD;
    var x = X0 + (X1 - X0) * u;
    cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
    label.textContent = (T0 + (T1 - T0) * u).toFixed(2) + " s";
    window.requestAnimationFrame(tick);
  }}
  window.requestAnimationFrame(tick);
}})();
</script>"""
    legend_html = "<br>".join(f'<div>{row}</div>' for row in legend)
    return (f'<div class="strips">{"".join(out)}'
            f'<div class="strip-legend">{legend_html}</div></div>{script}')


#: URDF links of the vendored humanSubject01 model, mapped onto the 12-segment
#: bag `whole_body_segments` returns (right leg 0-3, left leg 4-6, upper body
#: 7-11 = [r_fa, r_ua, l_fa, l_ua, torso]).
URDF_SEGMENT_MAP = {
    "RightFoot": 0, "RightLowerLeg": 1, "RightUpperLeg": 2, "Pelvis": 3,
    "LeftFoot": 4, "LeftLowerLeg": 5, "LeftUpperLeg": 6, "T8": 11,
}
#: `draw_urdf_skeleton` auto-scales a link by OUR segment length divided by the
#: link's own +Z extent, which is exactly right for the four limb links (they
#: are cylinders growing along +Z) and wrong for every BOX link, whose long
#: axis is not +Z: the feet would be scaled by the sole thickness (×2.3) and
#: the trunk box by its own depth (×3.7, giving a half-metre-wide slab). Those
#: get explicit uniform scales instead — the feet at the template model's own
#: size, the trunk enlarged enough to read as a trunk without ballooning.
URDF_SCALE_MAP = {"RightFoot": 1.0, "LeftFoot": 1.0, "Pelvis": 1.0, "T8": 2.4}

#: Segments whose URDF link grows along **−Z** from its own frame rather than
#: +Z — every limb cylinder in this model (its visual origin is at
#: z = −length/2, verified by reading the parsed origins). `draw_urdf_skeleton`
#: places the link node at `prox_pos` and aims +Z at `dist_pos`, so those links
#: would be drawn on the wrong side of the joint. Since `viz` is not ours to
#: change, the fix is on the input: hand it a kinematics in which those
#: segments run `prox' = dist`, `dist' = dist + (dist − prox)`. The direction
#: is unchanged, the node lands on the distal joint, and the −Z geometry then
#: spans distal → proximal, i.e. exactly the real segment.
URDF_FLIPPED_SEGMENTS = (1, 2, 5, 6)     # shanks and thighs


def urdf_kinematics(kin, flipped=URDF_FLIPPED_SEGMENTS):
    """Copy of `kin` with `flipped` segments re-anchored — see
    `URDF_FLIPPED_SEGMENTS` for why."""
    from dataclasses import replace
    prox = kin.prox_pos.copy()
    dist = kin.dist_pos.copy()
    for s in flipped:
        d = kin.dist_pos[:, s]
        prox[:, s] = d
        dist[:, s] = d + (d - kin.prox_pos[:, s])
    return replace(kin, prox_pos=prox, dist_pos=dist)


def p1_urdf_html(trial, chain, skel_u, kin_u, frame_index: int | None = None,
                 urdf_path=None) -> str:
    """One posed frame of the vendored URDF humanoid — the optional skeleton
    look, offered instead of the cylinder body.

    `viz.draw_urdf_skeleton` places each named URDF link at our own segment's
    proximal end with its +Z aimed at the distal end and a uniform scale, so
    the figure is this subject's measured geometry wearing the template model's
    shapes. Static: one frame is enough to show what the alternative looks
    like, and it keeps the page's second viewer cheap.
    """
    from pathlib import Path

    from . import viz

    if urdf_path is None:
        urdf_path = (Path(__file__).resolve().parents[2] / "assets" / "urdf"
                     / "humanSubject01_66dof.urdf")
    skel_w, kin_w = whole_body_segments(chain, skel_u, kin_u)
    shift = viewer_frame_shift(kin_w)
    kin_w = shift_kinematics(kin_w, shift)
    if frame_index is None:
        frame_index = int(0.5 * len(kin_w.t))
    floor = float(kin_w.dist_pos[:, 0, 2].min())

    vis = viz.start_viewer()
    try:
        viz.draw_urdf_skeleton(vis, urdf_path, URDF_SEGMENT_MAP,
                               urdf_kinematics(kin_w),
                               frame_index, scale_map=URDF_SCALE_MAP,
                               skeleton=skel_w, color=viz.SEGMENT_COLOR)
        viz.draw_ground(vis, size=1.4)
        vis[f"{viz.ROOT}/ground"].set_transform(
            viz.translation([0.0, 0.0, floor - 0.005]))
        vis["/Grid"].set_property("visible", False)
        vis["/Axes"].set_property("visible", False)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0),
                       offset=(1.55, -1.55, 0.35))
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


def main_v3d(argv=None):
    from pathlib import Path

    from . import io_v3d as io

    argv = list(argv or [])
    path = argv[0] if len(argv) > 0 else V3D_PATH
    index = int(argv[1]) if len(argv) > 1 else V3D_TRIAL

    params = p1_params()
    trials = io.load_v3d_trials(path)
    trial = next(tr for tr in trials if tr.index == index)
    chain, skel_u, kin_u, whole, audit = whole_body_run(trial, params)
    sweep = trial_sweep_stats(trials, params.contact_threshold_n)
    cross = cross_trial_rms(trials, params)
    powers = power_decomposition(trial, chain, skel_u, kin_u, params)
    hjc = hjc_comparison(trial, params)

    viewer_html = urdf_html = None
    try:
        viewer_html = p1_viewer_html(trial, chain, skel_u, kin_u, params)
    except Exception as exc:            # viz is optional for the report
        print(f"viz skipped: {exc}")
    try:
        urdf_html = p1_urdf_html(trial, chain, skel_u, kin_u)
    except Exception as exc:
        print(f"urdf viz skipped: {exc}")

    out = Path(__file__).resolve().parents[2] / "reports"
    out.mkdir(exist_ok=True)
    dest = out / "p1_report.html"
    dest.write_text(p1_report_html(trial, chain, skel_u, kin_u, whole, audit,
                                   params, sweep, cross, powers, hjc,
                                   viewer_html, urdf_html))
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    """`python -m boneid.report [v3d [matpath [trial_index]]]`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "v3d":
        main_v3d(argv[1:])
    else:
        main_s2s()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Quickstart report (examples/quickstart.py — the 10-step walkthrough)
# ---------------------------------------------------------------------------

def quickstart_report_html(trial, body_mass, params, chains, upper, contact,
                           crossover, whole, audit) -> str:
    """Compact report for the 10-step quickstart: the essential outputs of a
    whole-body inverse-dynamics run and nothing else. `chains` maps side ->
    (skeleton, kin, ground); `upper` is `(skel_u, kin_u)` from
    `io_v3d.build_upper_body`; `contact` maps side -> (mask, events);
    `crossover` maps side -> bool flags; `whole` is
    `core.inverse_dynamics_whole_body`."""
    skel_u, kin_u = upper
    skel_r, kin_r, ground_r = chains["r"]
    t = kin_r.t
    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    # forces + contact + crossover
    fig, ax = new_fig(height=2.9)
    for k, side in enumerate(SIDES):
        g = chains[side][2]
        ax.plot(t, g.force[:, 2], color=JOINT_COLORS[k], lw=1.4,
                label=f"{side.upper()} belt Fz")
        mask, _ = contact[side]
        ax.fill_between(t, 0, g.force[:, 2], where=mask,
                        color=JOINT_COLORS[k], alpha=0.12)
    ax.axhline(params.contact_threshold_n, color=FAINT, lw=1.0, ls="--")
    style_axes(ax, "time (s)", "vertical force (N)")
    ax.legend(frameon=False, fontsize=9)
    add(fig, f"Steps 2 & 6 — vertical force per belt with detected contact "
             f"shaded; threshold {params.contact_threshold_n:.0f} N. "
             f"Crossover flags: "
             f"{int(crossover['r'].sum())} (R) / {int(crossover['l'].sum())} "
             f"(L) frames (double support in walking).")

    # torques, both legs, equal axes per joint column
    rms = two_leg_rms(trial, whole)
    fig, axes = new_grid(2, 3, height=5.0, width=8.6)
    for row, side in enumerate(SIDES):
        refs = reference_torques(trial, side)
        legs = whole.right if side == "r" else whole.left
        for j in range(3):
            ax = axes[row, j]
            ax.plot(t, refs[j][:, 0], color=NEUTRAL, lw=2.6, alpha=0.4,
                    label="Visual3D")
            ax.plot(t, legs.joint_torque[:, j, 0], color=JOINT_COLORS[j],
                    lw=1.1, label="ours")
            ax.set_title(JOINT_LABELS[row * 3 + j], color=NEUTRAL, fontsize=10)
            style_axes(ax, "time (s)" if row == 1 else "",
                       "torque, lab x (N m)" if j == 0 else "")
            if row == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8.5)
    for j in range(3):  # RULE: bilateral panels share y-limits
        lims = axes[0, j].get_ylim() + axes[1, j].get_ylim()
        for row in range(2):
            axes[row, j].set_ylim(min(lims), max(lims))
    add(fig, "Steps 7 & 8 — the output: net joint torques (sagittal "
             "component), both legs, with Visual3D's independent computation "
             "behind ours. RMS differences: "
             + ", ".join(f"{JOINT_LABELS[k]} {rms[k]:.2f}" for k in range(6))
             + " N m.")

    # residual (now at the torso) + L5S1 joint wrench + energy
    edge = slice(10, -10)
    res_mean = whole.residual_force[edge].mean(axis=0)
    l5_mean = whole.l5s1_force[edge].mean(axis=0)
    upper_n = float(skel_u.mass.sum()) * 9.81
    fig, axes = new_grid(1, 3, height=2.9, width=9.2)
    for c, lab in enumerate("xyz"):
        axes[0, 0].plot(t, whole.residual_force[:, c], color=JOINT_COLORS[c],
                        lw=1.1, label=f"F{lab}")
        axes[0, 1].plot(t, whole.l5s1_force[:, c], color=JOINT_COLORS[c],
                        lw=1.1, label=f"F{lab}")
    axes[0, 0].axhline(res_mean[2], color=JOINT_COLORS[3], lw=1.3)
    axes[0, 1].axhline(-upper_n, color=NEUTRAL, lw=1.3, ls="--")
    style_axes(axes[0, 0], "time (s)", "residual force, torso COM (N)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, ncol=3)
    style_axes(axes[0, 1], "time (s)", "L5S1 joint force (N)")
    axes[0, 1].legend(frameon=False, fontsize=8.5, ncol=3)
    axes[0, 2].plot(t, audit.de_dt, color=NEUTRAL, lw=2.4, alpha=0.4,
                    label="d(KE+PE)/dt")
    axes[0, 2].plot(t, audit.power_total, color=JOINT_COLORS[0], lw=1.0,
                    label="summed wrench power")
    style_axes(axes[0, 2], "time (s)", "power (W)")
    axes[0, 2].legend(frameon=False, fontsize=8.5)
    add(fig, f"Steps 8 &amp; 9 — the other outputs every run ships with. Left: "
             f"the residual wrench, which with the whole body modelled sits at "
             f"the TORSO's centre of mass and averages {res_mean[2]:.1f} N "
             f"vertically — near zero, because no body part is missing any "
             f"more. Middle: the L5/S1 joint wrench (torso on pelvis), whose "
             f"mean of {l5_mean[2]:.0f} N should be minus the upper body's "
             f"weight ({-upper_n:.0f} N, dashed) and is. Right: the energy "
             f"audit (the two curves must lie on top of each other).")

    imb = np.abs(audit.imbalance[edge]).max()
    pk = max(np.abs(audit.de_dt).max(), 1e-9)
    stats = "".join([
        stat("body mass (from GRF)", f"{body_mass:.1f}", "kg"),
        stat("median torque RMS vs V3D", f"{np.median(rms):.2f}", "N m"),
        stat("energy imbalance", f"{100 * imb / pk:.2f}", "% of peak dE/dt",
             "pass" if imb < 0.03 * pk else "fail"),
        stat("mean vertical residual", f"{res_mean[2]:.1f}",
             "N at the torso COM"),
    ])

    body = f"""
<p class="eyebrow">boneid · quickstart · 10 steps</p>
<h1>Inverse Dynamics in Ten Steps</h1>
<p class="subtitle">The output of <code>examples/quickstart.py</code>: one
high-level call per step, from raw mocap and force data to net joint torques,
the L5/S1 joint wrench, a near-zero residual and an energy audit — the
<b>whole body</b>, twelve segments, trial {trial.index},
{len(t) / kin_r.rate:.1f} s at {kin_r.rate:.0f} Hz.</p>

<div class="stat-row">{stats}</div>

<h2>The ten steps</h2>
<ol>
<li><b>Mocap data</b> — <code>load_v3d_trial(path, index)</code>: markers, landmarks, rates.</li>
<li><b>Force data</b> — rides on the trial: per-belt force, COP, free moment at the analog rate.</li>
<li><b>Skeleton</b> — <code>estimate_body_mass(trial)</code> + de Leva regressions.</li>
<li><b>Treadmill definition</b> — <code>AnalysisParams(...)</code> and a fixed lab point; torques are about the treadmill origin, no COP division.</li>
<li><b>Segment kinematics</b> — <code>build_chain(trial, side)</code> plus <code>build_upper_body(trial)</code>: joint centers, segment frames, ground wrench — one call per side plus the upper body.</li>
<li><b>Contact &amp; crossover</b> — <code>detect_contact</code>, <code>crossover_flags</code>.</li>
<li><b>Inverse dynamics</b> — <code>inverse_dynamics_whole_body(...)</code>: both legs bottom-up into a shared pelvis, both arms into the torso, then the torso's own balance.</li>
<li><b>Outputs</b> — <code>joint_torque</code> per joint, the <code>l5s1</code> joint wrench, and the <code>residual</code> wrench at the torso COM (near zero: nothing is unmodelled).</li>
<li><b>Energy audit</b> — <code>energy_audit_whole_body(...)</code>: d(KE+PE)/dt vs summed wrench power, every run.</li>
<li><b>Report</b> — this page.</li>
</ol>

{"".join(figs)}

<h2>Reproduce</h2>
<p><code>uv run python examples/quickstart.py [mat_path] [trial_index]</code>
— edit the config block at the top of the script for your study; the ten
steps do not change.</p>
"""
    return html_page("Inverse Dynamics in Ten Steps", "qs", body)
