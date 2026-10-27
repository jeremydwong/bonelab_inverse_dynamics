"""boneid.report — matplotlib figures and HTML reports.

Free functions only. Figures are rendered to base64 PNGs and embedded in a
self-contained HTML page (no external assets except Google Fonts).

Chart conventions (kept consistent across every report):
  ankle #2a78d6 (blue)   knee #eb6834 (orange)   hip #1baf7a
  extra series #eda100 (yellow); neutrals for reference lines.
  One axis per chart, thin marks, faint grid, no top/right spines.

Run `uv run python examples/regenerate_reports.py` to rebuild every report
from its notebook, or `uv run python examples/regenerate_reports.py p1_report`
for the real-data
walkthrough at reports/p1_report.html, or
`uv run python examples/regenerate_reports.py p1_asics` for the legacy-
physics A/B ("Implication of Asics") at reports/p1_asics.html.
"""

from __future__ import annotations

import base64
import io
import json
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


def joint_dof_comparison_html(audit, names, edge: int = 5) -> str:
    """Same-motion 3/6-DOF comparison, with signed work in a stated window."""
    from html import escape
    from .core import joint_power, joint_work
    sl = slice(edge, -edge) if edge else slice(None)
    p3 = joint_power(audit, dof=3)
    p6 = joint_power(audit, dof=6)
    w3 = joint_work(audit, dof=3, window=sl)
    w6 = joint_work(audit, dof=6, window=sl)
    fig, axes = new_grid(rows=len(names), cols=1, height=1.7 * len(names),
                         sharex=True, sharey=True)
    for j, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.plot(audit.t[sl], p3[sl, j], label="3-DOF rotational", lw=1)
        ax.plot(audit.t[sl], p6[sl, j], label="6-DOF total", lw=1)
        ax.plot(audit.t[sl], (p6 - p3)[sl, j], label="translational", lw=0.8,
                ls=":")
        style_axes(ax, "time (s)", f"{names[j]} (W)")
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    rows = "".join(
        f"<tr><td>{escape(name)}</td>"
        + "".join(f"<td>{v[0,j]:.2f} / {v[1,j]:.2f} / {v[2,j]:.2f}</td>"
                  for v in (w3, w6, w6-w3)) + "</tr>"
        for j, name in enumerate(names))
    return (
        '<h2>3-DOF versus 6-DOF joint work</h2>'
        '<p>Same segment motion and joint wrenches. 3-DOF uses '
        'M·(ω_dist−ω_prox); 6-DOF adds F·(v_dist−v_prox), with both '
        'material velocities evaluated at the same wrench reference point. '
        'The energy audit always uses 6-DOF. This is a power-definition '
        'comparison, not a constrained-kinematics refit.</p>'
        f'<img src="{fig_svg(fig)}" style="width:100%" alt="3-DOF and 6-DOF joint powers">'
        f'<p>Work over {audit.t[sl][0]:.3f}–{audit.t[sl][-1]:.3f} s '
        '(whole retained window, not per stride). Each cell: positive / '
        'signed negative / net J. Components are summed before clipping.</p>'
        '<table><thead><tr><th>Joint</th><th>3-DOF (J)</th><th>6-DOF (J)</th>'
        '<th>6−3 difference (J)</th></tr></thead><tbody>' + rows + '</tbody></table>'
        '<p><a href="https://doi.org/10.1242/jeb.115451">Zelik, Takahashi '
        '&amp; Sawicki (2015)</a>: full wrench power accounts for translation '
        'and is invariant to shifting the wrench reference point with segment '
        'motion held fixed. Rebuilding segment frames/COMs under another hip '
        'centre convention can still change total power. Translational power '
        'may include marker artefact and model error; it is not isolated '
        'cartilage or muscle work. Distal-to-foot power remains separate.</p>')


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


def hjc_comparison(trial, params=None, edge: int = EDGE, *, dof: int = 3) -> dict:
    """Run the whole-body pipeline once per hip-centre convention.

    Returns `{method: {...}}` with, per method and per side, the joint powers
    (rotational for dof=3, full wrench for dof=6), reaction forces, integrated
    positive/negative work and peaks, plus the offset from the landmark centre.
    """
    from .core import inverse_dynamics_whole_body, energy_audit, joint_power
    if dof not in (3, 6):
        raise ValueError("dof must be 3 or 6")
    params = params or p1_params()
    sl = slice(edge, -edge) if edge else slice(None)
    out = {}
    for method in HJC_METHODS:
        chain, skel_u, kin_u = whole_body_chain(trial, params, method)
        whole = inverse_dynamics_whole_body(*chain, skel_u, kin_u, params)
        entry = {"offset_mm": {}, "power": {}, "force": {}, "work": {},
                 "peak_power": {}, "peak_force": {}}
        for k, side in enumerate(SIDES):
            skel, kin = chain[3 * k], chain[3 * k + 1]
            res = whole.right if side == "r" else whole.left
            p = joint_power(energy_audit(skel, kin, chain[3 * k + 2],
                                         res, params), dof=dof)
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
                   cross, powers, hjc, viewer_html: str | None = None) -> str:
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

    # ---- 6b. the residual at the torso COM --------------------------------
    fig, ax = new_fig(height=3.0)
    for c in range(3):
        ax.plot(t, whole.residual_force[:, c], color=JOINT_COLORS[c],
                lw=1.2, label=f"F{lab[c]}")
    ax.axhline(res_mean[2], color=JOINT_COLORS[3], lw=1.4)
    ax.annotate(f"mean Fz = {res_mean[2]:.1f} N", (t[0], res_mean[2]),
                ha="left", va="bottom", fontsize=8.5, color=JOINT_COLORS[3])
    shade(ax, contact_spans(mask["r"], t), NEUTRAL, 0.06)
    style_axes(ax, "time (s)", "residual force (N)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3)
    peak_grf = float(trial.force[:, :, 2].sum(axis=1).max())
    add(fig, f"The residual wrench at the torso's centre of mass — the model's "
             f"error term. Mean ({res_mean[0]:.1f}, {res_mean[1]:.1f}, "
             f"{res_mean[2]:.1f}) N against a body weight of {bw:.0f} N; the "
             f"peak of {res_peak:.0f} N ({res_peak / bw:.2f} BW, "
             f"{100 * res_peak / peak_grf:.0f}% of peak GRF) arrives with "
             f"heel strike.  CONTEXT: these spikes are expected, not a bug. "
             f"They are dominated by genuinely low-frequency soft-tissue "
             f"(wobbling-mass) motion the rigid segments cannot represent — "
             f"low-passing the forces at the same 12 Hz as the kinematics "
             f"changes the peak by under 10%, so this is not filter "
             f"mismatch — plus de Leva's population inertia estimates.  The "
             f"same rigid-body force deficit is the very signal soft-tissue "
             f"work studies measure (Zelik &amp; Kuo 2010; Riddick &amp; Kuo "
             f"2016).  For calibrated comparison: OpenSim's validation "
             f"guideline of residual force &lt; 5% of peak GRF (Hicks et "
             f"al. 2015) applies AFTER residual-reduction algorithms have "
             f"adjusted the model's mass distribution and kinematics to "
             f"absorb exactly this discrepancy; raw Newton–Euler on "
             f"measured kinematics, as here, reports it instead — and even "
             f"dedicated reduction tools often miss the residual-moment "
             f"threshold on running data (Fox 2023, bioRxiv 555836, "
             f"comparing RRA/MocoTrack/AddBiomechanics).  Near zero on "
             f"average, honest instant to instant.")

    # ---- 6c. is Visual3D's own residual in the same range? ---------------
    # Their model stops at the pelvis, so their exported pelvis-top wrench
    # (rPvProxEndForce) carries the whole unmodeled upper body — the same
    # role our two-leg residual played before the torso and arms were
    # modeled.  Subtracting our upper-body DEMAND (the L5S1 wrench our
    # whole-body model measures from the same markers) leaves the part of
    # THEIR pelvis-top force that rigid-body dynamics does not explain.
    v3d_pv = io.reference_series(trial, "rPvProxEndForce")
    implied = v3d_pv - whole.l5s1_force
    imp_mean = implied[edge:-edge].mean(axis=0)
    imp_fluct = np.sqrt(((implied[edge:-edge] - imp_mean) ** 2).mean(axis=0))
    imp_peak = float(np.abs(implied[edge:-edge]).max())
    ours_fluct = np.sqrt((whole.residual_force[edge:-edge] ** 2).mean(axis=0))
    r_z = float(np.corrcoef(v3d_pv[edge:-edge, 2],
                            whole.l5s1_force[edge:-edge, 2])[0, 1])
    fig, axes = new_grid(1, 2, height=3.0, width=9.2)
    axes[0, 0].plot(t, v3d_pv[:, 2], color=NEUTRAL, lw=2.4, alpha=0.45,
                    label="V3D pelvis-top Fz")
    axes[0, 0].plot(t, whole.l5s1_force[:, 2], color=JOINT_COLORS[0], lw=1.2,
                    label="our L5S1 Fz")
    style_axes(axes[0, 0], "time (s)", "pelvis-top vertical force (N)")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for c in range(3):
        axes[0, 1].plot(t, implied[:, c], color=JOINT_COLORS[c], lw=1.1,
                        label=f"implied F{lab[c]}")
    axes[0, 1].plot(t, whole.residual_force[:, 2], color=FAINT, lw=1.6,
                    label="our residual Fz (scale ref)")
    style_axes(axes[0, 1], "time (s)", "V3D implied residual (N)")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    add(fig, f"Is Visual3D's own leftover in the same range as ours?  Left: "
             f"the top of THEIR chain — Visual3D's exported pelvis-proximal "
             f"force — against our L5S1 wrench, the same physical quantity "
             f"estimated independently (vertical correlation r = {r_z:.2f}; "
             f"their mean {v3d_pv[edge:-edge, 2].mean():.0f} N vs ours "
             f"{whole.l5s1_force[edge:-edge, 2].mean():.0f} N — the ~"
             f"{abs(v3d_pv[edge:-edge, 2].mean() - whole.l5s1_force[edge:-edge, 2].mean()):.0f} N "
             f"offset is anthropometric bookkeeping: where the pelvis ends "
             f"and how much mass sits above it, ~5 kg between the two "
             f"segment tables).  Right: subtracting our measured upper-body "
             f"demand from their pelvis-top force leaves V3D's IMPLIED "
             f"residual: fluctuation RMS ({imp_fluct[0]:.0f}, "
             f"{imp_fluct[1]:.0f}, {imp_fluct[2]:.0f}) N, peak "
             f"{imp_peak:.0f} N — against our whole-body residual's "
             f"({ours_fluct[0]:.0f}, {ours_fluct[1]:.0f}, "
             f"{ours_fluct[2]:.0f}) N and {res_peak:.0f} N (grey trace for "
             f"scale).  ASSUMPTIONS, stated plainly: the upper-body demand "
             f"comes from OUR de Leva torso+arm model on the shared "
             f"markers, so their implied residual mixes their leg-analysis "
             f"error with our upper-body-model error, and their pelvis "
             f"segment differs from ours.  TENTATIVE CONCLUSION: the two "
             f"analyses leave the same order of unexplained force — "
             f"V3D's implied residual is, if anything, somewhat smaller "
             f"(heavier filtering); our heel-strike residual spikes are "
             f"not an artefact peculiar to our implementation.")

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

    # RULE: bilateral quantities are organised L | R per joint, not by side.
    # JOINT_LABELS order is (R ankle, R knee, R hip, L ankle, L knee, L hip),
    # so joint j pairs indices (j + 3, j) as (left, right).
    stats_rms = "".join([
        stat(f"{name} RMS  L | R",
             f"{rms[j + 3]:.2f} | {rms[j]:.2f}",
             f"N m (peaks {ref_peak[j + 3]:.0f} | {ref_peak[j]:.0f})",
             "pass" if max(rms[j + 3] / ref_peak[j + 3],
                           rms[j] / ref_peak[j]) < (0.20 if j == 2 else 0.08)
             else "fail")
        for j, name in enumerate(("ankle", "knee", "hip"))])
    stats = "".join([
        stat("whole-body residual Fz", f"{res_mean[2]:.1f}", "N mean",
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
        [("vertical GRF (N)",
          [("R belt", JOINT_COLORS[0], None, ground_r.force[strip_slice, 2]),
           ("L belt", JOINT_COLORS[2], "4 3",
            ground_l.force[strip_slice, 2])]),
         ("joint-centre\nheight (m)",
          [(f"{'RL'[k]} {joints[j]}", JOINT_COLORS[j],
            None if k == 0 else "4 3",
            (kin_r if k == 0 else kin_l).prox_pos[strip_slice, j, 2])
           for k in range(2) for j in range(3)]),
         ("sagittal joint\ntorque (N m)",
          [(f"{'RL'[k]} {joints[j]}", JOINT_COLORS[j],
            None if k == 0 else "4 3",
            legs[SIDES[k]].joint_torque[strip_slice, j, 0])
           for k in range(2) for j in range(3)])],
        period_s=period, fps=fps, n_frames=n_key,
        viewer_iframe="p1-viewer")

    viewer = ""
    if viewer_html is not None:
        esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
        viewer = (
            "<h2>Step 8 — the trial, moving</h2>"
            "<p>All twelve modelled segments — both legs (foot, shank, thigh), "
            "the pelvis, the torso+head and both arms — posed from the measured "
            "joint centres, every marker in the export in red (anatomical "
            "pairs, landmarks, and the SH/TH tracking clusters), and "
            "<em>both</em> ground reactions drawn as green arrows from their "
            "equivalent centres of pressure (1 kN per metre). Each arrow is "
            "hidden while its belt reads below the 20 N contact threshold — "
            "the residue there is anti-alias filter ripple with no defined "
            "centre of pressure — so the alternating stance is visible "
            f"directly. The pelvis is drawn at its real width, spanning the "
            f"two hip centres, and the torso as a cylinder sized to displace "
            f"its actual {torso_mass:.0f} kg at 1000 kg/m³ over the measured "
            f"L5/S1-to-acromion distance — it is not decoration, it is what "
            f"that mass looks like. Decimated to {fps:.0f} fps, looping. "
            "Drag to orbit.</p>"
            "<p>Directly below the animation: both belts' vertical force, "
            "joint-centre heights (both legs, one panel so the axes are "
            "necessarily identical), and sagittal joint torques, as a "
            "<b>scrolling scope</b>: the data streams right-to-left through a "
            "fixed window with the orange centre line pinned at the "
            "animation's current instant — recent past on the left, immediate "
            "future on the right. The scroll is <b>synchronised to the "
            "looping animation — the same period, and both start on page "
            "load</b> — and the scroll <b>listens to the viewer</b>: it reads the "
            "animation clock directly, so pausing in the viewer's Animations "
            "panel freezes the graph and dragging its time scrubber drags "
            f"the graph with it (period {period:.3f} s, {n_key} keyframes at "
            f"{fps:.0f} fps; a wall-clock loop stands in until the viewer "
            "has loaded).</p>"
            f'<iframe id="p1-viewer" class="viewer" srcdoc="{esc}"></iframe>'
            f"{strips}")

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
weight of everything above it on average. Step 6 checks that.</li>
<li>The <b>residual has moved to the torso's centre of mass</b>, and with
every segment of the body now modelled it has nothing structural left to
carry: its mean vertical component is {res_mean[2]:.1f} N. What is left is the honest whole-model error term —
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
Cardan-angle derivatives. The recovered legacy RoomSegAng also differentiates
matrices, but uses a different matrix product (see Appendix A).</p>

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

<p>Two more outputs ship with every run alongside the torques. The
<b>residual wrench</b> at the torso's centre of mass — the whole-model error
term, mean F<sub>z</sub> {res_mean[2]:.1f} N against a body weight of
{bw:.0f} N:</p>

{figs[8]}
{figs[9]}

<p>And the <b>L5/S1 joint wrench</b> — the wrench the torso exerts on the
pelvis, mirroring the hips. Its mean vertical force should equal minus the
de Leva weight of the segments above it ({-upper_n:.0f} N for the
{upper_mass:.1f} kg torso, head and arms), and nothing in the recursion is
told the subject's total mass — yet it lands within
{100 * abs(l5_gap) / abs(upper_n):.1f}% on this trial and stays on the
identity line across all {len(cross['index'])} trials:</p>

{figs[10]}

<h2>Step 7 — the energy audit</h2>

{figs[11]}
{figs[12]}

{viewer}

<h2>Step 9 — whole-body power: COM, peripheral, and below the foot</h2>
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
is Zelik &amp; Kuo's peripheral term (2010, <i>J Exp Biol</i> 213:4257–4264),
summed here over all twelve segments — swinging arms and a moving trunk
included — which is the whole-body quantity Zelik &amp; Kuo actually
define.</p>

{figs[14]}

<p><b>Distal-to-foot power.</b> Takahashi, Kepple &amp; Stanhope (2012,
<i>J Biomech</i> 45:2662–2667) unified-deformable segment power: the ground
wrench's power on the rigid foot segment, which is everything the structures
below that segment — shoe, heel pad, arch, all foot joints — deliver or
absorb. Zelik &amp; Honert (2018, <i>J Biomech</i> 75:1–12) argue this is the
measure to report rather than ankle power alone, because "ankle power" silently
attributes foot deformation to the ankle joint.</p>

{figs[15]}

<p><b>What cannot be done here.</b> Splitting the foot at the metatarso-
phalangeal joint into hindfoot and forefoot — the multi-segment foot analysis
that would separate the arch from the toes — needs forefoot and hallux markers.
This set has the calcaneus, the 5th metatarsal head and the malleoli: one rigid
foot and nothing more. So no MTP joint is offered and the distal-to-foot power
above lumps every sub-foot structure together. Inventing an MTP axis from three
markers would produce a curve, not a measurement.</p>

<h2>Step 10 — hip joint centre conventions</h2>
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

{figs[16]}

{hjc_work_table}

{figs[17]}

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

<h2>Step 11 — every trial, both legs</h2>
<p>The previous steps prove one trial. This one proves the pipeline: identical
code, identical parameters, no per-trial tuning, all
{len(cross['index'])} trials of the subject and all six leg joints, scored
against Visual3D.</p>

{figs[13]}

<h2>How to reproduce</h2>
<p><code>uv run python examples/regenerate_reports.py p1_report</code> regenerates this page from
<code>{trial.path}</code>; edit the input path and trial slot in
<code>notebooks/p1_report.ipynb</code> to analyse another recording.
<code>uv run pytest</code> runs the KEY suite and the loader tests behind it,
and <code>notebooks/p1_report.ipynb</code> is the same analysis step by step in
a notebook.</p>
"""
    body += joint_dof_comparison_html(audit,
        ["R ankle", "R knee", "R hip", "L ankle", "L knee", "L hip",
         "L5/S1", "R shoulder", "R elbow", "L shoulder", "L elbow"])
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
#: every marker in the export — anatomical pairs, landmarks, and the
#: SH/TH tracking clusters — so the viewer shows the data, not a selection.
VIEWER_MARKERS = ("RCAL", "R5TH", "RLML", "RMML", "RLEP", "RMEP", "RGTR",
                  "LCAL", "L5TH", "LLML", "LMML", "LLEP", "LMEP", "LGTR",
                  "RASI", "LASI", "SACR", "RAC", "LAC", "REP", "LEP",
                  "RWR", "LWR",
                  "RSH1", "RSH2", "RSH3", "LSH1", "LSH2", "LSH3",
                  "RTH1", "RTH2", "RTH3", "LTH1", "LTH2", "LTH3")


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


def p1_foot_blob(trial, kin, side):
    """Rigid visual envelope from the trial-median foot-local landmarks.

    P1 has a heel, two malleoli and one fifth-metatarsal marker per foot.
    The medial forefoot/toe tips are unmeasured; do not invent them.
    """
    from .viz import foot_blob_mesh
    prefix = side.upper()[0]
    points = np.stack([trial.markers[prefix + n]
                       for n in ("CAL", "5TH", "MML", "LML")], axis=1)
    local = np.einsum("tji,tmj->tmi", kin.r_world[:, 0],
                      points - kin.prox_pos[:, 0, None, :])
    return foot_blob_mesh(np.median(local, axis=0))


def viewer_floor(chain, shift) -> float:
    """Floor height in the viewer frame: the lowest distal foot point of
    either leg over the clip (the plane the GRF arrow tail is drawn on)."""
    return float(min(chain[1].dist_pos[:, 0, 2].min(),
                     chain[4].dist_pos[:, 0, 2].min()) + shift[2])


def _animate_whole_body(vis, trial, chain, skel_u, kin_u, shift, floor,
                        decimate, prefix=None, animation=None,
                        markers=VIEWER_MARKERS, skip_markers=(),
                        floor_drawn: bool = True):
    """Add one whole P1 body to `vis`: both legs (right chain carries the
    shared pelvis), the pelvis band, the mass-sized torso, both arms and the
    markers, all translated by `shift`, under ``prefix/{r,l,ar,al,markers}``
    (default `viz.ROOT`).  Every track lands in `animation` (a new one when
    None, which is returned).  Only that first body draws the floor, unless
    `floor_drawn` is False.  Used by `p1_viewer_html` (one body) and
    `cadence_viewer_html` (one body per step frequency).
    """
    from . import viz
    from .core import UPPER_BODY_ARM, UPPER_BODY_TORSO, slice_chain
    prefix = prefix or viz.ROOT
    skel_r, kin_r, ground_r, skel_l, kin_l, ground_l = chain
    vkin = {"r": shift_kinematics(kin_r, shift),
            "l": shift_kinematics(kin_l, shift),
            "u": shift_kinematics(kin_u, shift)}
    vground = {"r": shift_wrench(ground_r, shift),
               "l": shift_wrench(ground_l, shift)}
    # left leg without the (shared) pelvis segment
    skel_left3, kin_left3 = slice_chain(skel_l, vkin["l"], [0, 1, 2])
    traj = {name: trial.markers[name] + shift for name in markers
            if name in trial.markers and name not in skip_markers}
    root = {tag: f"{prefix}/{tag}" for tag in ("r", "l", "ar", "al")}
    pelvis_span = (vkin["l"].prox_pos[:, 2], vkin["r"].prox_pos[:, 2],
                   vkin["r"].prox_pos[:, 3])
    torso = (vkin["u"].prox_pos[:, UPPER_BODY_TORSO],
             vkin["u"].dist_pos[:, UPPER_BODY_TORSO],
             float(skel_u.mass[UPPER_BODY_TORSO]))
    first = animation is None
    animation = viz.animate(vis, skel_r, vkin["r"], ground=vground["r"],
                            decimate=decimate, repetitions=1, play=False,
                            plane_height=floor, root=root["r"],
                            pelvis_span=pelvis_span, torso=torso,
                            animation=animation, camera=first,
                            floor=first and floor_drawn,
                            segment_meshes={0: p1_foot_blob(trial, kin_r, "r")})
    viz.animate(vis, skel_left3, kin_left3, ground=vground["l"],
                decimate=decimate, repetitions=1, play=False,
                plane_height=floor, root=root["l"], animation=animation,
                camera=False, floor=False,
                top_girth_fraction=viz.GIRTH_FRACTION,
                segment_meshes={0: p1_foot_blob(trial, kin_l, "l")})
    for side in ("r", "l"):
        arm_skel, arm_kin = slice_chain(skel_u, vkin["u"],
                                        UPPER_BODY_ARM[side])
        # joint_balls=False: the arm joint centres ARE the AC/EP/WR
        # markers, so a dark joint sphere would swallow the red marker
        # that defines it — let the marker itself mark the joint.
        viz.animate(vis, arm_skel, arm_kin, decimate=decimate,
                    repetitions=1, play=False, plane_height=floor,
                    root=root["a" + side], animation=animation,
                    camera=False, joint_balls=False, floor=False,
                    top_girth_fraction=viz.GIRTH_FRACTION)
    viz.draw_markers(vis, traj, 0, root=prefix)
    step = max(1, int(decimate))
    for k, f in enumerate(range(0, len(kin_r.t), step)):
        with animation.at_frame(vis, k) as frame:
            for name, xyz in traj.items():
                frame[f"{prefix}/markers/{name}"].set_transform(
                    viz.translation(xyz[f]))
    return animation


def p1_viewer_html(trial, chain, skel_u, kin_u, params,
                   decimate: int = VIEWER_DECIMATE,
                   repetitions: int = 10000,
                   markers: tuple[str, ...] = VIEWER_MARKERS,
                   knee_shifts_m=(),
                   camera_offset=(1.55, -1.55, 0.35)) -> str:
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
    frame of `viewer_frame_shift`; the ground box and meshcat's grid sit at
    the subject's foot level (`viz.draw_ground(height=...)`), and meshcat's
    axes are hidden because at the origin they would float at the waist.

    `knee_shifts_m` (e.g. 0, 0.005, ... 0.05) draws the right shank + thigh
    once per lateral knee shift (`simulate.varus_shift`, markers carried by
    `varus_markers`) in groups `boneid/v0`, `boneid/v1`, ...; only v0 starts
    visible and the measured right shank/thigh are hidden.  A
    `strip_charts_grid(variants=...)` slider then picks the leg.
    """
    from . import viz
    from .core import slice_chain
    from .simulate import (VARUS_SHANK_MARKERS, VARUS_THIGH_MARKERS,
                           varus_markers, varus_shift)

    skel_r, kin_r, ground_r, skel_l, kin_l, ground_l = chain
    shift = viewer_frame_shift(kin_r, kin_l, kin_u)
    floor = viewer_floor(chain, shift)
    vkin_r = shift_kinematics(kin_r, shift)
    moved = VARUS_SHANK_MARKERS + VARUS_THIGH_MARKERS
    traj = {name: trial.markers[name] + shift for name in markers
            if name in trial.markers}
    variant_traj = [varus_markers(traj, vkin_r, float(d))
                    for d in knee_shifts_m]

    vis = viz.start_viewer()
    try:
        animation = _animate_whole_body(
            vis, trial, chain, skel_u, kin_u, shift, floor, decimate,
            markers=markers,
            skip_markers=moved if len(knee_shifts_m) else ())
        for v, d in enumerate(knee_shifts_m):
            leg_skel, leg_kin = slice_chain(
                skel_r, varus_shift(vkin_r, float(d)), [1, 2])
            viz.animate(vis, leg_skel, leg_kin, decimate=decimate,
                        repetitions=1, play=False, plane_height=floor,
                        root=f"{viz.ROOT}/v{v}", animation=animation,
                        camera=False, floor=False,
                        top_girth_fraction=viz.GIRTH_FRACTION)
            viz.draw_markers(vis, variant_traj[v], 0, root=f"{viz.ROOT}/v{v}")
        step = max(1, int(decimate))
        for k, f in enumerate(range(0, len(kin_r.t), step)):
            with animation.at_frame(vis, k) as frame:
                for v, vt in enumerate(variant_traj):
                    for name, xyz in vt.items():
                        frame[f"{viz.ROOT}/v{v}/markers/{name}"].set_transform(
                            viz.translation(xyz[f]))
        if len(knee_shifts_m):
            for v in range(len(knee_shifts_m)):
                vis[f"{viz.ROOT}/v{v}"].set_property("visible", v == 0)
            for s in (1, 2):
                for part in ("", "_joint"):
                    vis[f"{viz.ROOT}/r/skeleton/{skel_r.segment_names[s]}"
                        f"{part}"].set_property("visible", False)
        vis["/Axes"].set_property("visible", False)
        vis.set_animation(animation, play=True, repetitions=repetitions)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0), offset=camera_offset)
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


def right_strides(chain, threshold_n: float = 20.0) -> np.ndarray:
    """[(on, next_on)] mocap-frame pairs of every complete right stride
    (right-belt loading onsets, the same detector as everywhere else, minus
    blips: a stance must last 0.2 s and reach 0.3 BW)."""
    from .core import detect_contact
    fz = chain[2].force[:, 2]
    bw = float(np.mean(fz + chain[5].force[:, 2]))
    rate = float(chain[1].rate)
    _, ev = detect_contact(fz, threshold_n, min_gap=5)
    # a real stance lasts >= 0.2 s and carries >= 0.3 BW (P1 slot 17 has a
    # ~20 N belt blip that would otherwise count as one)
    on = [a for a, b in ev if a > 0 and (b - a) >= 0.2 * rate
          and fz[a:b + 1].max() >= 0.3 * bw]
    return np.array(list(zip(on[:-1], on[1:])), dtype=int)


def stride_mean(x, strides, n: int = 200) -> np.ndarray:
    """Mean over `strides` of `x` [T, ...] resampled onto n+1 phase points
    (0..100 % of the stride, both ends included so a loop wraps cleanly)."""
    x = np.asarray(x, float)
    ph = np.linspace(0.0, 1.0, n + 1)
    flat = x.reshape(len(x), -1)
    out = []
    for a, b in strides:
        src = np.arange(a, b + 1)
        u = (src - a) / float(b - a)
        out.append(np.stack([np.interp(ph, u, flat[src, c])
                             for c in range(flat.shape[1])], axis=1))
    return np.nanmean(out, axis=0).reshape((n + 1,) + x.shape[1:])


def take_frames(trial, run, idx, rate: float):
    """(trial, chain, skel_u, kin_u) restricted to mocap frames `idx`, with
    `rate` as the new frame rate (so a phase-sampled stride plays back at
    whatever period the caller wants)."""
    from dataclasses import replace
    chain, skel_u, kin_u = run[0], run[1], run[2]
    idx = np.asarray(idx, int)
    t = np.arange(len(idx)) / float(rate)

    def kin(k):
        return replace(k, t=t, rate=float(rate), r_world=k.r_world[idx],
                       prox_pos=k.prox_pos[idx], dist_pos=k.dist_pos[idx])

    def wrench(g):
        return replace(g, t=t, rate=float(rate), force=g.force[idx],
                       moment=g.moment[idx])

    skel_r, kin_r, gr, skel_l, kin_l, gl = chain
    new_chain = (skel_r, kin(kin_r), wrench(gr), skel_l, kin(kin_l), wrench(gl))
    new_trial = replace(trial, t=t, rate=float(rate),
                        markers={k: v[idx] for k, v in trial.markers.items()})
    return new_trial, new_chain, skel_u, kin(kin_u)


def cadence_viewer_html(bodies, rate: float, repetitions: int = 10000,
                        camera_offset=(1.55, -1.55, 0.35)) -> str:
    """Meshcat page with one whole body per condition, each under
    ``boneid/v{k}`` and only v0 visible: `bodies` is a list of
    `take_frames` tuples with the SAME number of frames (one phase-sampled
    stride each) at `rate`, so they share one looping clip.  The first body
    sets the recentring shift and the floor, and draws it; the slider in
    `strip_charts_grid(variants=..., variant_time_scale=...)` swaps bodies
    and playback speed."""
    from . import viz
    trial0, chain0, _, kin_u0 = bodies[0]
    shift = viewer_frame_shift(chain0[1], chain0[4], kin_u0)
    floor = viewer_floor(chain0, shift)
    vis = viz.start_viewer()
    try:
        animation = None
        for v, (trial, chain, skel_u, kin_u) in enumerate(bodies):
            animation = _animate_whole_body(
                vis, trial, chain, skel_u, kin_u, shift, floor, 1,
                prefix=f"{viz.ROOT}/v{v}", animation=animation)
        for v in range(len(bodies)):
            vis[f"{viz.ROOT}/v{v}"].set_property("visible", v == 0)
        vis["/Axes"].set_property("visible", False)
        vis.set_animation(animation, play=True, repetitions=repetitions)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0), offset=camera_offset)
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


def simulated_viewer_html(skeleton, kin, ground=None, markers=None,
                          segment_meshes=None, decimate: int = 10,
                          repetitions: int = 10000,
                          camera_offset=(1.6, -1.6, 0.5),
                          force_threshold: float = 20.0) -> str:
    """Standalone meshcat page for one simulated chain: segments (cylinders,
    or the rigid meshes in `segment_meshes`), the ground-force arrow when a
    wrench is given, and every synthetic marker as an animated red sphere.
    Shared by the brick and squat-to-stand tutorials so that "the movement in
    3-D" is one call there, as it is in the quickstart."""
    from . import viz

    vis = viz.start_viewer()
    try:
        animation = viz.animate(vis, skeleton, kin, ground=ground,
                                decimate=decimate, repetitions=1, play=False,
                                segment_meshes=segment_meshes or {},
                                force_threshold=force_threshold)
        if markers:
            viz.draw_markers(vis, markers, 0)
            step = max(1, int(decimate))
            for k, f in enumerate(range(0, len(kin.t), step)):
                with animation.at_frame(vis, k) as frame:
                    for name, xyz in markers.items():
                        frame[f"{viz.ROOT}/markers/{name}"].set_transform(
                            viz.translation(xyz[f]))
        vis.set_animation(animation, play=True, repetitions=repetitions)
        mid = np.array([np.mean(kin.prox_pos[..., 0]),
                        np.mean(kin.prox_pos[..., 1]),
                        0.5 * kin.prox_pos[..., 2].max()])
        viz.set_camera(vis, target=mid, offset=camera_offset)
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
    t_end = t[-1]
    step = max(1, int(np.ceil(len(t) / max_pts)))
    t0 = t[0]
    t, y = t[::step], y[::step]
    ok = np.isfinite(y)                   # gaps (NaN) are skipped, not drawn
    t, y = t[ok], y[ok]
    span = max(hi - lo, 1e-12)
    xs = x0 + (x1 - x0) * (t - t0) / max(t_end - t0, 1e-12)
    ys = ybot - (ybot - ytop) * (y - lo) / span
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def strip_charts(t, panels, period_s: float, fps: float,
                 n_frames: int, uid: str = "strip",
                 window_s: float = 2.4,
                 viewer_iframe: str | None = None) -> str:
    """Stacked SVG strips as a scrolling scope, synchronised to the animation.

    `panels` is a list of `(ylabel, [(label, color, dash, y), ...])`; every
    series in a panel is drawn on that panel's own shared y-scale, which is how
    the bilateral rule is honoured here — the left and right trace of a joint
    live in the SAME panel and therefore literally cannot have different axes.

    The DATA scrolls right-to-left through a fixed `window_s`-second window
    with "now" pinned at the CENTRE of the plot: the left half is the recent
    past, the right half the immediate future. The traces are drawn once at
    full length (plus a tiled copy on each side so the looping trial wraps
    without a gap) inside a clipped group, and `requestAnimationFrame`
    translates that group with period `period_s` — the meshcat clip's exact
    period, `(n_frames − 1) / fps` (see `viewer_period`). Both the clip and
    this loop start when the page loads, so they run at the same rate from the
    same instant.

    When `viewer_iframe` names the meshcat iframe's element id, the scroll is
    driven by the VIEWER'S OWN CLOCK instead: the srcdoc iframe shares the
    parent's origin, so the script reads `contentWindow.viewer.animator.time`
    each frame — meshcat's Animations panel (play / pause / reset / timeScale
    and the time scrubber) all write through that field, so pausing freezes
    the graph and scrubbing drags it. Falls back to the wall-clock loop until
    the viewer has loaded (or if access ever fails).
    """
    t = np.asarray(t, float)
    x0, x1 = STRIP_PAD_L, STRIP_WIDTH - STRIP_PAD_R
    plotw = x1 - x0
    span = float(t[-1] - t[0])
    px_per_s = plotw / float(window_s)
    data_w = span * px_per_s               # full-trial width in user units
    xc = x0 + 0.5 * plotw                  # the "now" line
    total_h = len(panels) * (STRIP_HEIGHT + STRIP_GAP) + 26
    axis_y = len(panels) * (STRIP_HEIGHT + STRIP_GAP)

    out = [f'<svg id="{uid}-svg" viewBox="0 0 {STRIP_WIDTH} {total_h}" '
           f'width="100%" role="img" '
           f'aria-label="time-locked scrolling strip charts">']
    legend = []
    traces: list[str] = []
    for p_i, (ylabel, series) in enumerate(panels):
        ytop = p_i * (STRIP_HEIGHT + STRIP_GAP)
        ybot = ytop + STRIP_HEIGHT
        lo = min(float(np.min(y)) for *_, y in series)
        hi = max(float(np.max(y)) for *_, y in series)
        pad = 0.06 * max(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        out.append(f'<rect x="{x0}" y="{ytop}" width="{plotw}" '
                   f'height="{STRIP_HEIGHT}" fill="{SURFACE}" '
                   f'stroke="{FAINT}" stroke-width="0.8"/>')
        if lo < 0 < hi:
            zy = ybot - (ybot - ytop) * (0 - lo) / (hi - lo)
            out.append(f'<line x1="{x0}" y1="{zy:.1f}" x2="{x1}" '
                       f'y2="{zy:.1f}" stroke="{FAINT}" stroke-width="0.7"/>')
        for label, color, dash, y in series:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            traces.append(f'<polyline fill="none" stroke="{color}" '
                          f'stroke-width="1.3"{d} points="'
                          f'{_strip_path(t, y, x0, x0 + data_w, ytop, ybot, lo, hi, max_pts=2000)}"/>')
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

    # scrolling time ticks, one per second, drawn with the data
    ticks = []
    sec = int(np.ceil(t[0]))
    while sec <= t[-1]:
        tx = x0 + (sec - t[0]) * px_per_s
        ticks.append(f'<line x1="{tx:.1f}" y1="{axis_y}" x2="{tx:.1f}" '
                     f'y2="{axis_y + 4}" stroke="{NEUTRAL}" stroke-width="0.8"/>'
                     f'<text x="{tx:.1f}" y="{axis_y + 15}" '
                     f'text-anchor="middle" font-size="10" '
                     f'fill="{NEUTRAL}">{sec:d}</text>')
        sec += 1
    pan_content = "".join(traces) + "".join(ticks)

    # three tiled copies so the looping window never shows a gap
    out.append(f'<clipPath id="{uid}-clip"><rect x="{x0}" y="0" '
               f'width="{plotw}" height="{axis_y + 18}"/></clipPath>')
    out.append(f'<g clip-path="url(#{uid}-clip)"><g id="{uid}-pan">'
               + "".join(f'<g transform="translate({k * data_w:.2f},0)">'
                         f'{pan_content}</g>' for k in (-1, 0, 1))
               + "</g></g>")
    out.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" '
               f'stroke="{FAINT}"/>')
    # "now": a fixed centre line spanning the strips
    out.append(f'<line x1="{xc}" y1="0" x2="{xc}" y2="{axis_y}" '
               f'stroke="{JOINT_COLORS[1]}" stroke-width="1.4"/>')
    out.append(f'<text id="{uid}-time" x="{xc}" y="{axis_y + 15}" '
               f'text-anchor="middle" font-size="10" font-weight="600" '
               f'fill="{JOINT_COLORS[1]}"></text>')
    out.append(f'<text x="{x1}" y="{axis_y + 15}" text-anchor="end" '
               f'font-size="10" fill="{NEUTRAL}">time (s)</text>')
    out.append("</svg>")

    viewer_iframe_js = f'"{viewer_iframe}"' if viewer_iframe else "null"
    script = f"""<script>
(function() {{
  var pan   = document.getElementById("{uid}-pan");
  var label = document.getElementById("{uid}-time");
  var T0 = {t[0]:.6f}, SPAN = {span:.6f}, PXS = {px_per_s:.6f};
  var XC_OFF = {0.5 * plotw:.6f};        // centre offset from the plot's left edge
  var PERIOD = {period_s:.6f};           // (n_frames - 1) / fps, exactly
  var IFRAME = {viewer_iframe_js};
  var start = null;
  function viewer_time() {{
    // the srcdoc iframe shares our origin; meshcat's play/pause/scrub all
    // write through animator.time, so this IS the animation's clock.
    if (!IFRAME) return null;
    try {{
      var f = document.getElementById(IFRAME);
      var a = f && f.contentWindow && f.contentWindow.viewer
              && f.contentWindow.viewer.animator;
      if (a && a.duration > 0) {{
        // the action's own clock is authoritative (it wraps per loop and is
        // written by play, pause and the time scrubber alike)
        var tt = (a.actions && a.actions.length) ? a.actions[0].time : a.time;
        if (isFinite(tt))
          return ((tt % a.duration) + a.duration) % a.duration;
      }}
    }} catch (e) {{}}
    return null;
  }}
  function tick(now) {{
    if (start === null) start = now;
    var tau = viewer_time();
    if (tau === null)
      tau = ((now - start) / 1000.0) % PERIOD;     // fallback: wall clock
    // translate so the sample at time tau sits at the centre line
    var tx = XC_OFF - tau * PXS;
    pan.setAttribute("transform", "translate(" + tx + ",0)");
    label.textContent = (T0 + tau).toFixed(2) + " s";
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
        viz.draw_ground(vis, size=1.4, height=floor)
        vis["/Axes"].set_property("visible", False)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0),
                       offset=(1.55, -1.55, 0.35))
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


# ---------------------------------------------------------------------------
# "Implication of Asics" — the legacy physics mistakes, A/B'd on real data
# ---------------------------------------------------------------------------

ASICS_VARIANTS = ("correct", "ab", "abc")

ASICS_LABELS = {
    "correct": "corrected physics",
    "ab": "legacy (a)+(b): one-sided R·I, no gyroscopic term",
    "abc": "(a)+(b) + hypothetical Cardan-angle α",
}

ASICS_STYLE = {                      # colour, linestyle, linewidth, zorder
    "correct": (None, "-", 2.4, 2),  # colour filled in per joint
    "abc": (NEUTRAL, (0, (5, 3)), 1.1, 4),
    "ab": ("#9a6bd6", (0, (1, 2)), 1.4, 3),
}


def asics_comparison(trial, params=None, edge: int = EDGE) -> dict:
    """Run the corrected and the legacy-physics pipelines on the same trial.

    Returns a dict with, for each of `ASICS_VARIANTS`, the
    `WholeBodyInverseDynamics`, the τ·ω_rel joint powers (computed with the
    CORRECT kinematics in every case, so the comparison is of torques alone),
    and the `core.energy_audit_whole_body` imbalance under those torques; plus
    per-joint RMS torque differences, peaks, and the term-magnitude
    diagnostics the prose quotes.

    `ab` is `cardan_alpha=False` — the two VERIFIED mistakes only; `abc` adds
    the hypothetical Cardan-angle "angular acceleration". Everything else (masses,
    COMs, joint centres, forces, filtering, chain) is identical across all
    three by construction: they are the same call with different flags.
    """
    from .core import (LegacyPhysics, cardan_alpha, chain_kinematics,
                       energy_audit_whole_body, inertia_world,
                       inverse_dynamics_whole_body,
                       inverse_dynamics_whole_body_asicssource, slice_chain)
    params = params or p1_params()
    sl = slice(edge, -edge)
    chain, skel_u, kin_u = whole_body_chain(trial, params)

    runs = {
        "correct": inverse_dynamics_whole_body(*chain, skel_u, kin_u, params),
        "ab": inverse_dynamics_whole_body_asicssource(
            *chain, skel_u, kin_u, params, cardan_alpha=False),
        "abc": inverse_dynamics_whole_body_asicssource(
            *chain, skel_u, kin_u, params, cardan_alpha=True),
    }
    out = {"chain": chain, "skel_u": skel_u, "kin_u": kin_u, "runs": runs,
           "torque": {}, "power": {}, "audit": {}, "imbalance": {},
           "rms": {}, "peak_diff": {}, "rms_xyz": {}}

    for key, whole in runs.items():
        # [T,6,3] lab-frame joint torques, ordered as JOINT_LABELS
        out["torque"][key] = np.concatenate(
            [whole.right.joint_torque, whole.left.joint_torque], axis=1)
        # τ·ω_rel with the CORRECT kinematics for every variant
        out["power"][key] = np.concatenate(
            [joint_power_tau_omega(chain[3 * k], chain[3 * k + 1],
                                   whole.right if k == 0 else whole.left,
                                   params)
             for k in range(2)], axis=1)                       # [T,6]
        audit = energy_audit_whole_body(*chain, skel_u, kin_u, whole, params)
        out["audit"][key] = audit
        out["imbalance"][key] = {
            "rms": float(np.sqrt((audit.imbalance[sl] ** 2).mean())),
            "max": float(np.abs(audit.imbalance[sl]).max()),
            "scale": float(np.abs(audit.de_dt[sl]).max())}

    base = out["torque"]["correct"]
    out["peak"] = np.array([np.linalg.norm(base[sl, j], axis=1).max()
                            for j in range(6)])
    out["peak_sag"] = np.abs(base[sl, :, 0]).max(axis=0)
    for key in ("ab", "abc"):
        d = out["torque"][key] - base                          # [T,6,3]
        out["rms"][key] = np.array(
            [float(np.sqrt((d[sl, j] ** 2).sum(axis=1).mean()))
             for j in range(6)])
        out["rms_xyz"][key] = np.sqrt((d[sl] ** 2).mean(axis=0))   # [6,3]
        out["peak_diff"][key] = np.array(
            [float(np.linalg.norm(d[sl, j], axis=1).max()) for j in range(6)])

    # --- each mistake ON ITS OWN, for the "which one dominates" section.
    # Same call, one flag at a time; only the RMS torque difference is kept.
    for key, phys in (("a", LegacyPhysics(one_sided_inertia=True)),
                      ("b", LegacyPhysics(gyroscopic=False)),
                      ("c", LegacyPhysics(cardan_alpha=True))):
        one = inverse_dynamics_whole_body(*chain, skel_u, kin_u, params,
                                          physics=phys)
        d = np.concatenate([one.right.joint_torque, one.left.joint_torque],
                           axis=1) - base
        out["rms"][key] = np.array(
            [float(np.sqrt((d[sl, j] ** 2).sum(axis=1).mean()))
             for j in range(6)])

    # --- term magnitudes on the right leg, for the honest-magnitude section
    leg_skel, leg_kin = slice_chain(chain[0], chain[1], [0, 1, 2])
    ck = chain_kinematics(leg_skel, leg_kin, params.lowpass_hz,
                          params.filter_order)
    a_cardan = cardan_alpha(ck["r"], leg_kin.rate,
                            params.lowpass_hz or 12.0, params.filter_order)
    terms = {}
    for s, name in enumerate(leg_skel.segment_names):
        r = ck["r"][:, s]
        i_w = inertia_world(r, leg_skel.inertia_local[s])
        i_one = np.einsum("tij,jk->tik", r, leg_skel.inertia_local[s])
        w = ck["omega"][:, s]
        i_alpha = np.einsum("tij,tj->ti", i_w, ck["alpha"][:, s])
        w_local = np.einsum("tji,tj->ti", r, w)
        frac = (np.abs(w_local[sl])
                / (np.linalg.norm(w_local[sl], axis=1, keepdims=True) + 1e-12))
        terms[name] = {
            "i_alpha": float(np.abs(i_alpha[sl]).max()),
            "gyro": float(np.abs(np.cross(
                w, np.einsum("tij,tj->ti", i_w, w))[sl]).max()),
            "one_sided": float(np.abs(
                (np.einsum("tij,tj->ti", i_one, ck["alpha"][:, s])
                 - i_alpha)[sl]).max()),
            "cardan": float(np.abs(np.einsum(
                "tij,tj->ti", i_one, (a_cardan[:, s] - ck["alpha"][:, s]))[sl]
            ).max()),
            "omega_frac": frac.mean(axis=0)}
    out["terms"] = terms
    return out


def asics_grid(t, series, joints, ylabel, spans=None, window=None,
               scale=1.0):
    """A 2x3 (rows = R/L side, cols = joint) overlay of variant curves.

    `series` maps variant key -> [T,6] array in JOINT_LABELS order. Y-limits
    are shared down each joint COLUMN (the CLAUDE.md bilateral rule) — the
    left and the right panel of the same joint are always directly comparable.
    """
    fig, axes = new_grid(2, 3, height=4.8, width=8.6, sharex=True)
    sl = slice(*window) if window else slice(None)
    handles = None
    for k in range(2):                                   # 0 = right, 1 = left
        for j in range(3):
            ax = axes[k, j]
            if spans:
                for a, b in spans[k]:
                    ax.axvspan(a, b, color=FAINT, alpha=0.30, lw=0)
            for key in ASICS_VARIANTS:
                if key not in series:
                    continue
                colour, ls, lw, z = ASICS_STYLE[key]
                ax.plot(t[sl], series[key][sl, k * 3 + j] * scale,
                        color=colour or JOINT_COLORS[j], ls=ls, lw=lw,
                        zorder=z, label=ASICS_LABELS[key])
            ax.set_xlim(t[sl][0], t[sl][-1])
            style_axes(ax, "time (s)" if k == 1 else "",
                       (f"{'RL'[k]} — {ylabel}") if j == 0 else "")
            if k == 0:
                ax.set_title(joints[j], color=NEUTRAL, fontsize=10)
            if handles is None:
                handles = ax.get_legend_handles_labels()
    for j in range(3):                                   # RULE: equal y-axes
        share_ylim(axes[0, j], axes[1, j])
    fig.legend(*handles, frameon=False, fontsize=8.0, ncols=3,
               loc="outside lower center")
    return fig


def original_dependency_status_html():
    """Derive recovery status from the actual reference tree, never a stale list."""
    from pathlib import Path
    from html import escape
    root = Path(__file__).resolve().parents[2] / "matlab-inverse-dynamics"
    requested = {
        "RoomSegAng.m": "Matrix derivatives; TMdot @ TM, not Cardan derivatives.",
        "multiprod.m": "Per-frame matrix/vector multiplication.",
        "matexp.m": "Replicates the local inertia across frames.",
        "VirtualPoint_v2.m": "Recovered 2026-09-11: SODER fit, exports the TRANSPOSED rotation as TM. Ported in boneid.legacy.",
        "getViconForces_Motek.m": "Recovered 2026-09-11: 18 Hz filter, plate choice by mean vertical force, COP from plate-centre offsets; free moment lands in Md's z slot. Ported.",
        "ContactEvents_Motek.m": "Recovered 2026-09-11: 5 Hz-smoothed peaks, 5 N walk-out, right/left by alternating COP. Ported.",
        "G2L.m": "Inverse coordinate transform; on a Tx3x3 TM MATLAB's folded indexing makes it TM*Global, a proper global-to-local transform. Ported.",
        "SIMMpatellaCS.m": "Patella-angle interpolation; scale argument is unused.",
        "TreadOffsets.m": "Constant analog baseline subtraction; call is commented out.",
        "generateMotFile.m": "GRF export; call is commented out.",
    }
    available = {p.name for p in root.rglob("*.m") if p.stat().st_size}
    rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td>"
        f"<td>{'Recovered' if name in available else 'Missing'}</td>"
        f"<td>{escape(note)}</td></tr>" for name, note in requested.items())
    missing = [name for name in requested if name not in available]
    return (f"<p>{len(requested) - len(missing)} of {len(requested)} originally requested "
            "files recovered. Status checked against the reference tree during notebook execution.</p>"
            "<p><b>Still missing:</b> " + (", ".join(f"<code>{escape(n)}</code>" for n in missing)
                                          or "none of the active calls")
            + ". The generateMotFile call is commented out; the whole inverse-dynamics "
              "path is ported in <code>boneid.legacy</code> and run in the c3d report.</p>"
            '<table class="missing-files"><thead><tr><th>Original dependency</th>'
            '<th>Current status</th><th>Assessment</th></tr></thead><tbody>'
            + rows + '</tbody></table>')


def p1_asics_report_html(trial, data, params) -> str:
    """The A/B report: the legacy MATLAB's physics mistakes against the
    corrected pipeline, same subject, same trial, same everything else.

    `data` is `asics_comparison(trial, params)`.
    """
    from . import io_v3d as io
    from .core import detect_contact

    chain = data["chain"]
    skel_r, kin_r = chain[0], chain[1]
    t = kin_r.t
    joints = skel_r.joint_names                     # [ankle, knee, hip]
    sl = slice(EDGE, -EDGE)
    rms, peak = data["rms"], data["peak"]
    imb = data["imbalance"]
    terms = data["terms"]

    plate = {s: io.plate_for_side(trial, s, params.contact_threshold_n)
             for s in SIDES}
    fz = io.analog_to_mocap(trial, trial.force[:, :, 2])
    mask = {s: detect_contact(fz[:, plate[s]], params.contact_threshold_n,
                              min_gap=12)[0] for s in SIDES}
    spans = [contact_spans(mask[s], t) for s in SIDES]
    a, b = stride_window(mask["r"], 1)

    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    # ---- 1. torque overlay, whole trial ----------------------------------
    add(asics_grid(t, {k: data["torque"][k][:, :, 0] for k in ASICS_VARIANTS},
                   joints, "sagittal torque (N m)", spans),
        "Sagittal (lab-x) net joint torque at all six leg joints over the "
        "whole trial: the corrected pipeline in colour, the legacy physics in "
        "grey dashed ((a)+(b) plus hypothetical Cardan α) and violet dotted (the two verified "
        "ones only). Shading is that side's stance. Rows are the two sides, "
        "columns the joint, and <b>the two panels of a column share y-limits "
        "by rule</b>. At this scale the three curves are one curve — which is "
        "the honest headline: in walking, the two real legacy mistakes are "
        f"a {100 * (rms['ab'] / peak).max():.0f}%-of-peak "
        "effect at worst, not a sign error.")

    # ---- 2. one stride, zoomed -------------------------------------------
    add(asics_grid(t, {k: data["torque"][k][:, :, 0] for k in ASICS_VARIANTS},
                   joints, "sagittal torque (N m)", spans, window=(a, b)),
        f"The same six panels over a single right-foot stride "
        f"({t[a]:.2f}–{t[b]:.2f} s, heel-strike to heel-strike). Zoomed, the "
        "separation becomes visible at the hip and knee, mostly in swing "
        "and around toe-off, where the segments are rotating fastest and the "
        "inertial term is a real fraction of the balance. The ankle traces "
        "still overlie each other: the foot's inertia is small enough that "
        "getting its tensor wrong barely moves the ankle moment.")

    # ---- 3. powers --------------------------------------------------------
    add(asics_grid(t, data["power"], joints, "joint power (W)", spans,
                   window=(a, b)),
        "Joint power τ·ω<sub>rel</sub> over the same stride. "
        "<b>ω<sub>rel</sub> is the CORRECT kinematics' relative angular "
        "velocity in all three curves</b> — only the torque differs, so this "
        "figure compares torques and does not double-count the Cardan-angle "
        "mistake through the velocity as well. Power differences track the "
        "torque differences directly, largest at the hip where ω is largest.")

    # ---- 4. the difference itself, decomposed ----------------------------
    fig, axes = new_grid(2, 3, height=4.8, width=8.6, sharex=True)
    dhandles = None
    for k in range(2):
        for j in range(3):
            ax = axes[k, j]
            col = k * 3 + j
            for key, colour, ls in (("ab", "#9a6bd6", (0, (1, 2))),
                                    ("abc", NEUTRAL, (0, (5, 3)))):
                d = data["torque"][key] - data["torque"]["correct"]
                ax.plot(t, d[:, col, 0], color=colour, ls=ls, lw=1.3,
                        label="sagittal Δτ, " + ASICS_LABELS[key])
            d = data["torque"]["abc"] - data["torque"]["correct"]
            ax.plot(t, np.linalg.norm(d[:, col], axis=1),
                    color=JOINT_COLORS[j], lw=1.0, alpha=0.85,
                    label="|Δτ|, all three components, (a)+(b)+(c) "
                          "(joint colour)")
            ax.axhline(0.0, color=FAINT, lw=0.8)
            ax.set_xlim(t[0], t[-1])
            style_axes(ax, "time (s)" if k == 1 else "",
                       f"{'RL'[k]} — Δ torque (N m)" if j == 0 else "")
            if k == 0:
                ax.set_title(joints[j], color=NEUTRAL, fontsize=10)
            if dhandles is None:
                dhandles = ax.get_legend_handles_labels()
    for j in range(3):
        share_ylim(axes[0, j], axes[1, j])
    fig.legend(*dhandles, frameon=False, fontsize=8.0, ncols=2,
               loc="outside lower center")
    add(fig,
        "<b>The decomposition — the most informative panel here.</b> "
        "Legacy minus corrected, sagittal component (dotted/dashed) and the "
        "magnitude of the full 3-vector difference (coloured). The two "
        "verified mistakes alone, (a) one-sided R·I and (b) the missing "
        "gyroscopic term, give the dotted curve; adding the hypothetical "
        "Cardan-angle α gives the dashed one. They are of the same order and "
        "they partially CANCEL: at the right hip the (a)+(b) error is "
        f"{rms['ab'][2]:.1f} N m RMS and (a)+(b)+(c) is {rms['abc'][2]:.1f} "
        f"N m, even though (c) on its own is {rms['c'][2]:.1f} N m — larger "
        "than either. That is "
        "luck, not structure — nothing makes a Cardan-angle second derivative "
        "cancel a non-symmetric inertia tensor.")

    # ---- 5. the audit as a bug detector -----------------------------------
    fig, axes = new_grid(1, 2, height=3.0, width=8.6)
    for key in ASICS_VARIANTS:
        colour, ls, lw, z = ASICS_STYLE[key]
        axes[0, 0].plot(t[sl], data["audit"][key].imbalance[sl],
                        color=colour or JOINT_COLORS[0], ls=ls, lw=lw,
                        zorder=z, label=ASICS_LABELS[key])
    style_axes(axes[0, 0], "time (s)", "energy imbalance dE/dt − ΣP (W)")
    axes[0, 0].legend(frameon=False, fontsize=7.4)
    bars = [imb[k]["rms"] for k in ASICS_VARIANTS]
    axes[0, 1].bar(np.arange(3), bars,
                   color=[JOINT_COLORS[2], "#9a6bd6", NEUTRAL], width=0.6)
    axes[0, 1].set_xticks(np.arange(3))
    axes[0, 1].set_xticklabels(["corrected", "(a)+(b)", "(a)+(b)+(c)"])
    style_axes(axes[0, 1], "", "imbalance, RMS (W)")
    add(fig,
        "<b>The energy audit, used as a bug detector.</b> The audit code is "
        "the corrected one in all three cases — correct inertia transform, "
        "correct rotational kinetic energy — and only the joint torques handed "
        "to it are swapped. d(KE+PE)/dt must equal the ground power plus the "
        "eleven joint powers plus the residual power; with correct torques it "
        f"does, to {imb['correct']['rms']:.1f} W RMS "
        f"({100 * imb['correct']['rms'] / imb['correct']['scale']:.2f}% of "
        "peak dE/dt), which is finite-difference noise. With the two real "
        f"legacy mistakes the same instrument reads {imb['ab']['rms']:.1f} W — "
        f"{imb['ab']['rms'] / imb['correct']['rms']:.0f}× worse. No reference "
        "data, no Visual3D, no ground truth was needed to see it: an "
        "independent energy balance detects a broken Euler equation on its "
        "own.")

    # ---- tiles ------------------------------------------------------------
    # RULE (CLAUDE.md): bilateral quantities are organised L | R per joint.
    # JOINT_LABELS order is (R ankle, R knee, R hip, L ankle, L knee, L hip),
    # so joint j pairs indices (j + 3, j) as (left, right).
    tiles_rms = "".join([
        stat(f"{name} Δτ RMS  L | R",
             f"{rms['abc'][j + 3]:.2f} | {rms['abc'][j]:.2f}",
             f"N m — {100 * rms['abc'][j + 3] / peak[j + 3]:.1f}% | "
             f"{100 * rms['abc'][j] / peak[j]:.1f}% of peak")
        for j, name in enumerate(("ankle", "knee", "hip"))])
    tiles = "".join([
        stat("energy imbalance, corrected", f"{imb['correct']['rms']:.1f}",
             "W RMS", "pass"),
        stat("energy imbalance, asics", f"{imb['abc']['rms']:.1f}",
             f"W RMS ({imb['abc']['rms'] / imb['correct']['rms']:.0f}× worse)",
             "fail"),
        stat("largest peak Δτ",
             f"{data['peak_diff']['abc'].max():.1f}",
             f"N m at {JOINT_LABELS[int(np.argmax(data['peak_diff']['abc']))]}"),
        stat("worst joint, % of peak",
             f"{100 * (rms['abc'] / peak).max():.1f}",
             f"% at {JOINT_LABELS[int(np.argmax(rms['abc'] / peak))]}"),
    ])

    rows = "".join(
        f"<tr><td>{JOINT_LABELS[j]}</td><td>{peak[j]:.0f}</td>"
        f"<td>{rms['ab'][j]:.2f}</td>"
        f"<td>{100 * rms['ab'][j] / peak[j]:.2f}</td>"
        f"<td>{rms['abc'][j]:.2f}</td>"
        f"<td>{100 * rms['abc'][j] / peak[j]:.2f}</td>"
        f"<td>{data['peak_diff']['abc'][j]:.2f}</td></tr>"
        for j in range(6))
    rms_table = (
        '<div class="overflow"><table><thead><tr><th>joint</th>'
        "<th>peak |τ| (N m)</th><th>(a)+(b) RMS Δτ (N m)</th><th>% of peak</th>"
        "<th>(a)+(b)+(c) RMS Δτ (N m)</th><th>% of peak</th>"
        f"<th>peak |Δτ| (N m)</th></tr></thead><tbody>{rows}</tbody></table>"
        "</div>")

    comp_rows = "".join(
        f"<tr><td>{JOINT_LABELS[j]}</td>"
        + "".join(f"<td>{data['rms_xyz']['abc'][j, c]:.2f}</td>"
                  for c in range(3))
        + "".join(f"<td>{100 * data['rms_xyz']['abc'][j, c] / max(np.abs(data['torque']['correct'][sl, j, c]).max(), 1e-9):.1f}</td>"
                  for c in range(3))
        + "</tr>" for j in range(6))
    comp_table = (
        '<div class="overflow"><table><thead><tr><th>joint</th>'
        "<th>Δτ<sub>x</sub></th><th>Δτ<sub>y</sub></th><th>Δτ<sub>z</sub></th>"
        "<th>% peak x</th><th>% peak y</th><th>% peak z</th>"
        f"</tr></thead><tbody>{comp_rows}</tbody></table></div>")

    single_rows = "".join(
        f"<tr><td>{label}</td>"
        + "".join(f"<td>{rms[key][j]:.2f}</td>" for j in range(6)) + "</tr>"
        for key, label in (
            ("a", "(a) one-sided R·I only"),
            ("b", "(b) no gyroscopic term only"),
            ("c", "(c) Cardan-angle α only (hypothetical)"),
            ("ab", "(a)+(b) — both verified mistakes"),
            ("abc", "(a)+(b) plus hypothetical Cardan α")))
    single_table = (
        '<div class="overflow"><table><thead><tr><th>mistake</th>'
        + "".join(f"<th>{name}</th>" for name in JOINT_LABELS)
        + f"</tr></thead><tbody>{single_rows}</tbody></table></div>")

    term_rows = "".join(
        f"<tr><td>{name}</td><td>{v['i_alpha']:.2f}</td>"
        f"<td>{v['gyro']:.3f}</td><td>{v['one_sided']:.2f}</td>"
        f"<td>{v['cardan']:.2f}</td>"
        f"<td>{v['omega_frac'][0]:.2f} / {v['omega_frac'][1]:.2f} / "
        f"{v['omega_frac'][2]:.2f}</td></tr>"
        for name, v in terms.items())
    term_table = (
        '<div class="overflow"><table><thead><tr><th>segment (right leg)</th>'
        "<th>peak |I<sub>w</sub>α| (N m)</th>"
        "<th>peak |ω×(I<sub>w</sub>ω)| (N m)</th>"
        "<th>peak |(R I − R I R<sup>T</sup>)α| (N m)</th>"
        "<th>peak |R I (α<sub>Cardan</sub> − α)| (N m)</th>"
        "<th>mean |ω<sub>local</sub>| fraction x / y / z</th>"
        f"</tr></thead><tbody>{term_rows}</tbody></table></div>")

    gyro_max = max(v["gyro"] for v in terms.values())
    one_max = max(v["one_sided"] for v in terms.values())
    ml_frac = terms["thigh"]["omega_frac"][1]
    worst = int(np.argmax(rms["abc"] / peak))
    speed = belt_speed(trial, params.contact_threshold_n)

    body = f"""
<p class="eyebrow">Report 4 · boneid · legacy A/B · subject P1 trial {trial.index}</p>
<h1>P1 walking — legacy ASICS equation comparison</h1>
<p class="subtitle">Two legacy equation errors plus a separate hypothetical Cardan experiment,
measured on real walking data at {speed:.2f} m/s. Same subject, same
trial, same anthropometry, same joint centres, same forces, same filtering,
same chain — <b>only the equations differ</b>.</p>

<div class="stat-row">{tiles_rms}</div>
<div class="stat-row">{tiles}</div>

<h2>Two verified changes and a sensitivity experiment</h2>
<p>The legacy pipeline (<code>Asics_Moments4JCF_v3.m</code> plus
<code>inv3d.m</code> and <code>RoomSegAng.m</code>) contains the following
three equation errors. The plots reproduce (a) and (b); error 3 is not reproduced:</p>
<ul>
<li><b>(a) The inertia tensor is rotated one-sidedly.</b> VERIFIED, at
<code>Asics_Moments4JCF_v3.m</code>, lines <b>370, 372, 374, 376, 378, 380</b>
(rearfoot, forefoot, foot, shank, thigh, pelvis respectively):
<code>IfootG = multiprod(R_FootTM, Ifoot)</code> — that is
I<sub>G</sub> = R I<sub>local</sub>, not the similarity transform
R I<sub>local</sub> R<sup>T</sup>. The result is not symmetric, so it is not
an inertia tensor at all (same Frobenius norm, but it maps ω to the wrong
angular momentum, magnitude and direction both).</li>
<li><b>(b) Euler's equation is missing its gyroscopic term.</b> VERIFIED, at
<code>Inverse Dynamics Subroutines/inv3d.m</code>, line <b>38</b>: the moment balance is
<code>multiprod(I,aANG)</code> and nothing else, so
ω × (I<sub>w</sub>ω) is simply absent. The correct balance is
d/dt(I<sub>w</sub>ω) = I<sub>w</sub>α + ω × (I<sub>w</sub>ω).</li>
<li><b>Error 3: incorrect angular-velocity matrix product.</b>
The calls are in <code>Asics_Moments4JCF_v3.m</code>, lines <b>442–447</b>.
In <code>originals/RoomSegAng.m</code>, lines <b>10–11</b> differentiate TM;
line <b>12</b> computes <code>TMdot * TM</code> without a transpose.
For a rotation R from segment to world, the correct products are
<code>Rdot * Rᵀ</code> (world angular velocity) or <code>Rᵀ * Rdot</code>
(body angular velocity). Reversing the frame convention changes the order/sign;
it does not make <code>TMdot * TM</code> generally correct.
Lines <b>26–28</b> extract components from this generally non-skew matrix,
and line <b>32</b> differentiates them into angular acceleration.
With <code>VirtualPoint_v2.m</code> recovered (TM is the transposed SODER
rotation), the full legacy path is reproduced in <code>boneid.legacy</code> and
run against the corrected pipeline on the Asics running data in
<code>asics_running_c3d</code>.</li>
</ul>
<p><b>Separate hypothetical experiment (c).</b> RoomSegAng does not differentiate
Cardan angles. Curves labelled (c) test that hypothetical approximation and
<b>do not measure error 3 above</b>. <code>cardan_alpha=False</code> isolates
(a)+(b); neither setting reproduces RoomSegAng.</p>
<p><b>What is <em>not</em> reproduced.</b> The legacy also had genuine
anthropometric typos — <code>Ileg = [Ilx 0 0; 0 Ify 0; 0 0 Ilz]</code> puts the
<em>foot's</em> transverse moment into the shank tensor, and the thigh's
<code>Ity</code> regression is fed the foot length <code>A(6)</code>. Those are
data-entry errors, not physics. Reproducing them would turn this into a
different-numbers comparison instead of a same-numbers-different-equations
one, so <code>inverse_dynamics_whole_body_asicssource</code> keeps the
corrected de Leva anthropometrics throughout and changes (a), (b), and the
separate hypothetical Cardan approximation (c).</p>
<p>The A/B is therefore exact: both arms of it are the same function call
(<code>core.inverse_dynamics_whole_body</code>) with a flags record set
differently. Joint <em>forces</em> come out bit-identical — F = m a − F<sub>d</sub>
− m g contains no inertia tensor — which is a useful check that nothing else
drifted.</p>

<h2>What it costs, joint by joint</h2>
{rms_table}
<p>RMS is over the 3-vector torque difference, so no component is hidden, over
the whole trial minus {EDGE} frames at each end. The (a)+(b) columns are the
real legacy mistakes; the (a)+(b)+(c) columns add the hypothetical Cardan term
and are here only as a sensitivity check. For scale, this same pipeline
reproduces Visual3D's ankle and knee torques to about 1.5–2 N m on this
subject. So the two real mistakes at the right ankle ({rms['ab'][0]:.2f} N m)
sit <em>below</em> that agreement floor, and at the right hip
({rms['ab'][2]:.1f} N m) they sit well above it.</p>

{figs[0]}
{figs[1]}
{figs[2]}

<h2>Which mistake dominates</h2>
{figs[3]}
<p>Running each flag <em>separately</em> — same call, one mistake at a time —
gives the answer plainly. RMS torque difference from the corrected pipeline,
N m:</p>
{single_table}
<p>At the right hip: (a) alone costs {rms['a'][2]:.1f} N m, (b) alone
{rms['b'][2]:.2f} N m, (c) alone {rms['c'][2]:.1f} N m. So <b>the missing
gyroscopic term (b) is nearly free in walking</b>, the one-sided inertia
rotation (a) is the expensive verified mistake, and the hypothetical (c) is of the
same order as (a) again — but they partially cancel, which is why all three
together ({rms['abc'][2]:.1f} N m) come out <em>smaller</em> than (a) and (c)
would suggest. That cancellation is luck, not structure. <b>Of the two
mistakes we are certain about, (a) is the one that matters.</b></p>
{term_table}
<p>The term magnitudes say why. The gyroscopic term the legacy dropped peaks at
{gyro_max:.2f} N m over the whole right leg — against an I<sub>w</sub>α term
that peaks at {terms['thigh']['i_alpha']:.1f} N m at the thigh and a joint
torque that peaks at {peak[2]:.0f} N m. The one-sided-inertia error is
{one_max / max(gyro_max, 1e-9):.0f}× larger.</p>

<h2>Honest about the magnitude</h2>
<p>These are small numbers, and it is worth being precise about why rather
than reciting the usual explanation. Two hypotheses, both checkable here:</p>
<ul>
<li><b>"Walking is near-planar, so ω lies close to a principal axis and
ω × (I ω) nearly vanishes." TRUE.</b> The last column of the table above is
the mean fraction of each segment's angular velocity along its own local axes:
the thigh spends {100 * ml_frac:.0f}% of |ω| on the medio-lateral axis, which
is a principal axis of a diagonal de Leva tensor, and for ω exactly along a
principal axis ω × (Iω) is identically zero. Hence a gyroscopic term of
{gyro_max:.2f} N m in a {peak[2]:.0f} N m signal. Mistake (b) is genuinely
harmless in walking — which is not a defence of the code, it is a statement
about this task. Run, cut, or throw and it grows.</li>
<li><b>"The one-sided R·I error partially cancels in the sagittal
component." PARTLY TRUE, and it is the smaller effect.</b> The per-component
table below shows the error is <em>not</em> concentrated in the sagittal
(lab-x) component that gait papers report; relative to each component's own
peak it is smallest there and largest out of plane, where the torques are
small and nobody looks.</li>
</ul>
{comp_table}
<p>So the real reason the legacy pipeline produced publishable-looking sagittal
moment curves is not that its physics was nearly right. It is that in level
walking the whole inertial term is a small fraction of the balance — the
segment weights and the ground reaction dominate — so any error <em>inside</em>
the inertial term is a small fraction of a small fraction, and it lands mostly
in the components that never get plotted. A movement with fast three-
dimensional segment rotation (a cutting manoeuvre, a throw, a fall) would put
ω well off the principal axes and the same code would be visibly wrong.</p>

<h2>The audit catches it without a reference</h2>
{figs[4]}
<p>This is the argument for building the energy audit in the first place. It
uses no reference data and no ground truth: it just asks whether the rate of
change of the model's own mechanical energy equals the power flowing into it
through the joints, the ground and the residual. Correct torques close it to
{imb['correct']['rms']:.1f} W RMS; the two real legacy mistakes (a)+(b) leave
{imb['ab']['rms']:.1f} W unaccounted for (and {imb['abc']['rms']:.1f} W with the
hypothetical Cardan term on top). Nothing about this check is new or hard. The
legacy code just never had it, so the imbalance was sitting there the whole
time.</p>

<h2>Reproducing this</h2>
<p><code>uv run python examples/regenerate_reports.py p1_asics</code> regenerates this page.
The comparison itself is
<code>core.inverse_dynamics_whole_body_asicssource(skel_r, kin_r, ground_r,
skel_l, kin_l, ground_l, skel_u, kin_u, params, cardan_alpha=True)</code> —
same signature and same return type as
<code>core.inverse_dynamics_whole_body</code>, so anything that consumes one
consumes the other, including the audit. <code>tests/test_asicssource.py</code>
pins the behaviour: on the KEY-GYRO simulation, whose entire required torque
<em>is</em> ω × (I<sub>w</sub>ω), the asicssource path returns approximately
zero — proof that the reimplementation really does reproduce the bug rather
than approximate it.</p>

<h2 id="appendix-missing-files">Appendix A — Original dependency recovery status</h2>
{original_dependency_status_html()}
<p><b>For the remaining missing routines, supply the versions used for the original analysis</b>, together
with their helper files, toolbox/package versions and MATLAB path setup.
A copy of the actual batch script used by the lab would let us check whether
it differs from this checkout.</p>
<p>For an end-to-end assessment, also request one matched static/dynamic C3D
trial pair, its subject anthropometrics and force-plate calibration/offset
settings, and the corresponding saved MATLAB outputs (ideally segment
rotations, angular velocities/accelerations, global/local moments and powers).
These are supporting materials to request, not additional named files whose
absence has been established here. They would allow us to compare recovered
code with the original outputs and then reassess the report’s assumptions.</p>
<style>
  .missing-files {{ width: 100%; min-width: 680px; }}
  .missing-files th, .missing-files td {{ text-align: left; vertical-align: top; }}
  .missing-files td:first-child {{ width: 25%; }}
</style>
"""
    return html_page("P1 walking — legacy ASICS equation comparison", "", body)


def main_asics(argv=None):
    """Compatibility alias for the canonical notebook build."""
    if argv:
        raise ValueError("Set study inputs in notebooks/p1_asics.ipynb")
    from .notebook_reports import main as build_reports
    return build_reports(["p1_asics"])


def main_v3d(argv=None):
    """Compatibility alias for the canonical notebook build."""
    if argv:
        raise ValueError("Set study inputs in notebooks/p1_report.ipynb")
    from .notebook_reports import main as build_reports
    return build_reports(["p1_report"])


def main(argv=None):
    """`python -m boneid.report [v3d|asics [matpath [trial_index]]]`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "v3d":
        main_v3d(argv[1:])
    elif argv and argv[0] == "asics":
        main_asics(argv[1:])
    else:
        from .notebook_reports import main as build_reports
        build_reports([])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Quickstart report (examples/quickstart.py — the 10-step walkthrough)
# ---------------------------------------------------------------------------

def quickstart_report_html(trial, body_mass, params, chains, upper, contact,
                           crossover, whole, audit,
                           viewer_html: str | None = None) -> str:
    """Compact report for the 10-step quickstart: the essential outputs of a
    whole-body inverse-dynamics run and nothing else. `chains` maps side ->
    (skeleton, kin, ground); `upper` is `(skel_u, kin_u)` from
    `io_v3d.build_upper_body`; `contact` maps side -> (mask, events);
    `crossover` maps side -> bool flags; `whole` is
    `core.inverse_dynamics_whole_body`; `viewer_html` is the standalone
    meshcat page from `p1_viewer_html`, embedded as the 3-D animation."""
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
             f"the torso's centre of mass and averages {res_mean[2]:.1f} N "
             f"vertically. The mean is near zero because no body part is "
             f"missing any more; the swings around it are not, and that's "
             f"soft tissue and joint-centre error (see the P1 report). Middle: the L5/S1 joint wrench (torso on pelvis), whose "
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

    viewer = ""
    if viewer_html is not None:
        esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
        viewer = (
            '<h2>3-D animation</h2>'
            '<p>The whole body in motion: both legs, the pelvis spanning the '
            'hips, the mass-sized torso, both arms, every marker (red), and '
            'the ground-reaction force arrow per belt. Drag to orbit, scroll '
            'to zoom; the clip loops.</p>'
            f'<iframe class="viewer" srcdoc="{esc}"></iframe>')

    body = f"""
<p class="eyebrow">boneid · quickstart · step 10</p>
<h2>The report</h2>
<p class="subtitle">Whole body, twelve segments, trial {trial.index},
{len(t) / kin_r.rate:.1f} s at {kin_r.rate:.0f} Hz. The four numbers I look
at first, then the body moving, then the figures.</p>

<div class="stat-row">{stats}</div>

{viewer}

{"".join(figs)}

<h2>Reproduce</h2>
<p><code>uv run python examples/quickstart.py</code> re-runs the notebook
and rewrites this page. For your own study, edit the config cell in
<code>notebooks/quickstart_report.ipynb</code>; the ten steps don't change.</p>
"""
    return html_page("Quickstart report", "qs", body)


# ---------------------------------------------------------------------------
# Static-calibrated c3d study (io_c3d): report + viewer
# ---------------------------------------------------------------------------

def right_step_events(ground, threshold_n: float, min_s: float = 0.1,
                      max_s: float = 0.6):
    """(onset, offset) mocap-frame pairs of clean instrumented-foot stances.

    Runs `detect_contact` on the (already attributed) ground wrench's vertical
    force and keeps bouts with plausible stance duration that do not touch
    the record edges."""
    from .core import detect_contact
    _, events = detect_contact(ground.force[:, 2], threshold_n,
                               min_gap=int(round(0.05 * ground.rate)))
    n = len(ground.t)
    out = []
    for s, e in events:
        dur = (e - s) / ground.rate
        if min_s <= dur <= max_s and s > 0 and e < n - 1:
            out.append((s, e))
    return out


def stance_profiles(y: np.ndarray, events, n: int = 101) -> np.ndarray:
    """Time-normalize `y` [T(,C)] over each (s,e) stance to `n` samples.

    Returns [K, n(, C)] — one row per stance, 0..100% of contact."""
    y = np.asarray(y, dtype=float)
    grid = np.linspace(0.0, 1.0, n)
    rows = []
    for s, e in events:
        seg = y[s:e + 1]
        x = np.linspace(0.0, 1.0, len(seg))
        if seg.ndim == 1:
            rows.append(np.interp(grid, x, seg))
        else:
            rows.append(np.stack([np.interp(grid, x, seg[:, c])
                                  for c in range(seg.shape[1])], axis=1))
    return np.stack(rows)


def _slice_kin(kin, sl: slice):
    from dataclasses import replace
    return replace(kin, t=kin.t[sl], r_world=kin.r_world[sl],
                   prox_pos=kin.prox_pos[sl], dist_pos=kin.dist_pos[sl])


def _slice_wrench(ground, sl: slice):
    from dataclasses import replace
    return replace(ground, t=ground.t[sl], force=ground.force[sl],
                   moment=ground.moment[sl])


def c3d_viewer_window(ground, params, n_steps: int = 6) -> slice:
    """A mocap-frame window spanning the first `n_steps` clean right stances,
    padded half a stride either side — enough to see the gait without
    shipping the whole 30 s trial in the animation."""
    ev = right_step_events(ground, params.contact_threshold_n)[:n_steps]
    if not ev:
        return slice(0, len(ground.t))
    pad = int(0.4 * ground.rate)
    return slice(max(0, ev[0][0] - pad), min(len(ground.t), ev[-1][1] + pad))


def c3d_decimate(kin) -> int:
    """Animation frame decimation for ~30 fps playback (10 on 300 Hz data)."""
    return max(1, int(round(kin.rate / 30.0)))


def c3d_foot_wedge(cal):
    """`viz.foot_wedge_mesh` sized from the static calibration.

    The wedge triangle is (ankle, heel, toebox tip), all three measured in
    the static pose and expressed in the foot frame relative to the ankle;
    the width is the measured MTP1-MTP5 distance.
    """
    from .viz import foot_wedge_mesh
    r0 = cal.r_static[0]
    ankle = cal.points["ankle"]
    tip_local = r0.T @ (cal.points["toebox"] - ankle)
    return foot_wedge_mesh(cal.foot_heel_local, tip_local,
                           cal.lengths["forefoot_width"])


def c3d_foot_blob(cal):
    """Rounded foot from static heel, forefoot and ankle landmarks."""
    from .viz import foot_blob_mesh
    if cal.foot_outline_local is None:
        # Older callers may supply a calibration without drawing landmarks.
        return c3d_foot_wedge(cal)
    return foot_blob_mesh(cal.foot_outline_local)


def c3d_viewer_html(trial, cal, skeleton, kin, ground, quality, params,
                    n_steps: int = 6, repetitions: int = 10000) -> str:
    """Meshcat snapshot of the instrumented leg + pelvis over a few steps.

    One chain [foot, shank, thigh, pelvis] with a real-width pelvis band
    spanning the two (GT-regression) hip centres, the attributed GRF arrow,
    every named marker (red) animated along, and the foot drawn as a rigid
    rounded landmark envelope bounded by the static heel, forefoot, toebox
    and ankle markers.  The clip loops, and the
    viewer's Animations panel (Open Controls) can pause and scrub it — the
    report's strip charts listen to that same clock.
    """
    from . import viz

    sl = c3d_viewer_window(ground, params, n_steps)
    kin_w = _slice_kin(kin, sl)
    ground_w = _slice_wrench(ground, sl)
    pts = quality["points"]

    shift = viewer_frame_shift(kin_w)
    vkin = shift_kinematics(kin_w, shift)
    vground = shift_wrench(ground_w, shift)
    pelvis_span = (pts["hip_l"][sl] + shift, pts["hip"][sl] + shift,
                   pts["l5s1"][sl] + shift)
    traj = {name: xyz[sl] + shift for name, xyz in trial.markers.items()}
    floor = float(vkin.dist_pos[:, 0, 2].min())
    decimate = c3d_decimate(kin)

    vis = viz.start_viewer()
    try:
        animation = viz.animate(vis, skeleton, vkin, ground=vground,
                                decimate=decimate, repetitions=1, play=False,
                                plane_height=floor,
                                force_threshold=params.contact_threshold_n,
                                pelvis_span=pelvis_span,
                                segment_meshes={0: c3d_foot_blob(cal)})
        viz.draw_markers(vis, traj, 0)
        for k, f in enumerate(range(0, len(vkin.t), decimate)):
            with animation.at_frame(vis, k) as frame:
                for name, xyz in traj.items():
                    frame[f"{viz.ROOT}/markers/{name}"].set_transform(
                        viz.translation(xyz[f]))
        vis["/Axes"].set_property("visible", False)
        vis.set_animation(animation, play=True, repetitions=repetitions)
        viz.set_camera(vis, target=(0.0, 0.0, 0.0), offset=(1.55, -1.55, 0.35))
        return viz.render_static_html(vis)
    finally:
        viz.stop_viewer(vis)


def c3d_belt_velocity(trial, ground, threshold_n: float = 50.0) -> np.ndarray:
    """Lab-frame belt surface velocity [3], measured (see `belt_velocity`).

    Median horizontal velocity vector of the heel-marker midpoint over the
    frames where the ATTRIBUTED (right-foot) wrench shows contact — during
    stance the foot rides the belt.  Vertical component dropped.
    """
    from .core import finite_difference
    heel = 0.5 * (trial.markers["HeelMed"] + trial.markers["HeelLat"])
    v = finite_difference(heel, trial.rate)
    mask = ground.force[:, 2] > threshold_n
    med = np.median(v[mask], axis=0)
    return np.array([med[0], med[1], 0.0])


def c3d_sagittal_series(cal, skeleton, kin, idres, params) -> dict:
    """Sagittal-plane angle / velocity / torque [T,3] for ankle, knee, hip.

    All three quantities are taken about the SAME axis — the proximal
    segment's medio-lateral (local y) axis:

    * ``angle`` [deg]: rotation of the distal segment's long axis in the
      proximal frame's sagittal (x-z) plane, ZEROED AT THE STATIC POSE
      (that is what the static trial buys: 0 deg = standing).  Positive =
      the distal long axis tilted anteriorly relative to standing.
    * ``vel`` [deg/s]: relative angular velocity (omega_dist - omega_prox)
      projected on the proximal y axis.
    * ``torque`` [N m]: ``joint_torque_local[..., 1]`` — the net joint torque
      in the proximal segment frame, y component.

    Kinematics are filtered with the same `params` as the dynamics.

    `cal=None` (data with no static trial, e.g. the Visual3D P1 export):
    the angle is NOT zeroed — it is the raw sagittal tilt of the distal long
    axis in the proximal frame, so knee/hip read ~0 when the segments line
    up and the ankle reads the foot's (heel->5th met) tilt off the shank
    axis, ~ +80..125 deg on P1.  No neutral pose is invented.
    """
    from .core import chain_kinematics
    ck = chain_kinematics(skeleton, kin, params.lowpass_hz,
                          params.filter_order)
    r = ck["r"]
    angle = np.zeros((r.shape[0], 3))
    vel = np.zeros_like(angle)
    for j in range(3):
        r_rel = np.einsum("tji,tjk->tik", r[:, j + 1], r[:, j])
        v = r_rel[:, :, 2]                       # distal z in prox frame
        theta = np.degrees(np.arctan2(v[:, 0], v[:, 2]))
        theta0 = 0.0
        if cal is not None:
            r0 = cal.r_static[j + 1].T @ cal.r_static[j]
            theta0 = np.degrees(np.arctan2(r0[0, 2], r0[2, 2]))
        angle[:, j] = (theta - theta0 + 180.0) % 360.0 - 180.0
        w_rel = ck["omega"][:, j] - ck["omega"][:, j + 1]
        vel[:, j] = np.degrees(np.sum(w_rel * r[:, j + 1, :, 1], axis=1))
    return {"angle": angle, "vel": vel,
            "torque": idres.joint_torque_local[:, :3, 1]}


#: strip-grid geometry (the header row reuses the STRIP_* constants)
GRID_GUTTER = 66                #: left gutter for row labels + shared y ticks
GRID_CELL_H = 78
GRID_GAP = 10


def strip_charts_grid(t, header, grid_rows, col_titles, period_s: float,
                      fps: float, n_frames: int, uid: str = "grid",
                      window_s: float = 1.6,
                      viewer_iframe: str | None = None,
                      variants: list[str] | None = None,
                      variant_root: str = "/meshcat/boneid/v",
                      variant_time_scale: list[float] | None = None,
                      variant_ref: int = 0,
                      phase_ticks: bool = False) -> str:
    """A scrolling-scope GRID synchronised to the animation clock.

    Layout: one full-width header panel (`header = (ylabel, [(label, color,
    dash, y), ...])`) above a `len(grid_rows)` x `len(col_titles)` grid where
    `grid_rows[i] = (row_label, [(color, y), ...])` holds ONE series per
    column and every cell in a row shares that row's y-scale (same quantity,
    same axis — the grid analogue of the bilateral shared-axes rule).

    Mechanics are `strip_charts`' (see there): every panel's traces are drawn
    once, pre-shifted so the sample at time tau sits on the panel's centre
    line under a common translation of ``-tau * px_per_s``, tiled once each
    side so the loop wraps, and a single `requestAnimationFrame` loop drives
    every panel — from the meshcat viewer's own clock when `viewer_iframe` is
    given (pause freezes the grid, the time scrubber drags it), else from the
    wall clock at the clip period.  One px_per_s everywhere: the header is
    simply a wider window (~3x `window_s`) onto the same stream.

    `variants` (labels) turns on a SLIDER above the grid: any series whose y
    is 2-D ``[len(variants), T]`` is drawn once per variant and the slider
    shows one at a time, with variant 0 kept as a faint dashed reference so
    the shift is visible.  Every panel keeps one y-scale across all variants.
    With `viewer_iframe`, the slider also flips meshcat's
    ``{variant_root}{k}`` group visible (the matching body, see
    `p1_viewer_html(knee_shifts_m=...)`).  `variant_time_scale` (one per
    variant) also sets the viewer clip's playback rate on each pick, so a
    phase-normalised clip plays each variant at its own real speed while the
    grid, which follows the clip's own clock, stays in phase.
    `variant_ref` is the variant shown first and drawn as the faint dashed
    reference.  `phase_ticks` labels the header in % of the period and the
    "now" readout in %, for one-stride (phase) data.
    """
    t = np.asarray(t, float)
    span = float(t[-1] - t[0])
    n_cols = len(col_titles)
    x1 = STRIP_WIDTH - STRIP_PAD_R
    cell_w = (x1 - GRID_GUTTER - (n_cols - 1) * GRID_GAP) / n_cols
    pxs = cell_w / float(window_s)
    data_w = span * pxs

    # vertical layout: header, tick row, column titles, grid rows
    y_head = 0
    y_ticks = y_head + STRIP_HEIGHT            # scrolling seconds live here
    y_titles = y_ticks + 22
    y_grid = y_titles + 14
    total_h = y_grid + len(grid_rows) * (GRID_CELL_H + GRID_GAP) + 4

    panels = []          # (x0, ytop, w, h, series[(color, dash, y)], lo, hi)
    hdr_label, hdr_series = header
    lo = min(float(np.nanmin(y)) for *_, y in hdr_series)
    hi = max(float(np.nanmax(y)) for *_, y in hdr_series)
    panels.append((STRIP_PAD_L, y_head, x1 - STRIP_PAD_L, STRIP_HEIGHT,
                   [(c, d, y) for _, c, d, y in hdr_series], lo, hi))
    for r_i, (_, cols) in enumerate(grid_rows):
        lo = min(float(np.nanmin(y)) for _, y in cols)
        hi = max(float(np.nanmax(y)) for _, y in cols)
        ytop = y_grid + r_i * (GRID_CELL_H + GRID_GAP)
        for c_i, (color, y) in enumerate(cols):
            x0 = GRID_GUTTER + c_i * (cell_w + GRID_GAP)
            panels.append((x0, ytop, cell_w, GRID_CELL_H,
                           [(color, None, y)], lo, hi))

    out = [f'<svg id="{uid}-svg" viewBox="0 0 {STRIP_WIDTH} {total_h}" '
           f'width="100%" role="img" aria-label="time-locked strip grid">']
    for p_i, (x0, ytop, w, h, series, lo, hi) in enumerate(panels):
        pad = 0.06 * max(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        ybot = ytop + h
        out.append(f'<rect x="{x0:.1f}" y="{ytop}" width="{w:.1f}" '
                   f'height="{h}" fill="{SURFACE}" stroke="{FAINT}" '
                   f'stroke-width="0.8"/>')
        if lo < 0 < hi:
            zy = ybot - h * (0 - lo) / (hi - lo)
            out.append(f'<line x1="{x0:.1f}" y1="{zy:.1f}" '
                       f'x2="{x0 + w:.1f}" y2="{zy:.1f}" stroke="{FAINT}" '
                       f'stroke-width="0.7"/>')
        content = []
        xa = x0 + 0.5 * w                       # "now" sits here
        for color, dash, y in series:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            y = np.asarray(y, float)
            if y.ndim == 1:
                content.append(
                    f'<polyline fill="none" stroke="{color}" '
                    f'stroke-width="1.2"{d} points="'
                    f'{_strip_path(t, y, xa, xa + data_w, ytop, ybot, lo, hi, max_pts=1600)}"/>')
                continue
            # one polyline per variant; the reference also as a faint dash
            content.append(
                f'<polyline fill="none" stroke="{FAINT}" stroke-width="1" '
                f'stroke-dasharray="3 2" points="'
                f'{_strip_path(t, y[variant_ref], xa, xa + data_w, ytop, ybot, lo, hi, max_pts=1600)}"/>')
            for v in range(y.shape[0]):
                show = "" if v == variant_ref else ' style="display:none"'
                content.append(
                    f'<polyline class="{uid}-v{v}"{show} fill="none" '
                    f'stroke="{color}" stroke-width="1.3"{d} points="'
                    f'{_strip_path(t, y[v], xa, xa + data_w, ytop, ybot, lo, hi, max_pts=1600)}"/>')
        if p_i == 0:                            # header: scrolling ticks
            if phase_ticks:
                ticks = [(t[0] + q * span, f"{100 * q:.0f}%")
                         for q in (0.0, 0.25, 0.5, 0.75)]
            else:
                ticks = [(float(sec), f"{sec:d}")
                         for sec in range(int(np.ceil(t[0])),
                                          int(np.floor(t[-1])) + 1)]
            for tt, lab in ticks:
                tx = xa + (tt - t[0]) * pxs
                content.append(
                    f'<line x1="{tx:.1f}" y1="{y_ticks}" x2="{tx:.1f}" '
                    f'y2="{y_ticks + 4}" stroke="{NEUTRAL}" '
                    f'stroke-width="0.8"/>'
                    f'<text x="{tx:.1f}" y="{y_ticks + 15}" '
                    f'text-anchor="middle" font-size="10" '
                    f'fill="{NEUTRAL}">{lab}</text>')
        clip_h = (y_ticks + 18 - ytop) if p_i == 0 else h
        # the loop-wrap copies are <use> clones: same live content (so the
        # variant slider's display toggles reach them too), a third the bytes
        pan = (f'<g id="{uid}-c{p_i}">' + "".join(content) + "</g>"
               + "".join(f'<use href="#{uid}-c{p_i}" '
                         f'transform="translate({k * data_w:.2f},0)"/>'
                         for k in (-1, 1)))
        out.append(f'<clipPath id="{uid}-clip{p_i}"><rect x="{x0:.1f}" '
                   f'y="{ytop}" width="{w:.1f}" height="{clip_h}"/>'
                   f'</clipPath>')
        out.append(f'<g clip-path="url(#{uid}-clip{p_i})">'
                   f'<g class="{uid}-pan">{pan}</g></g>')
        # y extremes: header at its left edge, grid rows on the leftmost cell
        if p_i == 0 or (p_i - 1) % n_cols == 0:
            fmt = "{:.1f}" if max(abs(lo), abs(hi)) < 10 else "{:.0f}"
            out.append(f'<text x="{x0 - 4:.1f}" y="{ytop + 9}" '
                       f'text-anchor="end" font-size="8.5" '
                       f'fill="{NEUTRAL}">{fmt.format(hi)}</text>')
            out.append(f'<text x="{x0 - 4:.1f}" y="{ybot - 1}" '
                       f'text-anchor="end" font-size="8.5" '
                       f'fill="{NEUTRAL}">{fmt.format(lo)}</text>')
        # "now" line
        out.append(f'<line x1="{xa:.1f}" y1="{ytop}" x2="{xa:.1f}" '
                   f'y2="{ybot}" stroke="{JOINT_COLORS[1]}" '
                   f'stroke-width="1.3"/>')

    # header ylabel, row labels, column titles, time readout
    for li, line in enumerate(hdr_label.split("\n")):
        out.append(f'<text x="{STRIP_PAD_L - 34}" '
                   f'y="{y_head + STRIP_HEIGHT / 2 + 11 * li - 2}" '
                   f'text-anchor="end" font-size="10.5" fill="{NEUTRAL}" '
                   f'font-weight="600">{line}</text>')
    for c_i, title in enumerate(col_titles):
        cx = GRID_GUTTER + c_i * (cell_w + GRID_GAP) + 0.5 * cell_w
        out.append(f'<text x="{cx:.1f}" y="{y_grid - 4}" '
                   f'text-anchor="middle" font-size="11" fill="{NEUTRAL}" '
                   f'font-weight="600">{title}</text>')
    for r_i, (row_label, _) in enumerate(grid_rows):
        ymid = (y_grid + r_i * (GRID_CELL_H + GRID_GAP) + 0.5 * GRID_CELL_H)
        lines = row_label.split("\n")
        for li, line in enumerate(lines):
            out.append(f'<text x="8" y="{ymid + 11 * li - 5.5 * (len(lines) - 1) + 4:.1f}" '
                       f'font-size="10" fill="{NEUTRAL}" '
                       f'font-weight="600">{line}</text>')
    xc_hdr = STRIP_PAD_L + 0.5 * (x1 - STRIP_PAD_L)
    out.append(f'<text id="{uid}-time" x="{xc_hdr:.1f}" y="{y_ticks + 15}" '
               f'text-anchor="middle" font-size="10" font-weight="600" '
               f'fill="{JOINT_COLORS[1]}"></text>')
    out.append("</svg>")

    viewer_iframe_js = f'"{viewer_iframe}"' if viewer_iframe else "null"
    slider, slider_js = "", ""
    if variants:
        slider = (f'<div class="variant-pick"><input type="range" '
                  f'id="{uid}-slider" min="0" max="{len(variants) - 1}" '
                  f'step="1" value="{variant_ref}" style="width:100%"><br>'
                  f'<span id="{uid}-vlabel">{variants[variant_ref]}</span></div>')
        slider_js = f"""
  var LABELS = {json.dumps(list(variants))}, ROOT = "{variant_root}";
  var TSCALE = {json.dumps([float(x) for x in variant_time_scale]) if variant_time_scale else "null"};
  var slider = document.getElementById("{uid}-slider");
  var vlabel = document.getElementById("{uid}-vlabel");
  var shown = {variant_ref}, synced = -1;
  function viewer() {{
    try {{
      var f = document.getElementById(IFRAME);
      return f && f.contentWindow && f.contentWindow.viewer;
    }} catch (e) {{ return null; }}
  }}
  function sync_viewer() {{
    var v = IFRAME ? viewer() : null;
    // wait until the page has replayed its scene (the animation is set
    // last), or the replay would undo what we set here
    if (!v || !v.handle_command || !(v.animator && v.animator.duration > 0))
      return;
    for (var k = 0; k < LABELS.length; k++)
      v.handle_command({{type: "set_property", path: ROOT + k,
                         property: "visible", value: k === shown}});
    if (TSCALE && v.animator && v.animator.actions)
      v.animator.actions.forEach(function(a) {{ a.timeScale = TSCALE[shown]; }});
    synced = shown;
  }}
  function pick(k) {{
    var all = document.querySelectorAll("[class^='{uid}-v']");
    for (var i = 0; i < all.length; i++)
      all[i].style.display =
        (all[i].getAttribute("class") === "{uid}-v" + k) ? "" : "none";
    shown = k;
    vlabel.innerHTML = LABELS[k];
    sync_viewer();
  }}
  slider.addEventListener("input", function() {{ pick(+slider.value); }});"""
    script = f"""<script>
(function() {{
  var pans  = document.querySelectorAll(".{uid}-pan");
  var label = document.getElementById("{uid}-time");
  var T0 = {t[0]:.6f}, PXS = {pxs:.6f}, PERIOD = {period_s:.6f};
  var IFRAME = {viewer_iframe_js};
  var start = null;
  function viewer_time() {{
    if (!IFRAME) return null;
    try {{
      var f = document.getElementById(IFRAME);
      var a = f && f.contentWindow && f.contentWindow.viewer
              && f.contentWindow.viewer.animator;
      if (a && a.duration > 0) {{
        var tt = (a.actions && a.actions.length) ? a.actions[0].time : a.time;
        if (isFinite(tt))
          return ((tt % a.duration) + a.duration) % a.duration;
      }}
    }} catch (e) {{}}
    return null;
  }}
  function tick(now) {{
    if (start === null) start = now;
    var tau = viewer_time();
    if (tau === null)
      tau = ((now - start) / 1000.0) % PERIOD;
    var tr = "translate(" + (-tau * PXS) + ",0)";
    for (var i = 0; i < pans.length; i++)
      pans[i].setAttribute("transform", tr);
    label.textContent = {'(100 * tau / PERIOD).toFixed(0) + "% stride"' if phase_ticks else '(T0 + tau).toFixed(2) + " s"'};
    {"if (synced !== shown) sync_viewer();" if variants else ""}
    window.requestAnimationFrame(tick);
  }}{slider_js}
  window.requestAnimationFrame(tick);
}})();
</script>"""
    legend = " &nbsp; ".join(f'<span style="color:{c}">&#9644;</span> {lab}'
                             for lab, c, _, _ in hdr_series)
    return (f'<div class="strips">{slider}{"".join(out)}'
            f'<div class="strip-legend">{legend}</div></div>{script}')


# ---------------------------------------------------------------------------
# Muscle forces (OpenSim static + dynamic optimization), synced to the viewer
# ---------------------------------------------------------------------------

#: where the OpenSim agent writes P1 trial-13 muscle results (repo-relative)
MUSCLE_NPZ = "data/opensim/p1_trial13_muscles.npz"
#: 20 categorical colours, dark enough for thin lines on the light card.
MUSCLE_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0",
                  "#d6417a", "#17a2b0", "#7a6a2e", "#5d7f1e", "#c24f1c",
                  "#3b4cc0", "#c86fb6", "#0f6f78", "#8c564b", "#a3a314",
                  "#6f6f6f", "#1f3d7a", "#a4161a", "#0b6e4f", "#6a4c93")
#: tendon-compliance MocoInverse stages (npz key prefix, selector label)
COMPLIANCE_STAGES = (("c1", "C1 triceps surae tendons"),
                     ("c2", "C2 long tendons"),
                     ("c3", "C3 all tendons + passive"))
#: canvas/SVG dash pattern per solution in the muscle grid
SOLUTION_DASH = {"so": [], "dyn": [], "c1": [7, 3], "c2": [2, 2.5],
                 "c3": [9, 3, 2, 3]}


def load_muscles(path) -> dict:
    """The muscle npz as a plain dict of numpy arrays (+ its JSON sidecar).

    Contract (written by the OpenSim side): `t` [T] s on the trial clock,
    `muscle_names` [M], `so_force`/`dyn_force` [T,M] N (dyn is NaN outside
    its `window` [2]), `so_activation`/`dyn_activation` [T,M], `joint_names`
    ["ankle","knee","hip"], `moment_arm` [T,M,3] m, `group` [M] in {0,1,2},
    `crosses` [M,3] bool, `so_moment`/`dyn_moment` [T,3],
    `id_moment_opensim`/`id_moment_boneid` [T,3] N m.  Optional (newer
    files, `scripts/opensim/fiber_kinematics.py`): `fiber_length_norm`,
    `fiber_velocity_norm` [T,M] (l_M/l_opt and v_M/(v_max l_opt), + =
    lengthening), `mtu_length`, `tendon_length` [T,M] m, `pennation` [T,M]
    rad, and per muscle `f_iso`, `l_opt`, `l_slack`, `v_max`, `alpha_opt`.
    Optional compliant-tendon MocoInverse stages (`scripts/opensim/
    compliance.py`), prefix `c1_`/`c2_`/`c3_`: `force`, `activation`,
    `fiber_length_norm`, `fiber_velocity_norm`, `tendon_length`,
    `tendon_strain` [T,M], `moment` [T,3], plus `compliant_mask_c*` [M].
    The sidecar (same stem, `.json`) lands in `["meta"]` when it exists.
    """
    import json
    import os
    with np.load(path, allow_pickle=False) as z:
        mus = {k: z[k] for k in z.files}
    side = os.path.splitext(str(path))[0] + ".json"
    mus["meta"] = json.load(open(side)) if os.path.exists(side) else {}
    return mus


def muscles_on_clock(mus: dict, t: np.ndarray) -> dict:
    """Copy of `mus` with every time series linearly resampled onto `t`.

    Samples outside the muscle file's own time range become NaN (drawn as
    gaps), and NaNs inside (dynamic optimization outside its window) stay
    gaps.  On the shared i/120 clock this is an exact index pick.
    """
    t = np.asarray(t, float)
    tm = np.asarray(mus["t"], float)
    out = dict(mus)
    for k, v in mus.items():
        if not (isinstance(v, np.ndarray) and v.dtype.kind == "f"
                and v.ndim >= 2 and v.shape[0] == len(tm)):
            continue
        flat = v.reshape(len(tm), -1)
        res = np.stack([np.interp(t, tm, flat[:, c], left=np.nan,
                                  right=np.nan)
                        for c in range(flat.shape[1])], axis=1)
        out[k] = res.reshape((len(t),) + v.shape[1:])
    out["t"] = t
    return out


def muscle_columns(group: np.ndarray, crosses: np.ndarray) -> list[list[int]]:
    """Muscle indices per joint column: grouped there OR crossing it, so a
    biarticular muscle appears in every column it spans."""
    group = np.asarray(group).astype(int)
    crosses = np.asarray(crosses, bool)
    return [[m for m in range(len(group)) if group[m] == j or crosses[m, j]]
            for j in range(crosses.shape[1])]


def muscle_colors(columns: list[list[int]], n: int) -> list[str]:
    """One colour per muscle, the same in every column it appears in, chosen
    greedily so no two muscles in the same column share one (until the
    palette runs out)."""
    pal = MUSCLE_PALETTE
    colors: list[str | None] = [None] * n
    for m in range(n):
        taken = {colors[o] for col in columns if m in col for o in col}
        free = [c for c in pal if c not in taken]
        colors[m] = free[0] if free else pal[m % len(pal)]
    return colors


def muscle_grid_html(t, mus: dict, period_s: float, uid: str = "mgrid",
                     window_s: float = 1.6,
                     viewer_iframe: str | None = None,
                     col_titles=("ankle", "knee", "hip")) -> str:
    """Muscle forces as a scrolling-scope GRID on the animation clock.

    `mus` must already be on the clock `t` (`muscles_on_clock`) — the same
    samples the meshcat clip keys, so ``index = tau * rate`` exactly.
    Columns are joints; muscles grouped at or crossing a joint appear in its
    column.  Rows: (a) static-optimization force, (b) dynamic-optimization
    force (both N, one shared scale over the checked muscles), (c) the
    summed moment of the CHECKED muscles, sum_m F_m * r_mj, for SO and
    dynamic opt, over the inverse-dynamics torque (OpenSim's and ours, as
    the npz gives them).  The checkboxes under each column toggle that
    muscle in that column only; row (c) and the force scale recompute on
    every change.  When `mus` has `fiber_length_norm` / `fiber_velocity_norm`
    two more rows, (d) normalized fibre length and (e) normalized fibre
    velocity, sit below (c), hidden until their toggle is ticked; one trace
    per checked muscle (rigid tendon + prescribed kinematics make them the
    same for SO and dynamic optimization).  If the npz carries compliant
    stages (`c1_force` ...), a "solutions drawn" row of checkboxes picks
    SO / rigid dynamic / C1 / C2 / C3 for rows (b)-(e) (one line style per
    solution, `SOLUTION_DASH`), and an optional row (f) shows tendon strain.

    Drawing is canvas in JS (the moments must be re-summed live, so the
    static-SVG trick of `strip_charts_grid` doesn't fit); the clock is the
    same — `viewer.animator.actions[0].time` of the meshcat iframe, with a
    wall-clock fallback at `period_s`.  Data ride along as base64 float32.
    """
    import json
    t = np.asarray(t, float)
    n_t = len(t)
    names = [str(s) for s in mus["muscle_names"]]
    n_m = len(names)
    cols = muscle_columns(mus["group"], mus["crosses"])
    colors = muscle_colors(cols, n_m)
    rate = (n_t - 1) / max(float(t[-1] - t[0]), 1e-12)
    win = [float(v) for v in np.asarray(mus.get("window", [np.nan, np.nan]))]
    data = {
        "N": n_t, "M": n_m, "RATE": rate, "T0": float(t[0]),
        "PERIOD": float(period_s), "W": float(window_s),
        "names": names, "colors": colors, "cols": cols,
        "so": _b64_f32(mus["so_force"]), "dyn": _b64_f32(mus["dyn_force"]),
        "ma": _b64_f32(mus["moment_arm"][:, :, :3]),
        "idO": _b64_f32(mus["id_moment_opensim"][:, :3]),
        "idB": _b64_f32(mus["id_moment_boneid"][:, :3]),
    }
    fib = "fiber_length_norm" in mus and "fiber_velocity_norm" in mus
    # dynamic solutions: the rigid MocoInverse plus whichever compliance
    # stages (scripts/opensim/compliance.py) the npz carries
    sols = [("dyn", "rigid dynamic")] + [
        (k, lab) for k, lab in COMPLIANCE_STAGES if f"{k}_force" in mus]
    data["sols"] = [{"key": k, "label": lab, "dash": SOLUTION_DASH[k]}
                    for k, lab in sols]
    data["F"] = {k: data["dyn"] if k == "dyn" else _b64_f32(mus[f"{k}_force"])
                 for k, _ in sols}
    data["solon"] = {"so": True, "dyn": True,
                     **{k: k == sols[-1][0] for k, _ in sols[1:]}}
    if fib:
        data["lf"] = {k: _b64_f32(mus["fiber_length_norm" if k == "dyn" else
                                      f"{k}_fiber_length_norm"])
                      for k, _ in sols}
        data["vf"] = {k: _b64_f32(mus["fiber_velocity_norm" if k == "dyn" else
                                      f"{k}_fiber_velocity_norm"])
                      for k, _ in sols}
    ten = [k for k, _ in sols[1:] if f"{k}_tendon_strain" in mus]
    if ten:
        data["ts"] = {k: _b64_f32(100.0 * mus[f"{k}_tendon_strain"])
                      for k in ten}
    ink = {"so": JOINT_COLORS[0], "dyn": JOINT_COLORS[1],
           "idO": "#1c1b18", "idB": NEUTRAL}
    gut = 100.0 * GRID_GUTTER / STRIP_WIDTH
    gap = 100.0 * GRID_GAP / STRIP_WIDTH
    padr = 100.0 * STRIP_PAD_R / STRIP_WIDTH
    rows = [("a", "SO force (N)"), ("b", "dyn-opt\nforce (N)"),
            ("c", "summed\nmoment (N m)")]
    if fib:
        rows += [("d", "fibre length\nl<sub>M</sub>/l<sub>opt</sub>"),
                 ("e", "fibre velocity\nv<sub>M</sub>/(v<sub>max</sub>"
                       "l<sub>opt</sub>)")]
    if ten:
        rows += [("f", "tendon strain\n(%)")]
    cells = ['<div></div>'] + [f'<div class="mg-title">{c}</div>'
                               for c in col_titles]
    for key, lab in rows:
        opt = (f' data-opt="{key}" style="display:none"'
               if key in "def" else "")
        cells.append(f'<div class="mg-row"{opt}>'
                     f'{lab.replace(chr(10), "<br>")}</div>')
        for j in range(len(col_titles)):
            cells.append(f'<canvas class="mg-cv" data-row="{key}" '
                         f'data-col="{j}"{opt}></canvas>')
    cells.append('<div></div>')
    for j in range(len(col_titles)):
        boxes = "".join(
            f'<label class="mg-mu" data-m="{m}"><input type="checkbox" '
            f'data-col="{j}" data-m="{m}" checked> '
            f'<span style="color:{colors[m]}">&#9644;</span> {names[m]}'
            f'{"" if int(mus["group"][m]) == j else " *"}</label>'
            for m in cols[j])
        cells.append(f'<div class="mg-checks"><div class="mg-ctl">'
                     f'<button data-col="{j}" data-all="1">all</button> '
                     f'<button data-col="{j}" data-all="0">none</button>'
                     f'</div>{boxes}</div>')
    win_note = ("" if not all(np.isfinite(win)) else
                f"; dynamic optimization covers {win[0]:.2f}&ndash;"
                f"{win[1]:.2f} s only (gaps elsewhere)")
    toggles = ("" if not fib else
               f'<div class="mg-opts">show below the moment row: '
               f'<label><input type="checkbox" class="mg-opt" data-row="d">'
               f' (d) fibre length l<sub>M</sub>/l<sub>opt</sub></label> '
               f'<label><input type="checkbox" class="mg-opt" data-row="e">'
               f' (e) fibre velocity v<sub>M</sub>/(v<sub>max</sub> '
               f'l<sub>opt</sub>), + = lengthening</label>'
               + ('' if not ten else
                  '<label><input type="checkbox" class="mg-opt" data-row="f">'
                  ' (f) tendon strain (%)</label>')
               + '</div>')
    if len(sols) > 1:
        sol_boxes = "".join(
            f'<label><input type="checkbox" class="mg-sol" data-sol="{k}"'
            f'{" checked" if data["solon"][k] else ""}> '
            f'<svg width="26" height="8" style="display:inline-block;'
            f'vertical-align:middle;width:26px;min-width:0;height:8px">'
            f'<line x1="1" y1="4" x2="25" y2="4" '
            f'stroke="{ink["so"] if k == "so" else ink["dyn"]}" '
            f'stroke-width="2" stroke-dasharray="'
            f'{" ".join(str(v) for v in SOLUTION_DASH.get(k, []))}"/></svg> '
            f'{lab}</label>'
            for k, lab in [("so", "SO")] + sols)
        toggles = (f'<div class="mg-opts mg-sols">solutions drawn: '
                   f'{sol_boxes}</div>' + toggles)
    fib_note = ("" if not fib else
                " &nbsp;&middot;&nbsp; (d)/(e): one trace per checked muscle"
                + (" and checked dynamic solution (SO shares the rigid"
                   " fibres; line style = solution)" if len(sols) > 1 else
                   ", the same for SO and dyn-opt (rigid tendon, prescribed"
                   " kinematics)")
                + "; dotted line = l<sub>opt</sub> / zero velocity")
    legend = (f'<span style="color:{ink["so"]}">&#9644;</span> SO muscles '
              f'(checked) &nbsp; <span style="color:{ink["dyn"]}">&#9644;'
              f'</span> dyn-opt muscles (checked) &nbsp; '
              f'<span style="color:{ink["idO"]}">&#9644;</span> ID torque '
              f'(OpenSim) &nbsp; <span style="color:{ink["idB"]}">&#9476;'
              f'</span> ID torque (boneid) &nbsp;&middot;&nbsp; '
              f'* = muscle grouped at another joint (biarticular)'
              f'{win_note}{fib_note} &nbsp;&middot;&nbsp; '
              f'<span id="{uid}-time"></span>')
    style = f"""<style>
#{uid} .mg-grid {{ display: grid; grid-template-columns: {gut:.3f}% repeat({len(col_titles)}, 1fr);
  column-gap: {gap:.3f}%; row-gap: 8px; padding-right: {padr:.3f}%; min-width: 520px; }}
#{uid} .mg-title {{ text-align: center; font-weight: 600; font-size: 0.78rem; color: {NEUTRAL}; }}
#{uid} .mg-row {{ font-size: 0.68rem; font-weight: 600; color: {NEUTRAL}; align-self: center; line-height: 1.25; }}
#{uid} canvas.mg-cv {{ width: 100%; height: 92px; display: block; }}
#{uid} .mg-checks {{ font-size: 0.72rem; line-height: 1.35; color: {NEUTRAL}; }}
#{uid} .mg-mu {{ display: block; cursor: pointer; white-space: nowrap; }}
#{uid} .mg-mu:hover {{ color: #1c1b18; }}
#{uid} .mg-opts {{ font-size: 0.74rem; color: {NEUTRAL}; margin: 2px 0 8px; }}
#{uid} .mg-opts label {{ margin-right: 14px; cursor: pointer; white-space: nowrap; }}
#{uid} .mg-ctl button {{ font: inherit; font-size: 0.7rem; padding: 0 6px; margin-bottom: 3px;
  border: 1px solid {FAINT}; background: {SURFACE}; border-radius: 3px; cursor: pointer; }}
</style>"""
    viewer_iframe_js = json.dumps(viewer_iframe)
    script = f"""<script>
(function() {{
  var D = {json.dumps(data)};
  var IFRAME = {viewer_iframe_js};
  var INK = {json.dumps(ink)};
  var root = document.getElementById("{uid}");
  var label = document.getElementById("{uid}-time");
  function f32(s) {{
    var b = atob(s), u = new Uint8Array(b.length);
    for (var i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
    return new Float32Array(u.buffer);
  }}
  var N = D.N, M = D.M, J = D.cols.length;
  var SO = f32(D.so), MA = f32(D.ma);
  var IDO = f32(D.idO), IDB = f32(D.idB);
  var SK = D.sols.map(function(s) {{ return s.key; }});   // dynamic solutions
  var DASH = {{}};
  D.sols.forEach(function(s) {{ DASH[s.key] = s.dash; }});
  function dec(o) {{
    if (!o) return null;
    var r = {{}};
    for (var k in o) r[k] = f32(o[k]);
    return r;
  }}
  var FS = dec(D.F), LF = dec(D.lf), VF = dec(D.vf), TS = dec(D.ts);
  var SOL = D.solon;                   // which solutions are drawn
  var llo = 0, lhi = 2, vlo = -1, vhi = 1, tlo = 0, thi = 1;
  var on = [];                         // on[j][m]: checked in column j
  for (var j = 0; j < J; j++) {{
    on.push({{}});
    D.cols[j].forEach(function(m) {{ on[j][m] = true; }});
  }}
  var hot = -1;                        // muscle under the mouse
  var sumSO = [], sumDY = {{}}, fmax = 1, mlo = -1, mhi = 1;
  function fibKeys() {{                // solutions whose fibre traces show
    var ks = SK.filter(function(k) {{ return SOL[k]; }});
    if (SOL.so && ks.indexOf("dyn") < 0) ks.unshift("dyn");   // SO = rigid fibres
    return ks;
  }}
  function recompute() {{
    fmax = 1e-9; mlo = 0; mhi = 0;
    sumSO = []; sumDY = {{}};
    SK.forEach(function(k) {{ sumDY[k] = []; }});
    for (var j = 0; j < J; j++) {{
      var a = new Float32Array(N), b = {{}};
      SK.forEach(function(k) {{ b[k] = new Float32Array(N); }});
      var ms = D.cols[j].filter(function(m) {{ return on[j][m]; }});
      for (var i = 0; i < N; i++) {{
        var sa = 0;
        for (var k = 0; k < ms.length; k++) {{
          var m = ms[k], r = MA[(i * M + m) * 3 + j];
          sa += SO[i * M + m] * r;
          if (SOL.so && SO[i * M + m] > fmax) fmax = SO[i * M + m];
        }}
        a[i] = sa;
        var vs = [IDO[i * 3 + j], IDB[i * 3 + j]];
        if (SOL.so) vs.push(a[i]);
        SK.forEach(function(s) {{
          var F = FS[s], sb = 0;
          for (var k = 0; k < ms.length; k++) {{
            var m = ms[k], f = F[i * M + m];
            sb += f * MA[(i * M + m) * 3 + j];
            if (SOL[s] && f > fmax) fmax = f;
          }}
          b[s][i] = sb;
          if (SOL[s]) vs.push(sb);
        }});
        for (var q = 0; q < vs.length; q++) if (isFinite(vs[q])) {{
          if (vs[q] < mlo) mlo = vs[q];
          if (vs[q] > mhi) mhi = vs[q];
        }}
      }}
      sumSO.push(a);
      SK.forEach(function(s) {{ sumDY[s].push(b[s]); }});
    }}
    var p = 0.06 * (mhi - mlo || 1); mlo -= p; mhi += p;
    fmax *= 1.06;
    if (LF) {{                         // fibre rows: range over checked muscles
      llo = 1; lhi = 1; vlo = 0; vhi = 0; tlo = 0; thi = 0.5;
      var ks = fibKeys();
      for (var j2 = 0; j2 < J; j2++) D.cols[j2].forEach(function(m) {{
        if (!on[j2][m]) return;
        ks.forEach(function(s) {{
          var L = LF[s], V = VF[s], T = TS && TS[s];
          for (var i = 0; i < N; i++) {{
            var l = L[i * M + m], v = V[i * M + m];
            if (isFinite(l)) {{ if (l < llo) llo = l; if (l > lhi) lhi = l; }}
            if (isFinite(v)) {{ if (v < vlo) vlo = v; if (v > vhi) vhi = v; }}
            if (T) {{
              var e = T[i * M + m];
              if (isFinite(e)) {{ if (e < tlo) tlo = e; if (e > thi) thi = e; }}
            }}
          }}
        }});
      }});
      var pl = 0.06 * (lhi - llo || 1), pv = 0.06 * (vhi - vlo || 1);
      var pt = 0.06 * (thi - tlo || 1);
      llo -= pl; lhi += pl; vlo -= pv; vhi += pv; tlo -= pt; thi += pt;
    }}
  }}
  var canv = Array.prototype.slice.call(root.querySelectorAll("canvas.mg-cv"));
  function trace(ctx, w, h, tau, lo, hi, get, color, lw, dash, alpha) {{
    var i0 = Math.floor((tau - D.W / 2) * D.RATE) - 1;
    var i1 = Math.ceil((tau + D.W / 2) * D.RATE) + 1;
    var P = N - 1;                     // sample N-1 is sample 0, one loop on
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.globalAlpha = alpha; ctx.setLineDash(dash || []);
    var pen = false;
    for (var i = i0; i <= i1; i++) {{
      var v = get(((i % P) + P) % P);
      if (!isFinite(v)) {{ pen = false; continue; }}
      var x = w / 2 + (i / D.RATE - tau) * w / D.W;
      var y = h - (h * (v - lo) / (hi - lo));
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    }}
    ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
  }}
  function frame(cv, tau) {{
    var dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {{
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }}
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "{SURFACE}"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "{FAINT}"; ctx.lineWidth = 0.8;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    var row = cv.dataset.row, j = +cv.dataset.col;
    var lo = row === "c" ? mlo : -0.03 * fmax, hi = row === "c" ? mhi : fmax;
    if (row === "d") {{ lo = llo; hi = lhi; }}
    if (row === "e") {{ lo = vlo; hi = vhi; }}
    if (row === "f") {{ lo = tlo; hi = thi; }}
    var ref = row === "d" ? 1 : 0;     // l_opt / zero velocity / zero
    if (lo < ref && hi > ref) {{
      var zy = h - h * (ref - lo) / (hi - lo);
      ctx.setLineDash(row === "d" || row === "e" ? [2, 3] : []);
      ctx.strokeStyle = row === "d" || row === "e" ? "{NEUTRAL}" : "{FAINT}";
      ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
      ctx.setLineDash([]);
    }}
    if (row === "c") {{
      trace(ctx, w, h, tau, lo, hi, function(i) {{ return IDB[i * 3 + j]; }},
            INK.idB, 1.3, [4, 3], 1);
      trace(ctx, w, h, tau, lo, hi, function(i) {{ return IDO[i * 3 + j]; }},
            INK.idO, 1.6, null, 1);
      if (SOL.so)
        trace(ctx, w, h, tau, lo, hi, function(i) {{ return sumSO[j][i]; }},
              INK.so, 1.4, null, 1);
      SK.forEach(function(s) {{
        if (!SOL[s]) return;
        trace(ctx, w, h, tau, lo, hi, function(i) {{ return sumDY[s][j][i]; }},
              INK.dyn, 1.4, DASH[s], 1);
      }});
    }} else {{
      var srcs;                        // [[array, dash], ...] for this row
      if (row === "a") srcs = SOL.so ? [[SO, []]] : [];
      else {{
        var tab = row === "b" ? FS : row === "d" ? LF : row === "e" ? VF : TS;
        var ks = row === "b" ? SK.filter(function(k) {{ return SOL[k]; }})
                             : fibKeys();
        srcs = ks.filter(function(k) {{ return tab[k]; }})
                 .map(function(k) {{ return [tab[k], DASH[k]]; }});
      }}
      D.cols[j].forEach(function(m) {{
        if (!on[j][m]) return;
        var a = (hot < 0 || hot === m || !(hot in on[j])) ? 1 : 0.18;
        srcs.forEach(function(sd) {{
          var src = sd[0];
          trace(ctx, w, h, tau, lo, hi, function(i) {{ return src[i * M + m]; }},
                D.colors[m], hot === m ? 2.4 : 1.2, sd[1], a);
        }});
      }});
    }}
    if (j === 0) {{                    // shared row scale, on the first cell
      ctx.fillStyle = "{NEUTRAL}"; ctx.font = "9px sans-serif";
      var fmt = function(v) {{ return Math.abs(v) < 10 ? v.toFixed(row === "d" || row === "e" ? 2 : 1) : v.toFixed(0); }};
      ctx.fillText(fmt(hi), 3, 10); ctx.fillText(fmt(lo), 3, h - 3);
    }}
    ctx.strokeStyle = "{JOINT_COLORS[1]}"; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
  }}
  root.addEventListener("change", function(e) {{
    var c = e.target;
    if (c.classList && c.classList.contains("mg-opt")) {{
      root.querySelectorAll('[data-opt="' + c.dataset.row + '"]')
        .forEach(function(el) {{ el.style.display = c.checked ? "" : "none"; }});
      return;
    }}
    if (c.classList && c.classList.contains("mg-sol")) {{
      SOL[c.dataset.sol] = c.checked; recompute();
      return;
    }}
    if (!c.dataset || c.dataset.m === undefined) return;
    on[+c.dataset.col][+c.dataset.m] = c.checked; recompute();
  }});
  root.addEventListener("click", function(e) {{
    var b = e.target;
    if (b.tagName !== "BUTTON") return;
    var j = +b.dataset.col, v = b.dataset.all === "1";
    root.querySelectorAll('input[data-col="' + j + '"]').forEach(function(c) {{
      c.checked = v; on[j][+c.dataset.m] = v;
    }});
    recompute();
  }});
  root.querySelectorAll(".mg-mu").forEach(function(l) {{
    l.addEventListener("mouseenter", function() {{ hot = +l.dataset.m; }});
    l.addEventListener("mouseleave", function() {{ hot = -1; }});
  }});
  function viewer_time() {{
    if (!IFRAME) return null;
    try {{
      var f = document.getElementById(IFRAME);
      var a = f && f.contentWindow && f.contentWindow.viewer
              && f.contentWindow.viewer.animator;
      if (a && a.duration > 0) {{
        var tt = (a.actions && a.actions.length) ? a.actions[0].time : a.time;
        if (isFinite(tt))
          return ((tt % a.duration) + a.duration) % a.duration;
      }}
    }} catch (e) {{}}
    return null;
  }}
  recompute();
  var start = null;
  function tick(now) {{
    if (start === null) start = now;
    var tau = viewer_time();
    if (tau === null) tau = ((now - start) / 1000.0) % D.PERIOD;
    for (var k = 0; k < canv.length; k++)
      if (canv[k].style.display !== "none") frame(canv[k], tau);
    label.textContent = "now " + (D.T0 + tau).toFixed(2) + " s";
    window.requestAnimationFrame(tick);
  }}
  window.requestAnimationFrame(tick);
}})();
</script>"""
    return (f'{style}<div class="strips" id="{uid}">{toggles}'
            f'<div class="mg-grid">{"".join(cells)}</div>'
            f'<div class="strip-legend">{legend}</div></div>{script}')


def p1_strip_grid(trial, chain, whole, params, viewer_iframe: str,
                  uid: str = "p1grid") -> tuple[str, slice, float]:
    """(html, clip_slice, period_s): the 3x3 sagittal scope for P1's right
    leg on the exact window and period `p1_viewer_html`'s clip keys.

    Same grid as the Asics report (`strip_charts_grid`), with the GRF of
    the right belt on top (left belt's vertical dashed).  There's no static
    trial, so angles are not zeroed (`c3d_sagittal_series(cal=None, ...)`).
    """
    skel_r, kin_r, ground_r, _, _, ground_l = chain
    n_key, fps, period = viewer_period(kin_r, VIEWER_DECIMATE)
    ssl = slice(0, (n_key - 1) * VIEWER_DECIMATE + 1)
    sag = c3d_sagittal_series(None, skel_r, kin_r, whole.right, params)
    grid_rows = [
        (label, [(JOINT_COLORS[j], sag[key][ssl, j]) for j in range(3)])
        for label, key in (("angle (deg)", "angle"),
                           ("velocity\n(deg/s)", "vel"),
                           ("torque\n(N m)", "torque"))]
    html = strip_charts_grid(
        kin_r.t[ssl],
        ("GRF (N)",
         [("R vertical", JOINT_COLORS[0], None, ground_r.force[ssl, 2]),
          ("R anterior-posterior", JOINT_COLORS[2], None,
           ground_r.force[ssl, 1]),
          ("R medio-lateral", JOINT_COLORS[3], "4 3", ground_r.force[ssl, 0]),
          ("L vertical", FAINT, "4 3", ground_l.force[ssl, 2])]),
        grid_rows, ("ankle", "knee", "hip"),
        period_s=period, fps=fps, n_frames=n_key, uid=uid,
        viewer_iframe=viewer_iframe)
    return html, ssl, period


def muscle_viewer_html(trial, muscles_npz_path, params=None,
                       viewer_html: str | None = None, run=None) -> str:
    """Full page: P1 animation, the 3x3 sagittal scope, and the muscle grid,
    all on the meshcat clip's clock.

    `run` is `whole_body_run(trial, params)`'s tuple (recomputed if None);
    `viewer_html` is `p1_viewer_html`'s output (built if None — that needs
    meshcat).  The muscle npz is resampled onto the clip's samples;
    `muscles_npz_path` may also be an already-loaded `load_muscles` dict.
    """
    params = params or p1_params()
    chain, skel_u, kin_u, whole, _ = run or whole_body_run(trial, params)
    if viewer_html is None:
        viewer_html = p1_viewer_html(trial, chain, skel_u, kin_u, params)
    mus = (muscles_npz_path if isinstance(muscles_npz_path, dict)
           else load_muscles(muscles_npz_path))
    sign_note = ("That row uses our sign convention, the same as the torque "
                 "row above." if mus.get("meta", {}).get("moment_convention") == "boneid"
                 else "That row is in OpenSim's sign convention, so its ankle "
                 "and hip can be flipped relative to the torque row above.")
    strips, ssl, period = p1_strip_grid(trial, chain, whole, params,
                                        "p1m-viewer")
    t = chain[1].t[ssl]
    grid = muscle_grid_html(t, muscles_on_clock(mus, t), period,
                            uid="p1mus", viewer_iframe="p1m-viewer")
    model = mus["meta"].get("model", "the OpenSim model")
    fib_text = ("" if "fiber_length_norm" not in mus else """
<p>The two tick boxes above the grid add rows (d) and (e): each muscle's
fibre length over its optimal length (1 = the top of the force-length
curve, the dotted line) and its fibre velocity over its maximum (v<sub>max</sub>
= 10 optimal lengths per second; positive = lengthening, &minus;1 would be
flat-out shortening with no force left).  They follow the same muscle
checkboxes and the same clock.  There's only one trace per muscle because,
with a rigid tendon and the kinematics prescribed from IK, fibre length and
velocity are pure geometry, so static and dynamic optimization see the same
ones.</p>
""")
    esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
    title = ("Subject P1 walking, muscle forces: "
             "which muscles carry each joint torque")
    body = f"""
<p class="eyebrow">boneid &middot; P1 trial 13 &middot; OpenSim muscles</p>
<h1>{title}</h1>
<p class="subtitle">{len(trial.t) / trial.rate:.1f} s of treadmill walking
at {trial.rate:.0f} Hz, right leg, muscle forces from {model}.</p>

<p>Everything on this page runs off one clock: the animation's.  Hit
<b>Open Controls &rarr; Animations</b> in the viewer to pause or scrub, and
both grids underneath come along (the orange centre line is "now").</p>
<iframe id="p1m-viewer" class="viewer" srcdoc="{esc}"></iframe>

<h2>Joint angles, velocities and torques: the inverse-dynamics side</h2>
<p>Ground reaction on top, then sagittal angle, angular velocity and torque
for the right ankle, knee and hip (each row on one shared scale).  There's
no static trial for P1, so angles aren't zeroed at standing: knee and hip
read about 0 when the segments line up, and the ankle is the tilt of the
heel-to-5th-met line off the shank axis (so it sits near 90&deg;).  Torque
here is our convention, about the proximal segment's left-pointing axis.</p>
{strips}

<h2>Muscle forces: how well the muscles add up to the joint torque</h2>
{grid}
<p>Columns are joints, and a muscle shows up in every column it crosses
(biarticular ones are starred where they aren't "home").  Row (a) is static
optimization, row (b) dynamic optimization, both in newtons on one scale.
Row (c) is the part I care about: the moment the <em>checked</em> muscles
make about that joint (force &times; moment arm, summed), for both solutions,
over the inverse-dynamics torque.  {sign_note}  Untick
muscles to see who actually carries the torque (hover a name to pick its
trace out).  If the coloured lines leave the black one, the rest is going
into reserve actuators or residuals, I think.</p>
{fib_text}"""
    return html_page(title, "muscles", body)


def knee_variant_series(chain, params, knee_shifts_m) -> dict:
    """Right-leg ankle/knee/hip series for each lateral knee shift d
    (`simulate.varus_shift` on the whole-body right chain, measured ground
    force, `core.inverse_dynamics`), each ``[D, T, 3]``:

    * ``sagittal`` [N m]: joint torque about the proximal segment's
      left-pointing axis (the same as `c3d_sagittal_series`' torque).
    * ``frontal`` [N m]: minus the torque about the proximal anterior axis,
      so + = EXTERNAL adduction (varus) moment on the right leg — the KAM
      at the knee (matches ``scripts/opensim/knee_kam_boneid.py``).
    * ``frontal_angle`` [deg]: the distal segment's rotation about the
      proximal anterior axis (from its medio-lateral axis), + = the distal
      segment adducted / varus for the right leg.  Not zeroed (no static
      trial), so read the CHANGE with d, not the level.
    """
    from .core import chain_kinematics, inverse_dynamics
    from .simulate import varus_shift
    skel_r, kin_r, ground_r = chain[:3]
    out = {k: [] for k in ("sagittal", "frontal", "frontal_angle")}
    for d in knee_shifts_m:
        kin_d = varus_shift(kin_r, float(d)) if d else kin_r
        res = inverse_dynamics(skel_r, kin_d, ground_r, params)
        r = chain_kinematics(skel_r, kin_d, params.lowpass_hz,
                             params.filter_order)["r"]
        out["sagittal"].append(res.joint_torque_local[:, :3, 1])
        out["frontal"].append(-res.joint_torque_local[:, :3, 0])
        ang = np.zeros((r.shape[0], 3))
        for j in range(3):
            r_rel = np.einsum("tji,tjk->tik", r[:, j + 1], r[:, j])
            ang[:, j] = np.degrees(np.arctan2(r_rel[:, 2, 1], r_rel[:, 1, 1]))
        out["frontal_angle"].append(ang)
    return {k: np.array(v) for k, v in out.items()}


#: front-on (lab -y is anterior), a little to the right: the bow reads best
VARUS_CAMERA = (-0.35, -1.75, 0.05)


def varus_viewer_html(trial, knee_contact_npz, params=None,
                      viewer_html: str | None = None, run=None,
                      stride: int = 1) -> str:
    """Full page: the walker with its right knee pushed out d = 0..50 mm,
    a slider over d that swaps BOTH the leg in the meshcat viewer and every
    trace in the synced grid underneath.

    The knee-contact header comes from OpenSim's varus sweep (static
    optimization + the two-point plateau split, ``sweep_*`` keys of the
    knee-contact npz); the 3x3 grid (sagittal torque, frontal torque,
    frontal angle x ankle/knee/hip) is boneid on `knee_variant_series`.
    `viewer_html` must come from ``p1_viewer_html(..., knee_shifts_m=d)``
    with the same d list (built here if None, which needs meshcat);
    `stride` keeps every stride-th d of the sweep.
    """
    params = params or p1_params()
    chain, skel_u, kin_u, whole, _ = run or whole_body_run(trial, params)
    kc = (knee_contact_npz if isinstance(knee_contact_npz, dict)
          else dict(np.load(knee_contact_npz)))
    pick = np.arange(0, len(kc["sweep_d_m"]), stride)
    d_m = kc["sweep_d_m"][pick]
    theta = kc["sweep_theta_deg"][pick]
    if viewer_html is None:
        viewer_html = p1_viewer_html(trial, chain, skel_u, kin_u, params,
                                     knee_shifts_m=d_m,
                                     camera_offset=VARUS_CAMERA)
    n_key, fps, period = viewer_period(chain[1], VIEWER_DECIMATE)
    ssl = slice(0, (n_key - 1) * VIEWER_DECIMATE + 1)
    t = chain[1].t[ssl]
    bw = float(kc["body_mass"]) * 9.81

    def on_clock(key):                    # [D, T] on the clip's samples
        return np.array([np.interp(t, kc["t"], kc[key][i], left=np.nan,
                                   right=np.nan) for i in pick])

    def peak(y):                          # mean over stances of the max
        return float(np.mean([np.nanmax(y[a:b + 1])
                              for a, b in kc["stance_frames"]]))

    ser = knee_variant_series(chain, params, d_m)
    dang = np.mean(ser["frontal_angle"][-1] - ser["frontal_angle"][0], axis=0)
    labels = []
    for k, i in enumerate(pick):
        med, lat = (peak(kc["sweep_so_medial"][i]),
                    peak(kc["sweep_so_lateral"][i]))
        kam = peak(ser["frontal"][k][:, 1])
        labels.append(f"<b>d = {1000 * d_m[k]:.0f} mm</b> "
                      f"({theta[k]:.1f}&deg; varus): peak medial {med:.2f} BW, "
                      f"lateral {lat:.2f} BW, knee adduction moment "
                      f"{kam:.0f} N m")
    header = ("knee contact\n(BW)", [
        ("medial (static opt)", JOINT_COLORS[0], None,
         on_clock("sweep_so_medial")),
        ("lateral (static opt)", JOINT_COLORS[1], None,
         on_clock("sweep_so_lateral")),
        ("lateral ligament tension needed", JOINT_COLORS[3], "2 2",
         on_clock("sweep_so_lig_lateral")),
        ("total", NEUTRAL, "5 3", on_clock("sweep_so_total"))])
    grid_rows = [
        (row, [(JOINT_COLORS[j], ser[key][:, ssl, j]) for j in range(3)])
        for row, key in (("sagittal\ntorque (N m)", "sagittal"),
                         ("frontal torque\n(N m, + ext.\nadduction)", "frontal"),
                         ("frontal angle\n(deg, + varus)", "frontal_angle"))]
    strips = strip_charts_grid(t, header, grid_rows, ("ankle", "knee", "hip"),
                               period_s=period, fps=fps, n_frames=n_key,
                               uid="varus", viewer_iframe="varus-viewer",
                               variants=labels)
    esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
    title = ("Subject P1 walking with the right knee pushed out: "
             "how the knee load moves as the leg bows")
    body = f"""
<p class="eyebrow">boneid &middot; P1 trial 13 &middot; varus sweep</p>
<h1>{title}</h1>
<p class="subtitle">Same walk, same measured ground force, right knee moved
laterally by 0&ndash;{1000 * d_m[-1]:.0f} mm (0&ndash;{theta[-1]:.1f}&deg;
of varus).</p>

<p>Drag the slider (it sits just under the viewer).  The right shank and thigh in the viewer swing out to
the knee you picked (hip, ankle and foot don't move, and the shank and thigh
markers ride along), and every trace below switches to that leg.  The faint
dashed line in each panel is the real, unshifted knee, so you can see how far
things moved.  The clock is still the animation's: pause or scrub it under
<b>Open Controls &rarr; Animations</b>.</p>
<iframe id="varus-viewer" class="viewer" srcdoc="{esc}"></iframe>
{strips}
<p>The top panel is OpenSim (static optimization, then the two-point medial/
lateral plateau split, {bw / 9.81:.1f} kg), for each d.  The grid is our own
inverse dynamics on the shifted leg.  Frontal torque is the external
adduction moment, so at the knee it's the KAM.  The frontal angle isn't
zeroed (P1 has no static trial), so read how much it <em>changes</em> with d
(at d = {1000 * d_m[-1]:.0f} mm the knee gains {dang[1]:.1f}&deg; while
the hip and ankle give back about {-dang[2]:.1f}&deg; and
{-dang[0]:.1f}&deg;; the knee's share is a bit under the {theta[-1]:.1f}&deg;
set in OpenSim because the knee is bent for much of the cycle, so part of
the swing shows up about the other axes).  The total contact barely moves while medial and lateral trade
places, and the sagittal torques don't care at all.  That's the whole story
of the sweep, I think.</p>"""
    return html_page(title, "varus", body)


def cadence_section_html(conditions, ref: int, params=None, uid: str = "cad",
                         n_key: int = 40, n_plot: int = 200,
                         viewer_html: str | None = None) -> str:
    """Viewer + synced grid for a set of walking conditions at one speed
    that differ in step frequency, with a slider over them.

    `conditions[i]` is a dict: ``label`` (slider readout, HTML), ``trial``,
    ``run`` (`whole_body_run`'s tuple) and optional ``contact`` (dict of
    ``total``/``medial``/``lateral``/``lig_lateral`` [T] in BW on
    ``trial.t``).  Everything is one stride, phase-normalised: the walker
    is the stride closest to the condition's median, sampled at n_key + 1
    phases; the traces are means over every complete right stride.  The
    clip's period is condition `ref`'s mean stride time, and the slider sets
    the playback rate to ref/condition, so each walker steps at its real
    cadence while the grid (on the clip's clock = stride phase) stays put.
    """
    params = params or p1_params()
    strides = [right_strides(c["run"][0]) for c in conditions]
    t_stride = np.array([np.mean(np.diff(st, axis=1)) / c["trial"].rate
                         for st, c in zip(strides, conditions)])
    period = float(t_stride[ref])
    rate = n_key / period
    if viewer_html is None:
        bodies = []
        for st, c in zip(strides, conditions):
            lens = np.diff(st, axis=1)[:, 0]
            a, b = st[int(np.argmin(np.abs(lens - np.median(lens))))]
            idx = np.round(np.linspace(a, b, n_key + 1)).astype(int)
            bodies.append(take_frames(c["trial"], c["run"], idx, rate))
        viewer_html = cadence_viewer_html(bodies, rate,
                                          camera_offset=(1.55, -1.55, 0.35))
    t = np.linspace(0.0, period, n_plot + 1)
    sag_angle, sag_torque, front = [], [], []
    hdr = {k: [] for k in ("medial", "lateral", "lig_lateral", "total")}
    grf = []
    for st, c in zip(strides, conditions):
        chain, whole = c["run"][0], c["run"][3]
        sag = c3d_sagittal_series(None, chain[0], chain[1], whole.right, params)
        sag_angle.append(stride_mean(sag["angle"], st, n_plot))
        sag_torque.append(stride_mean(sag["torque"], st, n_plot))
        front.append(stride_mean(-whole.right.joint_torque_local[:, :3, 0],
                                 st, n_plot))
        bw = float(np.mean(chain[2].force[:, 2] + chain[5].force[:, 2]))
        grf.append(stride_mean(chain[2].force[:, 2] / bw, st, n_plot))
        for k in hdr:
            if c.get("contact") is not None:
                hdr[k].append(stride_mean(c["contact"][k], st, n_plot))
    sag_angle, sag_torque, front = map(np.array, (sag_angle, sag_torque, front))
    if hdr["total"]:
        header = ("knee contact\n(BW)", [
            ("medial (static opt)", JOINT_COLORS[0], None, np.array(hdr["medial"])),
            ("lateral (static opt)", JOINT_COLORS[1], None, np.array(hdr["lateral"])),
            ("lateral ligament tension needed", JOINT_COLORS[3], "2 2",
             np.array(hdr["lig_lateral"])),
            ("total", NEUTRAL, "5 3", np.array(hdr["total"]))])
    else:
        header = ("right vertical\nGRF (BW)", [
            ("right vertical GRF", JOINT_COLORS[0], None, np.array(grf))])
    rows = [(row, [(JOINT_COLORS[j], arr[:, :, j]) for j in range(3)])
            for row, arr in (("sagittal\nangle (deg)", sag_angle),
                             ("sagittal\ntorque (N m)", sag_torque),
                             ("frontal torque\n(N m, + ext.\nadduction)", front))]
    strips = strip_charts_grid(
        t, header, rows, ("ankle", "knee", "hip"), period_s=period,
        fps=rate, n_frames=n_key + 1, uid=uid, viewer_iframe=f"{uid}-viewer",
        variants=[c["label"] for c in conditions],
        variant_time_scale=list(period / t_stride), variant_ref=ref,
        phase_ticks=True)
    esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
    return (f'<iframe id="{uid}-viewer" class="viewer" srcdoc="{esc}"></iframe>'
            + strips)


#: Repo-local cache of the published running reference curves (see
#: `build_rbds_reference` for exactly what it holds and where it came from).
RBDS_NPZ = "rbds_running_reference.npz"


def build_rbds_reference(dest, cache_dir) -> str:
    """Build the running-reference file from the public RBDS dataset.

    Source: Fukuchi RK, Fukuchi CA, Duarte M (2017), "A public dataset of
    running biomechanics and the effects of running speed on lower extremity
    kinematics and kinetics", PeerJ 5:e3298 — data at
    https://doi.org/10.6084/m9.figshare.4543435 (CC-BY): 28 regular runners
    on an instrumented treadmill at 2.5, 3.5 and 4.5 m/s, Visual3D-processed.
    Each ``RBDS<nnn>processed.txt`` in `cache_dir` (downloaded from the
    figshare record) holds, per 1% of the RIGHT gait cycle, right hip/knee/
    ankle angles [deg], moments [N m/kg], GRF [body weight] and sagittal
    joint powers [W/kg].

    This resamples each subject's STANCE (foot-strike to the sample where
    the vertical GRF — their Y, NaN in swing — drops below 5% BW) onto
    0-100% and averages across subjects.  Moments keep the dataset's
    clinical sign convention (ankle plantarflexion, knee extension
    positive); the sagittal component is their Z.  Saved arrays, per speed
    tag ``25``/``35``: ``torque_<tag>`` and ``power_<tag>`` [101, 3]
    (ankle, knee, hip; N m/kg and W/kg), ``n_<tag>`` subjects, plus a
    ``provenance`` string.  The report only ever READS the saved file
    (`load_rbds_reference`) — no network at report time.
    """
    import glob
    import os

    grid = np.linspace(0.0, 1.0, 101)
    speeds = ("25", "35")
    acc = {s: {"torque": [], "power": []} for s in speeds}
    for path in sorted(glob.glob(os.path.join(os.fspath(cache_dir),
                                              "RBDS*processed.txt"))):
        raw = np.genfromtxt(path, names=True, delimiter="\t")
        for s in speeds:
            if f"RgrfY{s}" not in (raw.dtype.names or ()):
                continue
            fz = raw[f"RgrfY{s}"]
            on = np.isfinite(fz) & (fz > 0.05)
            if not on[0] or on.sum() < 20:
                continue                     # cycle must start in stance
            toe_off = int(np.flatnonzero(~on)[0])
            x = np.linspace(0.0, 1.0, toe_off)
            tq = np.stack([np.interp(grid, x, raw[f"R{j}MomZ{s}"][:toe_off])
                           for j in ("ankle", "knee", "hip")], axis=1)
            pw = np.stack([np.interp(grid, x, raw[f"R{j}Pow{s}"][:toe_off])
                           for j in ("ankle", "knee", "hip")], axis=1)
            if np.isfinite(tq).all() and np.isfinite(pw).all():
                acc[s]["torque"].append(tq)
                acc[s]["power"].append(pw)
    out = {"provenance": np.array(
        "Mean right-leg stance-normalized sagittal joint kinetics from the "
        "RBDS public running dataset (Fukuchi, Fukuchi & Duarte 2017, "
        "PeerJ 5:e3298; doi:10.6084/m9.figshare.4543435, CC-BY). "
        "Built by boneid.report.build_rbds_reference.")}
    for s in speeds:
        out[f"torque_{s}"] = np.mean(acc[s]["torque"], axis=0)
        out[f"power_{s}"] = np.mean(acc[s]["power"], axis=0)
        out[f"n_{s}"] = np.array(len(acc[s]["torque"]))
    np.savez(os.fspath(dest), **out)
    return os.fspath(dest)


def load_rbds_reference():
    """The saved RBDS running reference (see `build_rbds_reference`), or
    None when ``data/rbds_running_reference.npz`` is absent."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                        RBDS_NPZ)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def figure_toggle_block(uri_plain: str, uri_ref: str, number: int,
                        caption: str, button_label: str) -> str:
    """A figure with a button that swaps between two pre-rendered images
    (without / with the literature reference overlay)."""
    uid = f"figtog{number}"
    # the page CSS sets `figure img { display: block }`, which outranks the
    # `hidden` attribute's UA default — toggle inline display instead.
    return (
        f'<figure><img id="{uid}-a" src="{uri_plain}" alt="figure {number}">'
        f'<img id="{uid}-b" src="{uri_ref}" alt="figure {number} with '
        f'reference overlay" style="display:none">'
        f'<button class="figtog" style="margin:6px 6px 2px; padding:4px '
        f'10px; font:500 0.78rem \'IBM Plex Mono\',monospace; color:#2a78d6;'
        f' background:none; border:1px solid #dedcd4; border-radius:4px; '
        f'cursor:pointer" onclick="'
        f"var a=document.getElementById('{uid}-a'),"
        f"b=document.getElementById('{uid}-b'),"
        f"on=a.style.display!=='none';"
        f"a.style.display=on?'none':'block';"
        f"b.style.display=on?'block':'none';"
        f"this.textContent=on?'hide reference':'{button_label}';"
        f'">{button_label}</button>'
        f'<figcaption><b>Figure {number}.</b> {caption}</figcaption>'
        f"</figure>")


def c3d_report_html(trial, static, cal, skeleton, kin, ground, quality,
                    idres, audit, params, viewer_html: str | None = None,
                    title: str = "Static-Calibrated Inverse Dynamics",
                    source_note: str = "") -> str:
    """Report for one static-calibrated c3d running trial (io_c3d pipeline).

    `trial`/`static` are `io_c3d.C3dTrial`; `cal` an
    `io_c3d.StaticCalibration`; the rest come from `io_c3d.build_chain`,
    `core.inverse_dynamics` and `core.energy_audit` on the 4-segment chain.
    Only ONE leg is instrumented, so there are no bilateral panels; the top
    residual honestly carries everything unmodelled (upper body + left leg).
    """
    from .io_c3d import SEGMENTS

    t = kin.t
    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    events = right_step_events(ground, params.contact_threshold_n)
    bouts = quality["bouts"]
    n_mine = sum(1 for b in bouts if b["mine"])
    n_other = len(bouts) - n_mine
    win = c3d_viewer_window(ground, params)
    tw = t[win]

    # --- Fig: per-plate vertical force + attribution, over the viewer window
    fig, ax = new_fig(height=2.9)
    from .io_v3d import analog_to_mocap
    for p in range(trial.force.shape[1]):
        fz = analog_to_mocap(trial, trial.force[:, p, 2])[win]
        ax.plot(tw, fz, color=FAINT if p == 0 else NEUTRAL, lw=1.0,
                label=f"plate {p} Fz")
    ax.plot(tw, ground.force[win, 2], color=JOINT_COLORS[0], lw=1.4,
            label="attributed to right foot")
    ax.fill_between(tw, 0, ground.force[win, 2],
                    where=ground.force[win, 2] > params.contact_threshold_n,
                    color=JOINT_COLORS[0], alpha=0.12)
    ax.axhline(params.contact_threshold_n, color=FAINT, lw=1.0, ls="--")
    style_axes(ax, "time (s)", "vertical force (N)")
    ax.legend(frameon=False, fontsize=9)
    add(fig, f"Which foot was it? Both feet land on the same belt, so I "
             f"assign each contact by how close the COP is to the right-foot "
             f"markers: {n_mine} contacts kept (right), {n_other} dropped "
             f"(left foot).  Blue is the ground wrench the right-leg chain "
             f"actually sees, so the left foot's steps are zero here, which "
             f"is what we want.  Same time window as the animation.")

    # --- Fig: joint torque time series over the window (sagittal, lab x)
    joints = ["ankle", "knee", "hip"]
    fig, ax = new_fig(height=3.0)
    for j, name in enumerate(joints):
        ax.plot(tw, idres.joint_torque[win, j, 0], color=JOINT_COLORS[j],
                lw=1.2, label=name)
    for s, e in events:
        if s >= win.start and e <= win.stop:
            ax.axvspan(t[s], t[e], color=JOINT_COLORS[0], alpha=0.06, lw=0)
    style_axes(ax, "time (s)", "joint torque, lab x (N m)")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    add(fig, "Net joint torques (sagittal, lab x) over the same window. "
             "The shaded bands are right-foot stances.  The torques you see "
             "in swing are just the leg's own inertia, since the ground "
             "wrench is zero there by construction.")

    # --- Figs 3/4: stance-normalized torques and powers, with an optional
    # published-reference overlay (RBDS running dataset; button-toggled)
    ref = load_rbds_reference()
    m_kg = cal.body_mass
    prof = {name: stance_profiles(idres.joint_torque[:, j, 0], events)
            for j, name in enumerate(joints)}
    pct = np.linspace(0, 100, 101)
    # figures 3-5 run over stance TIME, not %: each curve is drawn against
    # its own real duration, so timing differences are visible instead of
    # being normalized away.  Ours: the mean measured stance; the reference:
    # the RBDS ensemble-mean stance at each speed (stored in the npz; the
    # single-subject 2.5 m/s value stands in until the sweep fills it).
    stance_s = float(np.mean([e - s for s, e in events]) / kin.rate)
    t_ours = pct / 100.0 * stance_s

    def ref_time(tag: str) -> np.ndarray:
        key = f"stance_s_{tag}"
        dur = float(ref[key]) if (ref is not None and key in ref) else 0.268
        return pct / 100.0 * dur

    # our lab-x torque axis vs the dataset's clinical sagittal convention
    # differ only by a per-joint sign; align by the mean-curve inner product
    # (decisive: |correlation| is high for every joint) and say so.
    ref_sign = np.ones(3)
    if ref is not None:
        for j, name in enumerate(joints):
            ref_sign[j] = np.sign(prof[name].mean(axis=0) @ ref["torque_25"][:, j])

    def torque_fig(with_ref: bool):
        fig, axes = new_grid(1, 3, height=2.9, width=9.2, sharey=True)
        for j, name in enumerate(joints):
            ax = axes[0, j]
            ax.plot(t_ours, prof[name].T, color=JOINT_COLORS[j], lw=0.6,
                    alpha=0.25)
            ax.plot(t_ours, prof[name].mean(axis=0), color=JOINT_COLORS[j],
                    lw=2.0)
            if with_ref:
                for tag, ls in (("25", "--"), ("35", ":")):
                    ax.plot(ref_time(tag),
                            ref_sign[j] * m_kg * ref[f"torque_{tag}"][:, j],
                            color=NEUTRAL, lw=1.6, ls=ls,
                            label=f"RBDS {tag[0]}.{tag[1]} m/s" if j == 0 else None)
            ax.axhline(0.0, color=FAINT, lw=0.8)
            ax.set_title(name, color=NEUTRAL, fontsize=10)
            style_axes(ax, "stance time (s)",
                       "torque, lab x (N m)" if j == 0 else "")
        if with_ref:
            axes[0, 0].legend(frameon=False, fontsize=8)
        return fig

    torque_caption = (
        f"The standard output: sagittal joint torque over stance, in "
        f"seconds, not % (0 = foot contact).  Each of the {len(events)} "
        f"right steps is a thin line drawn over the mean stance of "
        f"{stance_s:.3f} s, and the thick line is the mean.  Reference "
        f"curves run over their own mean stance durations, so you can "
        f"still see timing differences.  Peaks are about "
        + ", ".join(f"{np.abs(prof[nm].mean(axis=0)).max() / m_kg:.1f}"
                    for nm in joints)
        + " N m/kg (ankle, knee, hip).")
    if ref is None:
        add(torque_fig(False), torque_caption)
    else:
        n[0] += 1
        figs.append(figure_toggle_block(
            fig_svg(torque_fig(False)), fig_svg(torque_fig(True)), n[0],
            torque_caption + (
                f"  The toggle puts the RBDS public running dataset on top "
                f"(Fukuchi et al. 2017): mean sagittal moments for "
                f"n={int(ref['n_25'])} runners at 2.5 m/s (dashed) and "
                f"n={int(ref['n_35'])} at 3.5 m/s (dotted), which bracket "
                f"this trial's measured 2.8 m/s.  They're scaled by this "
                f"subject's {m_kg:.1f} kg, and I flipped their clinical sign "
                f"convention per joint to match our lab-x axis."),
            "show reference (RBDS)"))

    pw_prof = {name: stance_profiles(audit.joint_power[:, j], events)
               for j, name in enumerate(joints)}

    def power_fig(with_ref: bool):
        fig, axes = new_grid(1, 3, height=2.9, width=9.2, sharey=True)
        for j, name in enumerate(joints):
            ax = axes[0, j]
            ax.plot(t_ours, pw_prof[name].T, color=JOINT_COLORS[j], lw=0.6,
                    alpha=0.25)
            ax.plot(t_ours, pw_prof[name].mean(axis=0),
                    color=JOINT_COLORS[j], lw=2.0)
            if with_ref:
                for tag, ls in (("25", "--"), ("35", ":")):
                    ax.plot(ref_time(tag), m_kg * ref[f"power_{tag}"][:, j],
                            color=NEUTRAL, lw=1.6, ls=ls,
                            label=f"RBDS {tag[0]}.{tag[1]} m/s" if j == 0 else None)
            ax.axhline(0.0, color=FAINT, lw=0.8)
            ax.set_title(name, color=NEUTRAL, fontsize=10)
            style_axes(ax, "stance time (s)",
                       "joint power (W)" if j == 0 else "")
        if with_ref:
            axes[0, 0].legend(frameon=False, fontsize=8)
        return fig

    power_caption = (
        "Joint powers over stance (the full wrench power: τ·ω_rel plus "
        "the joint-force term).  The ankle absorbing and then generating "
        "is the classic running signature.")
    if ref is None:
        add(power_fig(False), power_caption)
    else:
        n[0] += 1
        figs.append(figure_toggle_block(
            fig_svg(power_fig(False)), fig_svg(power_fig(True)), n[0],
            power_caption + (
                "  The toggle puts the RBDS mean sagittal joint powers on "
                "top (2.5 m/s dashed, 3.5 m/s dotted), scaled to this "
                "subject's mass.  Powers don't need any sign flipping."),
            "show reference (RBDS)"))

    # --- Fig: distal-to-foot (UD) power and what it does to "ankle power"
    from .core import foot_power_ud
    belt = c3d_belt_velocity(trial, ground, params.contact_threshold_n)
    plane = float(np.mean([p.corners[:, 2].mean() for p in trial.plates]))
    p_ud = foot_power_ud(skeleton, kin, ground, params, plane_height=plane,
                         surface_velocity=belt)
    p_ankle = audit.joint_power[:, 0]
    p_shank = p_ankle + p_ud                 # everything distal to the shank

    def step_work(p):
        """[K,2] positive / negative work per stance, J."""
        dt = 1.0 / kin.rate
        return np.array([[np.clip(p[s:e + 1], 0, None).sum() * dt,
                          np.clip(p[s:e + 1], None, 0).sum() * dt]
                         for s, e in events])

    w_ank, w_ud, w_sh = (step_work(p) for p in (p_ankle, p_ud, p_shank))

    def ud_fig(with_ref: bool):
        fig, axes = new_grid(1, 3, height=2.9, width=9.2, sharey=True)
        for k, (p, name, color) in enumerate((
                (p_ankle, "ankle joint", JOINT_COLORS[0]),
                (p_ud, "distal-to-foot (UD)", JOINT_COLORS[3]),
                (p_shank, "sum: distal-to-shank", JOINT_COLORS[2]))):
            prof_p = stance_profiles(p, events)
            ax = axes[0, k]
            ax.plot(t_ours, prof_p.T, color=color, lw=0.6, alpha=0.25)
            ax.plot(t_ours, prof_p.mean(axis=0), color=color, lw=2.0)
            if with_ref and k == 0:
                for tag, ls in (("25", "--"), ("35", ":")):
                    ax.plot(ref_time(tag), m_kg * ref[f"power_{tag}"][:, 0],
                            color=NEUTRAL, lw=1.6, ls=ls,
                            label=f"RBDS ankle {tag[0]}.{tag[1]} m/s")
                ax.legend(frameon=False, fontsize=8)
            ax.axhline(0.0, color=FAINT, lw=0.8)
            ax.set_title(name, color=NEUTRAL, fontsize=10)
            style_axes(ax, "stance time (s)", "power (W)" if k == 0 else "")
        return fig

    ud_ref_note = (
        "  The toggle puts the RBDS mean ankle power on the left panel.  "
        "I couldn't find public numbers for the UD term, but running "
        "studies keep reporting the distal foot as a net absorber (Kelly "
        "et al. 2015; Riddick &amp; Kuo 2016; Stearne et al. 2016), which "
        "matches the middle panel."
        if ref is not None else "")
    ud_caption = (f"Foot work, the way Zelik &amp; Honert (2018) do it.  "
             f"Middle: the unified-deformable distal-to-foot power "
             f"(Takahashi et al. 2012), which is the GRF acting through the "
             f"foot's COP-coincident rigid-body point.  In other words, "
             f"everything the shoe, heel pad, arch and foot joints do that a "
             f"rigid foot can't.  I evaluate it in the belt frame (measured "
             f"belt speed {np.linalg.norm(belt):.2f} m/s), so it's the "
             f"overground-equivalent number.  Left: the usual ankle joint "
             f"power.  Right: their sum, the total power of everything below "
             f"the shank, which I think is the honest \"ankle\" number.  "
             f"Per stance, the ankle joint alone does "
             f"+{w_ank[:, 0].mean():.1f} / {w_ank[:, 1].mean():.1f} J; the "
             f"distal foot adds +{w_ud[:, 0].mean():.1f} / "
             f"{w_ud[:, 1].mean():.1f} J, mostly absorption, so the total "
             f"below the shank is +{w_sh[:, 0].mean():.1f} / "
             f"{w_sh[:, 1].mean():.1f} J.  Treating the foot as rigid "
             f"overstates net ankle work by "
             f"{(w_ank.sum(axis=1).mean() - w_sh.sum(axis=1).mean()):+.1f} J "
             f"per step ("
             f"{100 * (1 - w_sh.sum(axis=1).mean() / max(w_ank.sum(axis=1).mean(), 1e-9)):.0f}%"
             f" of the net).  This markerset could even support a real MTP "
             f"joint (forefoot cluster MTP1/MTP5/ToeboxAnt); the UD measure "
             f"just lumps it into the one distal term.")
    if ref is None:
        add(ud_fig(False), ud_caption)
    else:
        n[0] += 1
        figs.append(figure_toggle_block(
            fig_svg(ud_fig(False)), fig_svg(ud_fig(True)), n[0],
            ud_caption + ud_ref_note, "show reference (RBDS)"))

    # --- Fig: the published running reference, for visual comparison
    import base64 as _b64
    import os as _os
    fs_png = _os.path.join(_os.path.dirname(__file__), "..", "..", "data",
                           "farris2012_fig2_running.png")
    if _os.path.exists(fs_png):
        with open(fs_png, "rb") as fh:
            uri = ("data:image/png;base64,"
                   + _b64.b64encode(fh.read()).decode())
        n[0] += 1
        fs_caption = (
            "A published reference to eyeball against: group-mean (d) "
            "ankle, (e) knee and (f) hip joint powers for treadmill running, "
            "from the running column of Fig. 2 of Farris &amp; Sawicki (2012), "
            "\"The mechanics and energetics of human walking and running: a "
            "joint level perspective\", J R Soc Interface 9:110-118 "
            "(reproduced for scholarly comparison; &copy; the authors / "
            "Royal Society).  Dashed 2.0, dotted 2.25, grey 2.75, black "
            "3.25 m/s; the grey curve is closest to this trial's 2.8 m/s.  "
            "Careful: their x-axis is % of stride, not % of stance.  Stance "
            "ends around 35-40% of stride, so squish our Figures 4-5 into "
            "the left third or so of each panel when you compare.  In W/kg, "
            "this subject's mean ankle power peaks at "
            f"{np.max(stance_profiles(p_ankle, events).mean(axis=0)) / m_kg:.1f}"
            " W/kg vs their ~7 W/kg at 2.75 m/s.")
        figs.append(
            f'<figure><img style="max-width:340px;display:block;'
            f'margin:0 auto" src="{uri}" alt="Farris and Sawicki 2012 '
            f'running joint powers">'
            f'<figcaption><b>Figure {n[0]}.</b> {fs_caption}</figcaption>'
            f'</figure>')

    # --- Fig: cluster fit residuals (data quality)
    fig, ax = new_fig(height=2.6)
    resid = quality["cluster_residual"]
    for i, seg in enumerate(SEGMENTS):
        ax.plot(t, 1000.0 * resid[:, i], color=JOINT_COLORS[i], lw=0.9,
                label=seg)
    style_axes(ax, "time (s)", "cluster fit residual (mm)")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    add(fig, "Data quality: for every frame, the RMS residual of the SVD "
             "rigid-body fit of each marker cluster against its static "
             "shape.  Soft tissue, marker wobble and any leftover labelling "
             "trouble show up here.  The pelvis cluster includes the "
             "skin-mounted GT markers, which is why its floor is higher.")

    # --- Fig: energy audit + residual
    edge = slice(30, -30)
    imb = float(np.abs(audit.imbalance[edge]).max())
    pk = float(max(np.abs(audit.de_dt[edge]).max(), 1e-9))
    res_mean = idres.residual_force[edge].mean(axis=0)
    unmod_n = (cal.body_mass - float(skeleton.mass.sum())) * 9.81
    fig, axes = new_grid(1, 2, height=2.9, width=9.2)
    axes[0, 0].plot(tw, audit.de_dt[win], color=NEUTRAL, lw=2.4, alpha=0.4,
                    label="d(KE+PE)/dt")
    axes[0, 0].plot(tw, audit.power_total[win], color=JOINT_COLORS[0], lw=1.0,
                    label="summed wrench power")
    style_axes(axes[0, 0], "time (s)", "power (W)")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for c, lab in enumerate("xyz"):
        axes[0, 1].plot(tw, idres.residual_force[win, c],
                        color=JOINT_COLORS[c], lw=1.0, label=f"F{lab}")
    axes[0, 1].axhline(-unmod_n, color=NEUTRAL, lw=1.3, ls="--",
                       label="− unmodelled weight")
    style_axes(axes[0, 1], "time (s)", "residual force at L5S1 (N)")
    axes[0, 1].legend(frameon=False, fontsize=8.5, ncol=2)
    add(fig, f"Left: the energy audit, d(KE+PE)/dt of the modelled chain vs "
             f"the summed wrench powers.  Max imbalance is {imb:.0f} W, "
             f"{100 * imb / pk:.1f}% of peak.  Right: the residual wrench at "
             f"the top of the chain.  Unlike the whole-body pipeline, this "
             f"model only has one leg and a pelvis, so the residual is NOT "
             f"near zero, and it shouldn't be: it's standing in for the "
             f"torso, arms and left leg "
             f"({cal.body_mass - float(skeleton.mass.sum()):.1f} kg, weight "
             f"{unmod_n:.0f} N, dashed).  Its mean vertical of "
             f"{res_mean[2]:.0f} N would only sit near that line if we "
             f"averaged over whole strides of both feet; here it only sees "
             f"the right side's loading.")

    max_res = 1000.0 * float(np.median(resid.max(axis=0)))
    stats = "".join([
        stat("body mass (static trial)", f"{cal.body_mass:.1f}", "kg"),
        stat("right steps analysed", f"{len(events)}", ""),
        stat("energy imbalance", f"{100 * imb / pk:.1f}", "% of peak dE/dt",
             "pass" if imb < 0.03 * pk else "fail"),
        stat("cluster residual (median of per-segment max)",
             f"{max_res:.0f}", "mm"),
    ])

    viewer = ""
    if viewer_html is not None:
        # strips share the animation's exact keyframe window and period
        decimate = c3d_decimate(kin)
        n_key = len(range(0, win.stop - win.start, decimate))
        fps = kin.rate / decimate
        period = (n_key - 1) / fps
        ssl = slice(win.start, win.start + (n_key - 1) * decimate + 1)
        sag = c3d_sagittal_series(cal, skeleton, kin, idres, params)
        grid_rows = [
            (label, [(JOINT_COLORS[j], sag[key][ssl, j]) for j in range(3)])
            for label, key in (("angle (deg)\nvs static", "angle"),
                               ("velocity\n(deg/s)", "vel"),
                               ("torque\n(N m)", "torque"))]
        strips = strip_charts_grid(
            t[ssl],
            ("GRF (N)",
             [("vertical", JOINT_COLORS[0], None, ground.force[ssl, 2]),
              ("anterior-posterior", JOINT_COLORS[2], None,
               ground.force[ssl, 1]),
              ("medio-lateral", JOINT_COLORS[3], "4 3",
               ground.force[ssl, 0])]),
            grid_rows, ("ankle", "knee", "hip"),
            period_s=period, fps=fps, n_frames=n_key, uid="c3dgrid",
            viewer_iframe="c3d-viewer")

        esc = viewer_html.replace("&", "&amp;").replace('"', "&quot;")
        viewer = (
            '<h2>The trial, moving (go ahead and scrub it)</h2>'
            '<p>This is the instrumented right leg and the pelvis over the '
            'first clean steps.  Segment volumes come from the de Leva '
            'masses, the foot is a rounded blob bounded by the static heel, '
            'forefoot, toebox and ankle markers, the pelvis band spans both '
            'hip centres, every named marker is red, and the arrow is the '
            'ground-reaction force we attributed to this foot.  Drag to '
            'orbit, scroll to zoom; the clip loops.  <b>Open Controls '
            '&rarr; Animations</b> for play/pause and a time scrubber.</p>'
            '<p>Underneath is the ground reaction (top) and the classic '
            '3&times;3: sagittal angle, angular velocity and torque for '
            'ankle, knee and hip, each row on one shared scale.  It works '
            'like a <b>scrolling scope</b>: the data streams right to left, '
            'and the orange centre line is "now" in the animation.  The '
            'scope <b>follows the viewer\'s own clock</b>, so if you pause, '
            'it freezes, and if you drag the scrubber, every panel comes '
            f'with it (period {period:.2f} s, {n_key} keyframes at '
            f'{fps:.0f} fps; until the viewer loads, a wall-clock loop fills '
            'in).  Angles are zeroed at the static pose, so 0&deg; means '
            'standing.</p>'
            f'<iframe id="c3d-viewer" class="viewer" srcdoc="{esc}"></iframe>'
            f'{strips}')

    steps_html = deleva_table_html(skeleton, cal.body_mass, cal.sex)
    body = f"""
<p class="eyebrow">boneid · io_c3d · static-calibrated pipeline</p>
<h1>{title}</h1>
<p class="subtitle">{source_note} {len(t) / kin.rate:.1f} s of running at
{kin.rate:.0f} Hz mocap / {trial.analog_rate:.0f} Hz analog, calibrated from
a standing static trial.</p>

<div class="stat-row">{stats}</div>

<p>The pipeline steps and the static-calibration story are in the notebook
above this report, so I don't repeat them here.  This part is the figures.</p>

{viewer}

{"".join(figs)}

<h2>Segment parameters (de Leva 1996, static lengths)</h2>
{steps_html}
"""
    body += joint_dof_comparison_html(audit, skeleton.joint_names)
    return html_page(title, "c3d", body)


# ---------------------------------------------------------------------------
# Interactive playground (p1): the quickstart with knobs, computed in JS
# ---------------------------------------------------------------------------

def _b64_f32(a: np.ndarray) -> str:
    import base64
    return base64.b64encode(
        np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def playground_data(trial, cutoffs=(8.0, 12.0, 20.0)) -> dict:
    """Everything the in-browser whole-body recursion needs, precomputed.

    The lag/threshold/force-filter knobs only touch the GROUND WRENCH and
    the recursion — pure per-frame algebra — while the expensive part
    (cluster-free marker kinematics: filtered poses, accelerations, and the
    rotational term I·α + ω×(I·ω) per segment) depends only on the KINEMATIC
    cutoff, so it is computed here once per cutoff choice and embedded.
    The JS then mirrors `core.inverse_dynamics_whole_body` exactly:
    3-segment leg recursions, pelvis balance with both hip reactions
    (top output = the L5S1 wrench), 2-segment arm recursions, torso balance
    (top output = the residual at the torso COM).
    """
    from . import io_v3d as io
    from .core import (AnalysisParams, _alpha_of, _rot_term, chain_kinematics,
                       slice_chain, wrench_from_cop)

    params = p1_params()
    skel_r, kin_r, _ = io.build_chain(trial, side="r")
    skel_l, kin_l, _ = io.build_chain(trial, side="l")
    skel_u, kin_u = io.build_upper_body(trial)

    units = {}
    for c_i, hz in enumerate(cutoffs):
        p = AnalysisParams(lowpass_hz=float(hz), force_lowpass_hz=0.0,
                           contact_threshold_n=params.contact_threshold_n)

        def pack(skel, kin, name):
            ck = chain_kinematics(skel, kin, p.lowpass_hz, p.filter_order)
            alpha = _alpha_of(ck, kin.rate, p, None)
            s_n = ck["com"].shape[1]
            rot = np.stack([_rot_term(ck["r"][:, s], skel.inertia_local[s],
                                      ck["omega"][:, s], alpha[:, s], None)
                            for s in range(s_n)], axis=1)
            units[f"c{c_i}_{name}"] = {
                "com": _b64_f32(ck["com"]), "acc": _b64_f32(ck["a_com"]),
                "prox": _b64_f32(ck["prox"]), "rot": _b64_f32(rot),
                "n_seg": s_n}

        pack(*slice_chain(skel_r, kin_r, [0, 1, 2]), "legR")
        pack(*slice_chain(skel_l, kin_l, [0, 1, 2]), "legL")
        pack(*slice_chain(skel_r, kin_r, [3]), "pelvis")
        pack(skel_u, kin_u, "upper")     # [r_fa, r_ua, l_fa, l_ua, torso]

    # analog-rate ground wrench about the origin, masked exactly as
    # io_v3d.ground_wrench does (COP is garbage below 20 N)
    f = trial.force.copy()                      # [Ta,2,3]
    m = np.empty_like(f)
    for pl in range(f.shape[1]):
        on = f[:, pl, 2] > 20.0
        m[:, pl] = wrench_from_cop(trial.force[:, pl], trial.cop[:, pl],
                                   trial.free_moment[:, pl], np.zeros(3))
        f[~on, pl] = 0.0
        m[~on, pl] = 0.0

    ref = np.stack([io.reference_series(trial, n)[:, 0]
                    for side in ("r", "l")
                    for n in v3d_ref_torque(side)])     # [6,T] sagittal

    return {
        "T": len(trial.t), "rate": trial.rate,
        "Ta": len(trial.t_analog), "arate": trial.analog_rate,
        "cutoffs": list(map(float, cutoffs)),
        "mass": {"legR": skel_r.mass[:3].tolist(),
                 "legL": skel_l.mass[:3].tolist(),
                 "pelvis": float(skel_r.mass[3]),
                 "upper": skel_u.mass.tolist()},
        "units": units,
        "wf": _b64_f32(f), "wm": _b64_f32(m),
        "ref": _b64_f32(ref),
        "body_mass": float(io.estimate_body_mass(trial)),
    }


PLAYGROUND_TEMPLATE = r"""<title>Inverse Dynamics Playground</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root { --paper:#f7f6f2; --card:#fcfcfb; --ink:#1c1b18; --ink2:#52514e;
          --accent:#2a78d6; --rule:#dedcd4; }
  html { color-scheme: light; }
  body { background:var(--paper); color:var(--ink); margin:0;
         font:16px/1.55 "IBM Plex Sans",system-ui,sans-serif; }
  main { max-width:900px; margin:0 auto; padding:44px 24px 96px; }
  h1,h2 { font-family:"Source Serif 4",Georgia,serif; line-height:1.15; }
  h1 { font-size:2.05rem; margin:0 0 4px; }
  h2 { font-size:1.3rem; margin:34px 0 8px; border-top:1px solid var(--rule);
       padding-top:22px; }
  p { max-width:68ch; }
  .eyebrow { font:600 .72rem/1 "IBM Plex Mono",monospace; letter-spacing:.14em;
             text-transform:uppercase; color:var(--accent); margin:0 0 10px; }
  .subtitle { color:var(--ink2); }
  pre.cmds { background:var(--card); border:1px solid var(--rule);
             border-radius:6px; padding:14px 16px; overflow-x:auto;
             font:12.5px/1.75 "IBM Plex Mono",monospace; color:var(--ink2); }
  pre.cmds b { color:var(--ink); font-weight:500; }
  pre.cmds .p { color:var(--accent); font-weight:600; }
  .controls { display:flex; gap:10px; margin:18px 0; }
  .ctl { background:var(--card); border:1px solid var(--rule); border-radius:6px;
         padding:8px 12px; flex:1; min-width:0; }
  .ctl .k { font:600 .62rem/1.3 "IBM Plex Mono",monospace; letter-spacing:.06em;
            text-transform:uppercase; color:var(--ink2); white-space:nowrap; }
  .ctl .v { font-size:1.05rem; font-weight:600; margin:2px 0 4px;
            font-variant-numeric:tabular-nums; white-space:nowrap; }
  input[type=range] { width:100%; accent-color:var(--accent); margin:0; }
  select { font:inherit; font-size:.95rem; padding:1px 4px;
           border:1px solid var(--rule); border-radius:4px; background:#fff; }
  .chips { display:flex; gap:10px; flex-wrap:wrap; margin:16px 0; }
  .chip { background:var(--card); border:1px solid var(--rule); border-radius:6px;
          padding:8px 12px; flex:1; min-width:140px; }
  .chip .k { font:500 .62rem/1.2 "IBM Plex Mono",monospace; letter-spacing:.06em;
             text-transform:uppercase; color:var(--ink2); }
  .chip .v { font-size:1.2rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .chip .u { font-size:.78rem; color:var(--ink2); font-weight:400; }
  .grid { display:grid; grid-template-columns:1fr 262px; gap:14px;
          align-items:start; }
  .panel { background:var(--card); border:1px solid var(--rule); border-radius:6px;
           padding:10px; }
  .panel.muted { opacity:.85; }
  canvas { width:100%; display:block; }
  .torques { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
  figcaption,.note { font-size:.84rem; color:var(--ink2); padding-top:6px;
                     line-height:1.45; }
  @media (max-width:760px){ .grid { grid-template-columns:1fr; } }
</style>
<main>
<p class="eyebrow">boneid · interactive · subject P1 · trial 13</p>
<h1>Inverse Dynamics Playground</h1>
<p class="subtitle">The whole-body quickstart with its choices exposed.
Every control is a real analysis decision; changing one re-runs the full
12-segment Newton&ndash;Euler recursion <b>in your browser</b> on the
embedded walking trial (120&nbsp;Hz mocap / 1200&nbsp;Hz forces) and
redraws everything below.</p>

<h2>The ten commands</h2>
<p>Exactly the calls <code>examples/quickstart.py</code> makes, with the
knob values substituted where they enter (<span style="color:var(--accent);
font-weight:600">blue</span> = live):</p>
<pre class="cmds"> 1  <b>trial</b>  = load_v3d_trial("p1_5StridesData.mat", trial=13)
 2  <b>forces</b> ride on the trial at 1200 Hz; the offset knob evaluates F(t &minus; &tau;), &tau; = <span class="p" id="c_lag">0</span> ms
 3  <b>mass</b>   = estimate_body_mass(trial)  &rarr; 85.0 kg   (full-stride mean of &Sigma;Fz/g; computed ONCE,
    held fixed across every knob &mdash; and insensitive to &tau; anyway: &plusmn;50 ms moves it &lt;0.2%)
 4  <b>params</b> = AnalysisParams(contact_threshold_n=<span class="p c_thr">20</span>, lowpass_hz=<span class="p" id="c_k">12</span>, force_lowpass_hz=<span class="p" id="c_f">50</span>)
 5  <b>chains</b>: build_chain(trial, side="r"|"l"), build_upper_body(trial)
 6  <b>contact</b> = detect_contact(ground.force[:,2], threshold=<span class="p c_thr">20</span>)
 7  <b>whole</b>  = inverse_dynamics_whole_body(legs, upper, params)
 8  <b>outputs</b>: whole.right/.left.joint_torque, whole.residual_force
 9  <b>audit</b>  = energy_audit_whole_body(...)   (see the main report)
10  <b>figures</b> below, live</pre>

<div class="controls">
  <div class="ctl"><div class="k">kinematic low-pass</div>
    <div class="v"><select id="kcut"><option>8</option>
      <option selected>12</option><option>20</option></select> Hz</div></div>
  <div class="ctl"><div class="k">force low-pass</div>
    <div class="v"><select id="fcut"><option selected>50</option>
      <option>25</option><option>12</option></select> Hz</div></div>
  <div class="ctl"><div class="k">contact threshold</div>
    <div class="v"><span id="thrv">20</span> N</div>
    <input id="thr" type="range" min="10" max="60" step="5" value="20"></div>
</div>

<h2>The result</h2>
<div class="chips" id="chips"></div>
<div class="grid">
  <div>
    <div class="panel"><canvas id="cRes" height="200"></canvas>
      <figcaption>Whole-body residual force at the torso COM &mdash; the
      model&rsquo;s error term (x blue, y orange, z green).</figcaption></div>
    <div class="torques" style="margin-top:14px">
      <div class="panel"><canvas id="cT0" height="180"></canvas></div>
      <div class="panel"><canvas id="cT1" height="180"></canvas></div>
      <div class="panel"><canvas id="cT2" height="180"></canvas></div>
    </div>
    <figcaption style="padding:6px 2px 0">Sagittal net joint torques:
    right leg solid, left dashed, Visual3D&rsquo;s export grey (computed by
    them at zero offset, so it is the fixed reference).</figcaption>
  </div>
  <div class="panel">
    <div class="k" style="font:600 .62rem/1.3 'IBM Plex Mono',monospace;
      letter-spacing:.06em; text-transform:uppercase; color:var(--ink2)">
      force &harr; mocap time offset</div>
    <div style="font-size:1.15rem; font-weight:600; margin:2px 0 4px">
      &tau; = <span id="lagv">0</span> ms</div>
    <input id="lag" type="range" min="-50" max="50" step="1" value="0">
    <canvas id="cLag" height="190" style="margin-top:10px"></canvas>
    <figcaption>Sum of squared residual force over the whole trial, at
    every offset &tau; (dot = current) &mdash; the time-sweep sync check.
    The two systems are <i>assumed</i> synchronized; using the residual to
    detect mis-sync is an established idea: &ldquo;If the other sources of
    experimental error are small, it is possible that residual forces can
    be helpful for identifying the presence or sensitivity of
    mis-synchronization&rdquo; (Lemaire &amp; Kuo 2024, bioRxiv 600392,
    who report residuals growing ~1% body weight per ms in walking; see
    also G&uuml;nther et al. 2005). Squaring
    SHARPENS the minimum rather than blurring it (&plusmn;5 ms raises the
    sum-squared 9% vs 4% for mean-absolute; both agree on the location to
    1 ms) &mdash; but the bottom is still shallow, so single-digit-ms lags
    remain uninferable. If the minimum sits a few ms below 0 (the curve is
    sampled every 2&nbsp;ms), that is at least consistent with
    the ~8 ms internal processing delay Lemaire &amp; Kuo measured on the
    same lab&rsquo;s Bertec treadmill that recorded this very
    dataset.</figcaption>
  </div>
</div>
<div class="panel muted" style="margin-top:14px">
  <canvas id="cGrf" height="150"></canvas>
  <figcaption>For reference only: vertical force per belt (right blue,
  left green), detected contact shaded, threshold dashed.</figcaption></div>
<p class="note">Same physics as <code>boneid.core</code>: legs bottom-up,
pelvis balanced with both hip reactions, arms into the torso, torso balance
&rarr; residual. Segment kinematics precomputed per filter choice by
<code>report.playground_data</code>; wrench shifting, filtering, contact
detection and every joint balance run live in ~100 lines of JS.
Prefer Python? The same ten steps, knobs and time-sweep live as a notebook:
<code>notebooks/p1_playground.ipynb</code> (also on Colab via the repo).</p>
</main>
<script>
const D = __DATA__;
function f32(b64){ const s=atob(b64); const u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++) u[i]=s.charCodeAt(i);
  return new Float32Array(u.buffer); }
const T=D.T, RATE=D.rate, TA=D.Ta, ARATE=D.arate, G=[0,0,-9.81], E0=10;
const WF=f32(D.wf), WM=f32(D.wm), REF=f32(D.ref);
const UNITS={};
for(const k in D.units){ const u=D.units[k];
  UNITS[k]={n:u.n_seg, com:f32(u.com), acc:f32(u.acc),
            prox:f32(u.prox), rot:f32(u.rot)}; }
function v(arr,t,s,c,nseg){ return arr[(t*nseg+s)*3+c]; }
function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                             a[0]*b[1]-a[1]*b[0]]; }
function filtfilt(x, fc, fs){
  const c=1/Math.tan(Math.PI*fc/fs), c2=c*c, s2=Math.SQRT2*c;
  const a0=1+s2+c2, b0=1/a0, b1=2/a0, b2=1/a0,
        a1=2*(1-c2)/a0, a2=(1-s2+c2)/a0;
  function pass(src){ const y=new Float32Array(src.length);
    let x1=src[0],x2=src[0],y1=src[0],y2=src[0];
    for(let i=0;i<src.length;i++){ const xi=src[i];
      const yi=b0*xi+b1*x1+b2*x2-a1*y1-a2*y2;
      y[i]=yi; x2=x1; x1=xi; y2=y1; y1=yi; }
    return y; }
  const fwd=pass(x); fwd.reverse(); const back=pass(fwd); back.reverse();
  return back; }
// pre-filter the analog wrench once per force-cutoff; lag only re-samples
let FCOLS=null, FCUT=-1;
function beltWrench(lagS, fcut){
  if(fcut!==FCUT){ FCOLS=[];
    for(let p=0;p<2;p++){ const cols=[];
      for(let arr of [WF,WM]) for(let c=0;c<3;c++){
        const col=new Float32Array(TA);
        for(let k=0;k<TA;k++) col[k]=arr[(k*2+p)*3+c];
        cols.push(filtfilt(col, fcut, ARATE)); }
      FCOLS.push(cols); }
    FCUT=fcut; }
  const out=[];
  for(let p=0;p<2;p++){
    const cols=FCOLS[p];
    const F=new Float32Array(T*3), M=new Float32Array(T*3);
    for(let i=0;i<T;i++){
      const ta=(i/RATE - lagS)*ARATE;
      const k=Math.max(0, Math.min(TA-2, Math.floor(ta)));
      const w=Math.max(0, Math.min(1, ta-k));
      for(let c=0;c<3;c++){
        F[i*3+c]=cols[c][k]*(1-w)+cols[c][k+1]*w;
        M[i*3+c]=cols[3+c][k]*(1-w)+cols[3+c][k+1]*w; }
    }
    out.push({F:F, M:M});
  }
  return out; }
function serial(unit, masses, Fd, Md, Pd){
  const n=unit.n, J=[];
  for(let j=0;j<n;j++) J.push({f:new Float32Array(T*3),
                               m:new Float32Array(T*3)});
  for(let i=0;i<T;i++){
    let fd=[Fd[i*3],Fd[i*3+1],Fd[i*3+2]],
        md=[Md[i*3],Md[i*3+1],Md[i*3+2]],
        pd=[Pd[i*3],Pd[i*3+1],Pd[i*3+2]];
    for(let s=0;s<n;s++){
      const m=masses[s];
      const com=[v(unit.com,i,s,0,n),v(unit.com,i,s,1,n),v(unit.com,i,s,2,n)];
      const acc=[v(unit.acc,i,s,0,n),v(unit.acc,i,s,1,n),v(unit.acc,i,s,2,n)];
      const rot=[v(unit.rot,i,s,0,n),v(unit.rot,i,s,1,n),v(unit.rot,i,s,2,n)];
      const pp=[v(unit.prox,i,s,0,n),v(unit.prox,i,s,1,n),v(unit.prox,i,s,2,n)];
      const fp=[0,0,0], mp=[0,0,0];
      for(let c=0;c<3;c++) fp[c]=m*acc[c]-fd[c]-m*G[c];
      const c1=cross([pd[0]-com[0],pd[1]-com[1],pd[2]-com[2]],fd);
      const c2=cross([pp[0]-com[0],pp[1]-com[1],pp[2]-com[2]],fp);
      for(let c=0;c<3;c++) mp[c]=rot[c]-md[c]-c1[c]-c2[c];
      for(let c=0;c<3;c++){ J[s].f[i*3+c]=fp[c]; J[s].m[i*3+c]=mp[c]; }
      fd=[-fp[0],-fp[1],-fp[2]]; md=[-mp[0],-mp[1],-mp[2]]; pd=pp;
    }
  }
  return J; }
function balance(unit, mass, loads, aboutCom){
  const rf=new Float32Array(T*3), rm=new Float32Array(T*3);
  for(let i=0;i<T;i++){
    const com=[v(unit.com,i,0,0,1),v(unit.com,i,0,1,1),v(unit.com,i,0,2,1)];
    const acc=[v(unit.acc,i,0,0,1),v(unit.acc,i,0,1,1),v(unit.acc,i,0,2,1)];
    const rot=[v(unit.rot,i,0,0,1),v(unit.rot,i,0,1,1),v(unit.rot,i,0,2,1)];
    const fs=[0,0,0], ms=[0,0,0];
    for(const L of loads){
      const f=[-L.f[i*3],-L.f[i*3+1],-L.f[i*3+2]];
      const m=[-L.m[i*3],-L.m[i*3+1],-L.m[i*3+2]];
      const p=[L.p[i*3],L.p[i*3+1],L.p[i*3+2]];
      const cr=cross([p[0]-com[0],p[1]-com[1],p[2]-com[2]],f);
      for(let c=0;c<3;c++){ fs[c]+=f[c]; ms[c]+=m[c]+cr[c]; }
    }
    for(let c=0;c<3;c++){
      rf[i*3+c]=mass*acc[c]-fs[c]-mass*G[c];
      rm[i*3+c]=rot[c]-ms[c]; }
    if(!aboutCom){
      const pp=[v(unit.prox,i,0,0,1),v(unit.prox,i,0,1,1),v(unit.prox,i,0,2,1)];
      const cr=cross([pp[0]-com[0],pp[1]-com[1],pp[2]-com[2]],
                     [rf[i*3],rf[i*3+1],rf[i*3+2]]);
      for(let c=0;c<3;c++) rm[i*3+c]-=cr[c];
    }
  }
  return {f:rf, m:rm}; }
function subProx(unit, s){
  const out=new Float32Array(T*3); const n=unit.n;
  for(let i=0;i<T;i++) for(let c=0;c<3;c++) out[i*3+c]=v(unit.prox,i,s,c,n);
  return out; }
function subUnit(unit, s){
  const n=unit.n, com=new Float32Array(T*3), acc=new Float32Array(T*3),
        prox=new Float32Array(T*3), rot=new Float32Array(T*3);
  for(let i=0;i<T;i++) for(let c=0;c<3;c++){
    com[i*3+c]=v(unit.com,i,s,c,n); acc[i*3+c]=v(unit.acc,i,s,c,n);
    prox[i*3+c]=v(unit.prox,i,s,c,n); rot[i*3+c]=v(unit.rot,i,s,c,n); }
  return {n:1, com:com, acc:acc, prox:prox, rot:rot}; }
function subChain(unit, idx){
  const n=unit.n, m=idx.length;
  const com=new Float32Array(T*m*3), acc=new Float32Array(T*m*3),
        prox=new Float32Array(T*m*3), rot=new Float32Array(T*m*3);
  for(let i=0;i<T;i++) for(let k=0;k<m;k++) for(let c=0;c<3;c++){
    const o=(i*m+k)*3;
    com[o+c]=v(unit.com,i,idx[k],c,n); acc[o+c]=v(unit.acc,i,idx[k],c,n);
    prox[o+c]=v(unit.prox,i,idx[k],c,n); rot[o+c]=v(unit.rot,i,idx[k],c,n); }
  return {n:m, com:com, acc:acc, prox:prox, rot:rot}; }
function run(lagMs, kIdx, fcut){
  const belts=beltWrench(lagMs/1000, fcut);
  const legR=UNITS["c"+kIdx+"_legR"], legL=UNITS["c"+kIdx+"_legL"];
  const pelv=UNITS["c"+kIdx+"_pelvis"], up=UNITS["c"+kIdx+"_upper"];
  const zero=new Float32Array(T*3);
  const JR=serial(legR, D.mass.legR, belts[1].F, belts[1].M, zero);
  const JL=serial(legL, D.mass.legL, belts[0].F, belts[0].M, zero);
  const l5=balance(pelv, D.mass.pelvis,
    [{f:JR[2].f, m:JR[2].m, p:subProx(legR,2)},
     {f:JL[2].f, m:JL[2].m, p:subProx(legL,2)}], false);
  const armR=serial(subChain(up,[0,1]), [D.mass.upper[0],D.mass.upper[1]],
                    zero, zero, zero);
  const armL=serial(subChain(up,[2,3]), [D.mass.upper[2],D.mass.upper[3]],
                    zero, zero, zero);
  const res=balance(subUnit(up,4), D.mass.upper[4],
    [{f:l5.f, m:l5.m, p:subProx(pelv,0)},
     {f:armR[1].f, m:armR[1].m, p:subProx(subChain(up,[0,1]),1)},
     {f:armL[1].f, m:armL[1].m, p:subProx(subChain(up,[2,3]),1)}], true);
  return {belts:belts, torq:[JR[0].m,JR[1].m,JR[2].m,JL[0].m,JL[1].m,JL[2].m],
          res:res}; }
function meanAbsRes(r){ let s=0,n2=0;
  for(let i=E0;i<T-E0;i++){ const x=r.res.f[i*3],y=r.res.f[i*3+1],
    z=r.res.f[i*3+2]; s+=Math.sqrt(x*x+y*y+z*z); n2++; }
  return s/n2; }
function sumSqRes(r){ let s=0;
  for(let i=E0;i<T-E0;i++){ const x=r.res.f[i*3],y=r.res.f[i*3+1],
    z=r.res.f[i*3+2]; s+=x*x+y*y+z*z; }
  return s/1e6; }
// plotting ----------------------------------------------------------
const COL=["#2a78d6","#eb6834","#1baf7a"], NEU="#52514e", FAINT="#b9b8b2";
const JN=["ankle","knee","hip"];
function niceTicks(lo,hi,n){ const span=hi-lo, raw=span/n,
  mag=Math.pow(10,Math.floor(Math.log10(raw))),
  step=[1,2,5,10].map(m=>m*mag).find(s=>span/s<=n+1)||10*mag;
  const t=[]; for(let x=Math.ceil(lo/step)*step; x<=hi+1e-9; x+=step)
    t.push(+x.toFixed(6)); return t; }
function panel(ctx,x0,y0,w,h,xlo,xhi,lo,hi,title,yunit,xunit){
  ctx.fillStyle="#fff"; ctx.fillRect(x0,y0,w,h);
  ctx.strokeStyle=FAINT; ctx.strokeRect(x0,y0,w,h);
  ctx.fillStyle=NEU; ctx.font="10px IBM Plex Sans";
  for(const ty of niceTicks(lo,hi,4)){
    const y=y0+h*(hi-ty)/(hi-lo);
    ctx.strokeStyle="#eceae4"; ctx.beginPath();
    ctx.moveTo(x0,y); ctx.lineTo(x0+w,y); ctx.stroke();
    ctx.textAlign="right"; ctx.fillText(ty, x0-3, y+3); }
  for(const tx of niceTicks(xlo,xhi,5)){
    const x=x0+w*(tx-xlo)/(xhi-xlo);
    ctx.textAlign="center"; ctx.fillText(tx, x, y0+h+11); }
  if(lo<0&&hi>0){ const zy=y0+h*hi/(hi-lo);
    ctx.strokeStyle=FAINT; ctx.beginPath();
    ctx.moveTo(x0,zy); ctx.lineTo(x0+w,zy); ctx.stroke(); }
  ctx.textAlign="left"; ctx.fillStyle=NEU;
  ctx.font="600 10.5px IBM Plex Sans";
  ctx.fillText(title, x0+5, y0+12);
  ctx.font="10px IBM Plex Sans";
  if(yunit) ctx.fillText(yunit, x0+5, y0+24);
  if(xunit){ ctx.textAlign="right"; ctx.fillText(xunit, x0+w, y0+h+22); } }
function line(ctx,x0,y0,w,h,lo,hi,ys,stride,off,color,width,dash){
  ctx.strokeStyle=color; ctx.lineWidth=width;
  if(dash) ctx.setLineDash(dash);
  ctx.beginPath();
  for(let i=0;i<T;i++){
    const x=x0+w*i/(T-1), y=y0+h*(hi-ys[i*stride+off])/(hi-lo);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }
  ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth=1; }
function prep(id,hcss){ const cv=document.getElementById(id);
  cv.style.height=hcss+"px";
  cv.width=cv.clientWidth*2; cv.height=hcss*2;
  const ctx=cv.getContext("2d"); ctx.scale(2,2);
  return [ctx, cv.clientWidth, hcss]; }
const DUR=(T-1)/RATE;
function draw(r, thr, lagSweep, lagNow){
  // residual (with real axes)
  { const [g,W,H]=prep("cRes",200); const x0=40,y0=6,w=W-x0-8,h=H-28;
    let lo=1e9,hi=-1e9;
    for(let i=0;i<T;i++) for(let c=0;c<3;c++){
      lo=Math.min(lo,r.res.f[i*3+c]); hi=Math.max(hi,r.res.f[i*3+c]); }
    const pad=.06*(hi-lo); lo-=pad; hi+=pad;
    panel(g,x0,y0,w,h,0,DUR,lo,hi,"residual force at torso COM","N","time (s)");
    for(let c=0;c<3;c++) line(g,x0,y0,w,h,lo,hi,r.res.f,3,c,COL[c],1.2); }
  // 3 torques, R solid / L dashed / V3D grey
  const rms=[];
  for(let j=0;j<6;j++){ let s=0,n2=0;
    for(let i=E0;i<T-E0;i++){ const d=r.torq[j][i*3]-REF[j*T+i]; s+=d*d; n2++; }
    rms.push(Math.sqrt(s/n2)); }
  for(let jj=0;jj<3;jj++){
    const [g,W,H]=prep("cT"+jj,180); const x0=34,y0=6,w=W-x0-6,h=H-28;
    let lo=1e9,hi=-1e9;
    for(const j of [jj,jj+3]) for(let i=0;i<T;i++){
      lo=Math.min(lo,r.torq[j][i*3],REF[j*T+i]);
      hi=Math.max(hi,r.torq[j][i*3],REF[j*T+i]); }
    const pad=.06*(hi-lo); lo-=pad; hi+=pad;
    panel(g,x0,y0,w,h,0,DUR,lo,hi,JN[jj],"N m","s");
    line(g,x0,y0,w,h,lo,hi,REF.subarray(jj*T,(jj+1)*T),1,0,FAINT,2.2);
    line(g,x0,y0,w,h,lo,hi,REF.subarray((jj+3)*T,(jj+4)*T),1,0,FAINT,2.2);
    line(g,x0,y0,w,h,lo,hi,r.torq[jj],3,0,COL[jj],1.1);
    line(g,x0,y0,w,h,lo,hi,r.torq[jj+3],3,0,COL[jj],1.1,[4,3]); }
  // lag sweep
  { const [g,W,H]=prep("cLag",190); const x0=36,y0=6,w=W-x0-6,h=H-30;
    let hi=-1e9,lo=1e9;
    for(const p of lagSweep){ hi=Math.max(hi,p[1]); lo=Math.min(lo,p[1]); }
    lo=0; hi*=1.08;
    panel(g,x0,y0,w,h,-50,50,lo,hi,"Σ|F_res|² vs τ","10⁶ N²","τ (ms)");
    g.strokeStyle=COL[0]; g.lineWidth=1.4; g.beginPath();
    lagSweep.forEach((p,i)=>{ const x=x0+w*(p[0]+50)/100,
      y=y0+h*(hi-p[1])/(hi-lo);
      if(i===0) g.moveTo(x,y); else g.lineTo(x,y); });
    g.stroke(); g.lineWidth=1;
    let best=lagSweep[0];
    for(const p of lagSweep) if(p[1]<best[1]) best=p;
    const bx=x0+w*(best[0]+50)/100, by=y0+h*(hi-best[1])/(hi-lo);
    g.fillStyle=NEU; g.beginPath(); g.arc(bx,by,2.5,0,7); g.fill();
    g.font="10px IBM Plex Sans"; g.textAlign="center";
    g.fillText("min "+best[0]+" ms", bx, by-8);
    const cur=lagSweep.reduce((a,p)=>Math.abs(p[0]-lagNow)<Math.abs(a[0]-lagNow)?p:a);
    const cx=x0+w*(lagNow+50)/100, cy=y0+h*(hi-cur[1])/(hi-lo);
    g.fillStyle="#eb6834"; g.beginPath(); g.arc(cx,cy,3.5,0,7); g.fill(); }
  // GRF (reference)
  { const [g,W,H]=prep("cGrf",150); const x0=40,y0=6,w=W-x0-8,h=H-28;
    let hi=0; for(let p=0;p<2;p++) for(let i=0;i<T;i++)
      hi=Math.max(hi,r.belts[p].F[i*3+2]);
    panel(g,x0,y0,w,h,0,DUR,0,hi*1.05,"vertical force per belt","N","time (s)");
    for(let p=0;p<2;p++){
      g.globalAlpha=.10; g.fillStyle=p?COL[0]:COL[2];
      for(let i=0;i<T;i++) if(r.belts[p].F[i*3+2]>thr)
        g.fillRect(x0+w*i/(T-1),y0,1,h);
      g.globalAlpha=1;
      line(g,x0,y0,w,h,0,hi*1.05,r.belts[p].F,3,2,p?COL[0]:COL[2],1.2); }
    g.strokeStyle=NEU; g.setLineDash([4,3]);
    const ty=y0+h*(hi*1.05-thr)/(hi*1.05);
    g.beginPath(); g.moveTo(x0,ty); g.lineTo(x0+w,ty); g.stroke();
    g.setLineDash([]); }
  return rms; }
function chips(rms, r, lagSweep){
  let pk=0; for(let i=E0;i<T-E0;i++)
    for(let c=0;c<3;c++) pk=Math.max(pk,Math.abs(r.res.f[i*3+c]));
  let best=lagSweep[0];
  for(const p of lagSweep) if(p[1]<best[1]) best=p;
  const items=[
    ["mean |F residual|", meanAbsRes(r).toFixed(0)+" <span class='u'>N</span>"],
    ["residual peak |F|", pk.toFixed(0)+" <span class='u'>N</span>"],
    ["median torque RMS vs V3D",
     (rms.slice().sort((a,b)=>a-b)[3]).toFixed(1)+" <span class='u'>N m</span>"],
    ["residual-minimizing τ", best[0]+" <span class='u'>ms</span>"]];
  document.getElementById("chips").innerHTML = items.map(
    it=>"<div class='chip'><div class='k'>"+it[0]+
        "</div><div class='v'>"+it[1]+"</div></div>").join(""); }
let SWEEP=null, SWEEP_KEY="";
function sweep(kIdx, fcut){
  const key=kIdx+"_"+fcut;
  if(SWEEP_KEY===key) return SWEEP;
  SWEEP=[];
  for(let lag=-50;lag<=50;lag+=2)
    SWEEP.push([lag, sumSqRes(run(lag,kIdx,fcut))]);
  SWEEP_KEY=key;
  return SWEEP; }
function recompute(){
  const lag=+document.getElementById("lag").value;
  const kv=+document.getElementById("kcut").value;
  const kIdx={8:0,12:1,20:2}[kv];
  const fcut=+document.getElementById("fcut").value;
  const thr=+document.getElementById("thr").value;
  document.getElementById("lagv").textContent=lag;
  document.getElementById("thrv").textContent=thr;
  document.getElementById("c_lag").textContent=lag;
  document.getElementById("c_k").textContent=kv;
  document.getElementById("c_f").textContent=fcut;
  for(const el of document.querySelectorAll(".c_thr")) el.textContent=thr;
  const sw=sweep(kIdx,fcut);
  const r=run(lag,kIdx,fcut);
  const rms=draw(r,thr,sw,lag);
  chips(rms,r,sw); }
for(const id of ["lag","kcut","fcut","thr"])
  document.getElementById(id).addEventListener("change",recompute);
window.addEventListener("resize",recompute);
recompute();
</script>
"""


def p1_playground_html(trial) -> str:
    """The interactive quickstart page: `playground_data` + the JS template."""
    import json
    return PLAYGROUND_TEMPLATE.replace("__DATA__",
                                       json.dumps(playground_data(trial)))


def main_playground(argv=None):
    """Compatibility alias for the canonical notebook build."""
    if argv:
        raise ValueError("Set study inputs in notebooks/p1_playground.ipynb")
    from .notebook_reports import main as build_reports
    return build_reports(["p1_playground"])

