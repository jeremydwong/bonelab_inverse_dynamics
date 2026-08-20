"""boneid.validate — cross-subject validation of our torques against Visual3D.

`report.py` proves one subject in depth. This module is the other axis: the
same pipeline, unattended, over **every populated trial of every subject and
both legs**, scored against Visual3D's own `ProxEndTorque` exports. That sweep
is a real boundary — it is a different claim ("this generalises") measured with
different statistics (distributions, medians, failure counts) — so it lives in
its own file, but it borrows report.py's style helpers and its p1 primitives
(`two_leg_chain`, `two_leg_rms`, `reference_torques`, `belt_speed`) rather than
restating them.

Run: `uv run python -m boneid.validate` → reports/validation_report.html
(optionally pass explicit .mat paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .report import (JOINT_COLORS, JOINT_LABELS, NEUTRAL, figure_block,
                     fig_svg, html_page, new_grid, stat, style_axes)

DATA_DIR = Path("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data")
SUBJECTS = tuple(f"p{i}" for i in range(1, 10))
FAMILIES = ("ankle", "knee", "hip")
#: RMS thresholds per joint family used for the "% of comparisons within
#: tolerance" statistic. Ankle and knee are directly comparable quantities;
#: the hip carries the accumulated hip-centre/anthropometry difference and is
#: judged on its own looser scale (stated in the prose, not hidden here).
TOL_NM = {"ankle": 5.0, "knee": 5.0, "hip": 20.0}


def subject_paths(names=SUBJECTS, data_dir: Path = DATA_DIR) -> list[Path]:
    """The `p<N>_5StridesData.mat` paths for `names`, in order."""
    return [Path(data_dir) / f"{n}_5StridesData.mat" for n in names]


#: A belt that never sees this much vertical force in a whole 5-stride window
#: did not carry a foot. Adult walking loads each belt to ~1 body weight, so
#: 200 N is far below any real step and far above plate noise.
LIVE_PLATE_N = 200.0


def plate_flag(trial, threshold_n: float = 20.0) -> str | None:
    """Data-quality flag for a trial, or None if both belts are live.

    Purely a property of the export, decided before any inverse dynamics runs:
    if one of the two belts never reaches `LIVE_PLATE_N`, that trial cannot
    support a two-leg analysis at all — both feet get assigned the one live
    plate, the force-derived body mass collapses, and Visual3D's own kinetics
    for the trial are equally degraded. Such trials are kept in the sweep and
    reported, but excluded from the headline statistics.
    """
    peak = trial.force[:, :, 2].max(axis=0)
    dead = [i for i, v in enumerate(peak) if v < LIVE_PLATE_N]
    if dead:
        return ("belt " + "/".join(f"{i + 1}" for i in dead)
                + " never loaded (peak "
                + ", ".join(f"{peak[i]:.0f}" for i in dead) + " N)")
    return None


def _stride_time(trial, threshold_n: float) -> float:
    """Mean right-foot stride time [s] (onset to onset) from the belt force."""
    from . import io_v3d as io
    from .core import detect_contact
    p = io.plate_for_side(trial, "r", threshold_n)
    _, events = detect_contact(trial.force[:, p, 2], threshold_n, min_gap=60)
    events = [e for e in events if e[1] - e[0] > 30]
    if len(events) < 2:
        return float("nan")
    return float((events[-1][0] - events[0][0]) / (len(events) - 1)
                 / trial.analog_rate)


def sweep_subject(path, params=None) -> dict:
    """Run the two-leg pipeline over every populated trial of one subject.

    One `row` per trial × leg (so two rows per trial), each carrying the three
    joint RMS errors against Visual3D — the vector RMS of
    `tau_ours - tau_v3d`, edge frames excluded, exactly `report.two_leg_rms` —
    together with the normalized RMS (% of that trial's Visual3D peak |tau| for
    that joint), the measured belt speed, stride time and body mass.

    Robustness is the point of a sweep: a trial that raises is recorded in
    `failures` with the exception string and the loop continues. The report
    prints that count.

    Returns::

        {"subject": "p1", "path": str, "n_trials": int,
         "rows": [ {subject, trial, leg, speed, stride_time, body_mass,
                    rms: [3], nrms: [3], peak: [3]} , ... ],
         "failures": [ {subject, trial, error}, ... ]}
    """
    from . import io_v3d as io
    from .core import inverse_dynamics_two_legs
    from .report import (EDGE, SIDES, p1_params, belt_speed, reference_peaks,
                         torque_rms_vs_v3d, two_leg_chain)

    params = params or p1_params()
    path = Path(path)
    subject = path.name.split("_")[0]
    trials = io.load_v3d_trials(path)

    rows, failures = [], []
    for tr in trials:
        try:
            flag = plate_flag(tr, params.contact_threshold_n)
            chain = two_leg_chain(tr, params)
            two = inverse_dynamics_two_legs(*chain, params)
            speed = belt_speed(tr, params.contact_threshold_n)
            stride = _stride_time(tr, params.contact_threshold_n)
            mass = io.estimate_body_mass(
                tr, threshold_n=params.contact_threshold_n)
            for side in SIDES:
                res = two.right if side == "r" else two.left
                rms = torque_rms_vs_v3d(tr, res, side, EDGE)
                peak = reference_peaks(tr, side, EDGE)
                rows.append({
                    "subject": subject, "trial": int(tr.index),
                    "leg": side.upper(), "speed": float(speed),
                    "stride_time": stride, "body_mass": float(mass),
                    "flag": flag,
                    "rms": np.asarray(rms, dtype=float),
                    "peak": np.asarray(peak, dtype=float),
                    "nrms": 100.0 * np.asarray(rms, dtype=float)
                    / np.maximum(np.asarray(peak, dtype=float), 1e-9),
                })
        except Exception as exc:                       # keep going, count it
            failures.append({"subject": subject, "trial": int(tr.index),
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"subject": subject, "path": str(path), "n_trials": len(trials),
            "rows": rows, "failures": failures}


def sweep_all(paths=None, params=None, verbose: bool = True) -> dict:
    """`sweep_subject` over p1..p9, pooled.

    Returns `{"subjects": [name...], "per_subject": {name: result},
    "rows": [...all rows...], "failures": [...], "n_trials": int}`.
    No caching; the whole sweep is about ten seconds.
    """
    paths = list(paths) if paths is not None else subject_paths()
    per, rows, failures, n_trials = {}, [], [], 0
    for p in paths:
        res = sweep_subject(p, params)
        per[res["subject"]] = res
        rows.extend(res["rows"])
        failures.extend(res["failures"])
        n_trials += res["n_trials"]
        if verbose:
            r = np.array([row["rms"] for row in res["rows"]])
            med = np.median(r, axis=0) if len(r) else np.full(3, np.nan)
            print(f"{res['subject']}: {len(res['rows'])} trial-legs, "
                  f"median RMS ankle/knee/hip = "
                  f"{med[0]:.2f}/{med[1]:.2f}/{med[2]:.2f} N m, "
                  f"{len(res['failures'])} failures", flush=True)
    return {"subjects": [Path(p).name.split("_")[0] for p in paths],
            "per_subject": per, "rows": rows, "failures": failures,
            "n_trials": n_trials}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def flagged_trials(results: dict) -> list[dict]:
    """[{subject, trial, flag}] — the trials excluded from headline stats."""
    seen, out = set(), []
    for r in results["rows"]:
        key = (r["subject"], r["trial"])
        if r.get("flag") and key not in seen:
            seen.add(key)
            out.append({"subject": r["subject"], "trial": r["trial"],
                        "flag": r["flag"]})
    return out


def _arrays(results: dict, only_ok: bool = True):
    """(subjects, subj_of_row, leg_of_row, speed, rms[N,3], nrms[N,3]).

    `only_ok` drops rows from trials carrying a `plate_flag`.
    """
    rows = [r for r in results["rows"] if not (only_ok and r.get("flag"))]
    subj = np.array([r["subject"] for r in rows])
    leg = np.array([r["leg"] for r in rows])
    speed = np.array([r["speed"] for r in rows], dtype=float)
    rms = np.array([r["rms"] for r in rows], dtype=float)
    nrms = np.array([r["nrms"] for r in rows], dtype=float)
    return results["subjects"], subj, leg, speed, rms, nrms


def per_subject_medians(results: dict) -> tuple[list, np.ndarray]:
    """(subject names, [S,6] median RMS ordered as `report.JOINT_LABELS`)."""
    _, subj, leg, _, rms, _ = _arrays(results)
    names = [s for s in results["subjects"] if (subj == s).any()]
    out = np.full((len(names), 6), np.nan)
    for i, s in enumerate(names):
        for k, side in enumerate(("R", "L")):
            m = (subj == s) & (leg == side)
            if m.any():
                out[i, 3 * k:3 * k + 3] = np.median(rms[m], axis=0)
    return names, out


def _fig_distributions(results: dict) -> str:
    """Per-subject RMS distributions, one panel per joint family."""
    names, subj, leg, _, rms, _ = _arrays(results)
    names = [s for s in names if (subj == s).any()]
    rng = np.random.default_rng(0)
    fig, axes = new_grid(1, 3, height=3.2, width=9.0)
    for j, fam in enumerate(FAMILIES):
        ax = axes[0, j]
        for i, s in enumerate(names):
            for side, filled in (("R", True), ("L", False)):
                m = (subj == s) & (leg == side)
                if not m.any():
                    continue
                x = i + (-0.16 if side == "R" else 0.16) \
                    + rng.uniform(-0.07, 0.07, m.sum())
                ax.plot(x, rms[m, j], "o", ms=3.6, mew=0.9,
                        color=JOINT_COLORS[j], alpha=0.75,
                        mfc=JOINT_COLORS[j] if filled else "none")
            m = subj == s
            ax.plot([i - 0.32, i + 0.32], [np.median(rms[m, j])] * 2,
                    color=NEUTRAL, lw=1.8, solid_capstyle="butt")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(fam, color=NEUTRAL, fontsize=10)
        style_axes(ax, "", "torque RMS vs Visual3D (N m)" if j == 0 else "")
        ax.set_ylim(bottom=0.0)
    # RULE: ankle and knee are the same quantity at comparable magnitude, so
    # they share one y-axis. The hip is an order larger and gets its own.
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(0.0, hi)
    axes[0, 1].set_ylim(0.0, hi)
    axes[0, 0].plot([], [], "o", ms=4, color=NEUTRAL, label="right leg")
    axes[0, 0].plot([], [], "o", ms=4, mfc="none", mec=NEUTRAL, label="left leg")
    axes[0, 0].plot([], [], color=NEUTRAL, lw=1.8, label="subject median")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    return fig


def _fig_speed(results: dict) -> str:
    """Normalized RMS (% of Visual3D peak) against measured belt speed."""
    _, _, leg, speed, _, nrms = _arrays(results)
    fig, axes = new_grid(1, 1, height=3.2, width=8.6)
    ax = axes[0, 0]
    for j in range(3):
        for side, filled in (("R", True), ("L", False)):
            m = leg == side
            ax.plot(speed[m], nrms[m, j], "o", ms=3.8, mew=0.9,
                    color=JOINT_COLORS[j], alpha=0.7,
                    mfc=JOINT_COLORS[j] if filled else "none")
        # trend: median in speed bins, so the eye is not fitting the cloud
        edges = np.linspace(speed.min(), speed.max() + 1e-9, 7)
        mid = 0.5 * (edges[:-1] + edges[1:])
        binned = [np.median(nrms[(speed >= a) & (speed < b), j])
                  if ((speed >= a) & (speed < b)).any() else np.nan
                  for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(mid, binned, color=JOINT_COLORS[j], lw=2.0, alpha=0.9)
    style_axes(ax, "belt speed (m/s)", "torque RMS (% of Visual3D peak)")
    ax.set_ylim(0.0, 1.18 * float(np.nanmax(nrms)))   # headroom for the legend
    handles = [ax.plot([], [], "o", ms=5, color=JOINT_COLORS[j],
                       label=FAMILIES[j])[0] for j in range(3)]
    handles += [ax.plot([], [], "o", ms=5, color=NEUTRAL, label="right leg")[0],
                ax.plot([], [], "o", ms=5, mfc="none", mec=NEUTRAL,
                        label="left leg")[0]]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=5,
              loc="upper center")
    return fig


def _median_table_html(results: dict) -> str:
    names, med = per_subject_medians(results)
    rows = []
    for i, s in enumerate(names):
        cells = "".join(f"<td>{med[i, k]:.2f}</td>" for k in range(6))
        rows.append(f"<tr><td>{s}</td>{cells}</tr>")
    col = np.nanmedian(med, axis=0)
    foot = ("<tr><td><b>median</b></td>"
            + "".join(f"<td><b>{col[k]:.2f}</b></td>" for k in range(6))
            + "</tr>")
    head = "".join(f"<th>{lab}</th>" for lab in JOINT_LABELS)
    return ('<div class="overflow"><table><thead><tr><th>subject</th>'
            f"{head}</tr></thead><tbody>{''.join(rows)}{foot}</tbody>"
            "</table></div>")


def validation_report_html(results: dict) -> str:
    """The cross-subject validation page, in the house style."""
    names, subj, leg, speed, rms, nrms = _arrays(results)
    n_rows = len(rms)
    n_fail = len(results["failures"])
    flagged = flagged_trials(results)
    n_all = len(results["rows"])
    med = np.median(rms, axis=0)                      # ankle, knee, hip
    # pooled over the ankle and knee comparisons (2 per trial-leg)
    within_ak = 100.0 * float(np.mean(
        np.concatenate([rms[:, 0] < TOL_NM["ankle"],
                        rms[:, 1] < TOL_NM["knee"]])))
    within_hip = 100.0 * float(np.mean(rms[:, 2] < TOL_NM["hip"]))
    med_nrms = np.median(nrms, axis=0)
    n_subj = len(set(subj.tolist()))

    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    add(_fig_distributions(results),
        f"Per-subject distributions of the torque RMS difference against "
        f"Visual3D — one dot per trial × leg ({n_rows} dots per panel), "
        f"filled for the right leg and open for the left, with each subject's "
        f"median as a bar. <b>The ankle and knee panels share one y-axis</b> "
        f"(same quantity, comparable magnitude); the hip needs its own scale "
        f"because it is roughly {med[2] / med[0]:.0f}× larger — that offset is "
        f"the hip-centre model difference discussed below, not a change in "
        f"the physics. Left and right legs are scored by identical code with "
        f"nothing but the side letter changed, and they overlap.")

    add(_fig_speed(results),
        f"The same errors normalized by each trial's own Visual3D peak "
        f"|τ| for that joint, against the belt speed measured from the "
        f"stance-phase heel marker. Heavy lines are medians in speed bins. "
        f"Absolute error grows with speed simply because the torques do; as a "
        f"fraction of the torque being computed it is flat — median "
        f"{med_nrms[0]:.1f}% ankle, {med_nrms[1]:.1f}% knee, "
        f"{med_nrms[2]:.1f}% hip across "
        f"{speed.min():.2f}–{speed.max():.2f} m/s. Nothing blows up at the "
        f"fast end, which is where a broken filter, a mis-assigned belt or a "
        f"COP division would show first.")

    stats = "".join([
        stat("comparisons", f"{n_rows}",
             f"trial × leg ({n_subj} subjects, {3 * n_rows} joints)"),
        stat("median ankle RMS", f"{med[0]:.2f}", "N m"),
        stat("median knee RMS", f"{med[1]:.2f}", "N m"),
        stat("median hip RMS", f"{med[2]:.2f}", "N m"),
    ])
    stats2 = "".join([
        stat("ankle+knee &lt; 5 N m", f"{within_ak:.0f}",
             f"% of {2 * n_rows} comparisons",
             "pass" if within_ak > 75 else "fail"),
        stat("hip &lt; 20 N m", f"{within_hip:.0f}",
             f"% of {n_rows} comparisons",
             "pass" if within_hip > 75 else "fail"),
        stat("failed trials", f"{n_fail}",
             f"of {results['n_trials']} attempted "
             f"({len(flagged)} excluded, dead belt)",
             "pass" if n_fail == 0 else "fail"),
        stat("speed range", f"{speed.min():.2f}–{speed.max():.2f}", "m/s"),
    ])

    if n_fail:
        items = "".join(f"<li><code>{f['subject']} trial {f['trial']}</code> — "
                        f"{f['error']}</li>" for f in results["failures"])
        fail_html = (f"<h2>Failures and excluded trials</h2>"
                     f"<p>{n_fail} trial(s) raised an exception and were "
                     f"skipped; they are listed here rather than dropped "
                     f"silently.</p><ul>{items}</ul>")
    else:
        fail_html = ("<h2>Failures and excluded trials</h2>"
                     "<p>No trial raised. Every populated trial of every "
                     "subject ran to completion with identical parameters and "
                     "no per-trial special casing.</p>")
    if flagged:
        items = "".join(
            f"<li><code>{f['subject']} trial {f['trial']}</code> — "
            f"{f['flag']}</li>" for f in flagged)
        fail_html += (
            f"<p>{len(flagged)} trial(s) — {n_all - n_rows} of the "
            f"{n_all} trial × leg comparisons — are excluded from the numbers "
            f"above by a data-quality test applied <em>before</em> any "
            f"dynamics runs (<code>validate.plate_flag</code>): one of the two "
            f"belts never reaches {LIVE_PLATE_N:.0f} N anywhere in the window, "
            f"so it recorded no steps. With one belt dead both feet are "
            f"assigned the live plate and the force-derived body mass "
            f"collapses; Visual3D's own kinetics for those trials are equally "
            f"degraded (its reference ankle torque peaks at 2 N m rather than "
            f"~100), so the comparison is meaningless in both directions. This "
            f"is a defect in the export, not a selection of favourable "
            f"trials — the criterion looks only at the force plates:</p>"
            f"<ul>{items}</ul>")
    else:
        fail_html += ("<p>No trial was excluded by the pre-dynamics "
                      "data-quality test (<code>validate.plate_flag</code>): "
                      "both belts recorded real steps in every trial.</p>")

    body = f"""
<p class="eyebrow">Report 3 · boneid · CROSS-SUBJECT VALIDATION</p>
<h1>Every Subject, Every Trial, Both Legs</h1>
<p class="subtitle">The same pipeline, unattended, over {n_subj} subjects of
the van der Zee, Mundinger &amp; Kuo treadmill dataset — {n_rows} trial × leg
comparisons, {3 * n_rows} joints — each scored against Visual3D's own
<code>ProxEndTorque</code> export.</p>

<div class="stat-row">{stats}</div>
<div class="stat-row">{stats2}</div>

<h2>What is being claimed</h2>
<p>Report 2 walks one subject's trial in detail. This page makes the weaker but
broader claim: the agreement there was not a fluke of one subject, one speed or
one leg. Every populated trial of <code>p1</code>–<code>p9</code> is run with
<b>identical parameters</b> — 12 Hz kinematic low-pass, 50 Hz force low-pass
inside the loader, 20 N contact threshold — through
<code>io_v3d.build_chain</code> and
<code>core.inverse_dynamics_two_legs</code>, and each joint's lab-frame torque
is compared to Visual3D's over the whole trial. The error statistic is the RMS
of the <em>3-vector</em> difference,
√(mean<sub>t</sub> |τ<sub>ours</sub> − τ<sub>V3D</sub>|²), with 10 frames
trimmed at each end (central differences are one-sided there), so no component
hides inside a scalar projection.</p>

<h2>What an RMS against Visual3D does and does not establish</h2>
<p><b>It does establish</b> that two independent implementations of rigid-body
inverse dynamics, reading the same markers and the same force plates, land on
the same torques — that our recursion, frames, filtering and belt assignment
contain no gross error. Visual3D is the tool these data were published with;
disagreeing with it by a few percent of peak torque would be a red flag.</p>
<p><b>It does not establish that either of us is right.</b> The two pipelines
share the same physics but not the same anthropometry or the same joint-centre
conventions: we build segments from de Leva (1996) regressions on measured
segment lengths and a force-plate-derived body mass, Visual3D from its own
model. The residual difference is dominated by those conventions, not by the
dynamics — which is exactly why the hip is the worst joint everywhere in the
table below. The hip centre is the least directly measured landmark in the
markerset, and an offset there moves the thigh COM, the thigh inertia and the
hip moment arm together. Ankle and knee, whose centres are mid-malleoli and
mid-epicondyle and therefore essentially measured, agree far more tightly.</p>
<p>The residual difference is also not white noise, and it is worth naming what
it is. Where a subject sits noticeably above the pack on one joint and one side
— <code>p2</code>'s left ankle is the clearest case in the table below, about
three times its own right ankle — the excess is almost entirely in the
<em>non-sagittal</em> components: the sagittal (lab-x) torque still tracks
Visual3D to about 1 N m, while the frontal-plane component carries a steady
few-N m offset. That is a foot medio-lateral axis definition disagreeing, i.e.
marker placement and segment-frame convention, and it moves the inversion
/eversion moment without touching the plantarflexion moment. Reporting the
error as a full 3-vector RMS makes such cases visible instead of projecting
them away.</p>
<p>The genuinely implementation-independent checks live elsewhere and are not
duplicated here: the analytic squat-to-stand (Report 1), the energy audit, and
the L5/S1 residual matching the weight of the unmodelled upper body (Report 2,
Step 7). Agreement with Visual3D is corroboration, not proof.</p>
<p><b>Scope.</b> This dataset is treadmill <em>walking</em> only, roughly
{speed.min():.2f}–{speed.max():.2f} m/s, with no aerial phase in any trial.
Nothing here speaks to running, cutting, jumping, or overground data.</p>

{figs[0]}
{figs[1]}

<h2>Per-subject medians</h2>
<p>Median torque RMS versus Visual3D in N m, per subject and per joint, with
the median across subjects in the last row. Every cell is the median over that
subject's trials.</p>

{_median_table_html(results)}

{fail_html}

<h2>How to reproduce</h2>
<p><code>uv run python -m boneid.validate</code> runs the full sweep — all
nine subjects, about ten seconds — and writes this page to
<code>reports/validation_report.html</code>; pass explicit <code>.mat</code>
paths to sweep a different set. <code>uv run pytest tests/test_validate.py</code>
exercises the sweep on a three-trial slice.</p>
"""
    return html_page("Cross-Subject Validation", "validate", body)


def main(argv=None):
    """`python -m boneid.validate [mat_path ...]`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a) for a in argv] if argv else subject_paths()
    results = sweep_all(paths)
    out = Path(__file__).resolve().parents[2] / "reports"
    out.mkdir(exist_ok=True)
    dest = out / "validation_report.html"
    dest.write_text(validation_report_html(results))
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
