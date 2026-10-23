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
        fail_html = (f"<p>{n_fail} trial(s) raised an exception and were "
                     f"skipped; they are listed here rather than dropped "
                     f"silently.</p><ul>{items}</ul>")
    else:
        fail_html = ("<p>No trial raised. Every populated trial of every "
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
<p class="eyebrow">boneid · {n_subj} walkers vs Visual3D</p>
<p class="subtitle">The same pipeline, left alone, over {n_subj} subjects of
the van der Zee, Mundinger &amp; Kuo treadmill dataset: {n_rows} trial × leg
comparisons, {3 * n_rows} joints, each scored against Visual3D's own
<code>ProxEndTorque</code> export.</p>

<div class="stat-row">{stats}</div>
<div class="stat-row">{stats2}</div>

<h2>1. One set of parameters, every trial, both legs</h2>
<p>The <a href="p1_report.html">P1 report</a> goes through one subject in
detail. This page makes a weaker but broader claim: the agreement there wasn't
a fluke of one subject, one speed or one leg. I run every populated trial of
<code>p1</code>–<code>p9</code> with <b>identical parameters</b> (12 Hz
kinematic low-pass, 50 Hz force low-pass inside the loader, 20 N contact
threshold) through <code>io_v3d.build_chain</code> and
<code>core.inverse_dynamics_two_legs</code>, and compare each joint's
lab-frame torque to Visual3D's over the whole trial. The error is the RMS of
the <em>3-vector</em> difference,
√(mean<sub>t</sub> |τ<sub>ours</sub> − τ<sub>V3D</sub>|²), with 10 frames
trimmed at each end (central differences are one-sided there). That way no
component gets to hide inside a scalar projection.</p>

<h2>2. How big the error is, subject by subject</h2>
{figs[0]}

<h2>3. Error as a fraction of the torque, against speed</h2>
{figs[1]}

<h2>4. Why the hip is worst, and the one outlier</h2>
<p>Here's the table behind Figure 1: median torque RMS vs Visual3D in N m,
per subject and per joint (each cell is the median over that subject's
trials), with the median across subjects in the last row.</p>

{_median_table_html(results)}

<p>The two pipelines share the same physics but not the same anthropometry or
the same joint-centre conventions. We build segments from de Leva (1996)
regressions on measured segment lengths and a force-plate-derived body mass;
Visual3D uses its own model. I think the leftover difference is mostly those
conventions, not the dynamics, and that's why the hip is the worst joint for
every subject. The hip centre is the least directly measured landmark in the
markerset, and moving it shifts the thigh COM, the thigh inertia and the hip
moment arm all at once. Ankle and knee centres are mid-malleoli and
mid-epicondyle (so basically measured), and they agree much more tightly.</p>
<p>The leftover isn't white noise either. The clearest outlier in the table is
<code>p2</code>'s left ankle, about three times its own right ankle. Nearly
all of that excess is in the frontal-plane (lab-y) component; the sagittal
(lab-x) part is only a bit worse than the right leg's. That looks like the two
tools disagreeing on the foot's medio-lateral axis (marker placement and
segment-frame convention), which moves the inversion/eversion moment without
touching plantarflexion. Scoring the full 3-vector is what makes a case like
this show up instead of getting projected away.</p>

<h2>5. Failed and excluded trials</h2>
{fail_html}

<h2>What this does and doesn't prove</h2>
<p>It does show that two independent implementations of rigid-body inverse
dynamics, reading the same markers and force plates, land on the same torques,
so our recursion, frames, filtering and belt assignment have no gross error.
(Visual3D is what these data were published with; missing it by a lot would be
a red flag.) It doesn't show that either of us is right. The checks that
don't depend on anyone's implementation are elsewhere: the
<a href="tutorial_s2s_markers.html">squat-to-stand tutorial</a>, the energy
audit, and the L5/S1 residual matching the weight of the unmodelled upper body
(<a href="p1_report.html">P1 report</a>). Agreement with Visual3D is
corroboration, not proof. Also, this is treadmill <em>walking</em> only,
{speed.min():.2f}–{speed.max():.2f} m/s, no aerial phase in any trial; for
running see the <a href="rbds_report.html">RBDS report</a>.</p>
"""
    return html_page("Cross-Subject Validation", "validate", body)


def main(argv=None):
    """`python -m boneid.validate [rbds [n] | mat_path ...]`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "rbds":
        return main_rbds(argv[1:])
    if argv and argv[0] == "rbds-ablation":
        return main_rbds_ablation(argv[1:])
    if argv and argv[0] == "rbds-cumulative":
        return main_rbds_cumulative(argv[1:])
    if argv:
        raise ValueError("Set input paths in notebooks/validation_report.ipynb")
    from .notebook_reports import main as build_reports
    return build_reports(["validation_report"])





# ---------------------------------------------------------------------------
# RBDS: their raw data through OUR pipeline (Fukuchi et al. 2017, CC-BY)
# ---------------------------------------------------------------------------
#
# The RBDS public running dataset ships RAW c3d (markers + treadmill forces +
# a static calibration trial per subject) alongside its own Visual3D-processed
# output — which makes it a full-pipeline validation set: run the raw data
# through io_c3d's RBDS loader (static-calibrated CAST, per-bout foot
# attribution, Newton-Euler) and score our right-leg sagittal torques and
# powers against theirs, per subject, per speed.  See io_c3d.py's RBDS
# section for the data layout and frame conventions.

RBDS_ARTICLE = "https://api.figshare.com/v2/articles/4543435"
RBDS_CACHE = Path.home() / ".cache" / "boneid" / "rbds"
RBDS_SPEEDS = ("25", "35", "45")

#: our proximal-frame local-y (ML) torque vs the dataset's clinical MomZ
#: differ by a fixed per-joint sign (measured on RBDS001, |r| > 0.98):
RBDS_TORQUE_SIGN = np.array([1.0, -1.0, -1.0])   # ankle, knee, hip


def rbds_file_index(cache: Path = RBDS_CACHE) -> dict:
    """name -> download_url for every file in the figshare record (cached)."""
    import json
    import urllib.request
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = cache / "figshare.json"
    if not meta.exists():
        with urllib.request.urlopen(RBDS_ARTICLE, timeout=60) as r:
            meta.write_bytes(r.read())
    d = json.loads(meta.read_text())
    return {f["name"]: f["download_url"] for f in d["files"]}


def rbds_fetch(name: str, cache: Path = RBDS_CACHE) -> Path:
    """Download one dataset file into the cache (skips if present)."""
    import urllib.request
    cache = Path(cache)
    dest = cache / name
    if not dest.exists() or dest.stat().st_size == 0:
        url = rbds_file_index(cache)[name]
        with urllib.request.urlopen(url, timeout=300) as r:
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.read())
            tmp.rename(dest)
    return dest


def rbds_info(cache: Path = RBDS_CACHE) -> dict:
    """subject id -> {mass, sex, height} from RBDSinfo.txt (one row per
    file; rows repeat per subject, first wins)."""
    path = rbds_fetch("RBDSinfo.txt", cache)
    out = {}
    with open(path, encoding="latin-1") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        i_sub = hdr.index("Subject")
        i_mass = hdr.index("Mass")
        i_sex = hdr.index("Gender")
        i_h = hdr.index("Height")
        for line in fh:
            row = line.rstrip("\n").split("\t")
            if len(row) <= i_h:
                continue
            sid = int(row[i_sub])
            if sid not in out:
                out[sid] = {"mass": float(row[i_mass]),
                            "sex": ("male" if row[i_sex].strip().upper()
                                    .startswith("M") else "female"),
                            "height": float(row[i_h])}
    return out


def rbds_reference_curves(processed_path, speed: str):
    """Their stance-normalized right-leg curves from RBDS*processed.txt.

    Returns (torque [101,3] N m/kg, power [101,3] W/kg, stance_frac) with
    joints ordered (ankle, knee, hip), or None when the file lacks that
    speed.  Stance runs from the cycle start (right foot strike) to the
    first sample where their vertical GRF (Y, NaN in swing) drops below
    5% BW."""
    raw = np.genfromtxt(processed_path, names=True, delimiter="\t")
    if f"RgrfY{speed}" not in (raw.dtype.names or ()):
        return None
    fz = raw[f"RgrfY{speed}"]
    on = np.isfinite(fz) & (fz > 0.05)
    if not on[0] or on.sum() < 20:
        return None
    toe = int(np.flatnonzero(~on)[0])
    grid = np.linspace(0.0, 1.0, 101)
    x = np.linspace(0.0, 1.0, toe)
    tq = np.stack([np.interp(grid, x, raw[f"R{j}MomZ{speed}"][:toe])
                   for j in ("ankle", "knee", "hip")], axis=1)
    pw = np.stack([np.interp(grid, x, raw[f"R{j}Pow{speed}"][:toe])
                   for j in ("ankle", "knee", "hip")], axis=1)
    if not (np.isfinite(tq).all() and np.isfinite(pw).all()):
        return None
    return tq, pw, toe / 100.0


#: The ablation arms: each brings ONE of our conventions to the dataset's
#: own (decoded from RBDSmodelV3D.mdh / RBDSpipelineV3D.v3s), plus "all".
RBDS_VARIANTS = {
    "baseline": {},
    "hip_bell": {"hip": "bell"},          # their hip landmark (Bell 1990)
    "lp10": {"lowpass": 10.0},            # their 10 Hz marker+analog filter
    "dempster": {"segments": "dempster"},  # their (Dempster-based) segments
    "power_scalar": {"power": "scalar"},  # their JOINT_POWER_SCALAR (τ·ω)
    "all": {"hip": "bell", "lowpass": 10.0, "segments": "dempster",
            "power": "scalar"},
}


def rbds_process_subject(sid: int, cache: Path = RBDS_CACHE,
                         speeds=RBDS_SPEEDS,
                         variant: dict | None = None) -> dict:
    """Run one RBDS subject's raw c3d through our pipeline, all speeds.

    Downloads what it needs into `cache`.  Returns a JSON-able dict:
    per speed — our stance-normalized mean torque/power curves [101,3]
    (mass-normalized, our local-y sign), their reference curves, RMSE and
    correlation per joint (after the fixed RBDS_TORQUE_SIGN alignment),
    energy imbalance %, step count/stance seconds, max cluster residuals.

    `variant` (see RBDS_VARIANTS) swaps individual conventions for the
    dataset's own: hip='bell', lowpass=10.0, segments='dempster',
    power='scalar' (τ·ω_rel only, dropping the translational joint-force
    term our full wrench power carries).
    """
    from .core import (AnalysisParams, chain_kinematics, energy_audit,
                       inverse_dynamics)
    from .io_c3d import (Y_UP_AXES, build_rbds_chain, calibrate_rbds_static,
                         load_c3d)
    from .report import right_step_events, stance_profiles

    variant = variant or {}
    tag = f"RBDS{sid:03d}"
    info = rbds_info(cache)[sid]
    index = rbds_file_index(cache)
    # subjects 3-9's static c3d is named with TWO digits in the record
    static_name = (f"{tag}static.c3d" if f"{tag}static.c3d" in index
                   else f"RBDS{sid:02d}static.c3d")
    static = load_c3d(rbds_fetch(static_name, cache), axes=Y_UP_AXES)
    cal = calibrate_rbds_static(static, body_mass=info["mass"], side="r",
                                sex=info["sex"],
                                hip_method=variant.get("hip", "harrington"))
    processed = rbds_fetch(f"{tag}processed.txt", cache)
    lp = float(variant.get("lowpass", 12.0))
    params = AnalysisParams(contact_threshold_n=50.0, lowpass_hz=lp,
                            force_lowpass_hz=lp)
    out = {"sid": sid, "mass": info["mass"], "sex": info["sex"],
           "speeds": {}}
    grid = np.linspace(0.0, 1.0, 101)
    for sp in speeds:
        name = f"{tag}runT{sp}.c3d"
        if name not in index:
            name = f"RBDS{sid:02d}runT{sp}.c3d"   # subjects 3-9: two digits
        if name not in index:
            continue
        ref = rbds_reference_curves(processed, sp)
        if ref is None:
            continue
        trial = load_c3d(rbds_fetch(name, cache), axes=Y_UP_AXES)
        skel, kin, ground, q = build_rbds_chain(
            trial, cal, segments=variant.get("segments", "deleva"))
        idres = inverse_dynamics(skel, kin, ground, params)
        audit = energy_audit(skel, kin, ground, idres, params)
        events = right_step_events(ground, params.contact_threshold_n)
        if len(events) < 5:
            out["speeds"][sp] = {"error": f"only {len(events)} steps"}
            continue
        tq = np.stack([stance_profiles(
            idres.joint_torque_local[:, j, 1], events).mean(axis=0)
            for j in range(3)], axis=1) / info["mass"]
        if variant.get("power") == "scalar":
            # the dataset's JOINT_POWER_SCALAR: τ · ω_rel, rotational only
            ck = chain_kinematics(skel, kin, params.lowpass_hz,
                                  params.filter_order)
            power = np.stack(
                [np.sum(idres.joint_torque[:, j]
                        * (ck["omega"][:, j] - ck["omega"][:, j + 1]),
                        axis=1) for j in range(3)], axis=1)
        else:
            power = audit.joint_power[:, :3]
        pw = np.stack([stance_profiles(power[:, j], events).mean(axis=0)
                       for j in range(3)], axis=1) / info["mass"]
        tq_ref, pw_ref, stance_frac = ref
        tq_aligned = tq * RBDS_TORQUE_SIGN
        edge = slice(30, -30)
        imb = float(np.abs(audit.imbalance[edge]).max()
                    / max(np.abs(audit.de_dt[edge]).max(), 1e-9))
        out["speeds"][sp] = {
            "torque_ours": tq.tolist(), "torque_theirs": tq_ref.tolist(),
            "power_ours": pw.tolist(), "power_theirs": pw_ref.tolist(),
            "torque_rmse": [float(np.sqrt(np.mean(
                (tq_aligned[:, j] - tq_ref[:, j]) ** 2))) for j in range(3)],
            "torque_corr": [float(np.corrcoef(
                tq_aligned[:, j], tq_ref[:, j])[0, 1]) for j in range(3)],
            "power_rmse": [float(np.sqrt(np.mean(
                (pw[:, j] - pw_ref[:, j]) ** 2))) for j in range(3)],
            "power_corr": [float(np.corrcoef(
                pw[:, j], pw_ref[:, j])[0, 1]) for j in range(3)],
            "n_steps": len(events),
            "stance_s": float(np.mean([e - s for s, e in events])
                              / trial.rate),
            "imbalance_frac": imb,
            "cluster_residual_max_mm": (
                1000.0 * q["cluster_residual"].max(axis=0)).tolist(),
        }
    return out


def rbds_sweep(sids=None, cache: Path = RBDS_CACHE, workers: int = 6,
               verbose: bool = True, variant: dict | None = None) -> dict:
    """Fan the per-subject runs out over a process pool.

    `sids` defaults to every subject with a static c3d in the figshare
    record.  A subject that raises is recorded as {'error': ...} rather
    than sinking the sweep.  `variant` is passed through to
    `rbds_process_subject`."""
    import re
    from concurrent.futures import ProcessPoolExecutor, as_completed
    index = rbds_file_index(cache)
    if sids is None:
        sids = sorted({int(m.group(1))
                       for n in index
                       for m in [re.match(r"RBDS(\d+)static\.c3d$", n)]
                       if m})
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(rbds_process_subject, sid, cache,
                            RBDS_SPEEDS, variant): sid
                for sid in sids}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                results[sid] = fut.result()
            except Exception as exc:
                results[sid] = {"sid": sid, "error": repr(exc)}
            if verbose:
                r = results[sid]
                if "error" in r:
                    print(f"RBDS{sid:03d}: ERROR {r['error']}")
                else:
                    done = ", ".join(
                        f"{sp}: tq RMSE "
                        + "/".join(f"{v:.2f}" for v in d["torque_rmse"])
                        if "error" not in d else f"{sp}: {d['error']}"
                        for sp, d in sorted(r["speeds"].items()))
                    print(f"RBDS{sid:03d}: {done}", flush=True)
    return {"results": {str(k): v for k, v in sorted(results.items())}}


def rbds_ablation_summary(results: dict) -> dict:
    """Median torque/power RMSE per joint from one variant's sweep."""
    rows = [d for r in results["results"].values() if "error" not in r
            for d in r.get("speeds", {}).values() if "error" not in d]
    return {
        "n": len(rows),
        "torque_rmse": np.median([d["torque_rmse"] for d in rows],
                                 axis=0).tolist(),
        "power_rmse": np.median([d["power_rmse"] for d in rows],
                                axis=0).tolist(),
        "torque_corr": np.median([d["torque_corr"] for d in rows],
                                 axis=0).tolist(),
        "power_corr": np.median([d["power_corr"] for d in rows],
                                axis=0).tolist(),
    }


def main_rbds_ablation(argv=None):
    """`python -m boneid.validate rbds-ablation [n_subjects]` — re-run the
    sweep once per RBDS_VARIANTS arm and tabulate the medians."""
    import json
    argv = list(argv or [])
    sids = None
    if argv:
        import re
        index = rbds_file_index()
        all_sids = sorted({int(m.group(1)) for n in index
                           for m in [re.match(r"RBDS(\d+)static\.c3d$", n)]
                           if m})
        sids = all_sids[:int(argv[0])]
    out = {}
    for name, variant in RBDS_VARIANTS.items():
        print(f"=== variant {name}: {variant or 'ours as published'}")
        res = rbds_sweep(sids, variant=variant, verbose=False)
        out[name] = {"variant": variant,
                     "summary": rbds_ablation_summary(res)}
        s = out[name]["summary"]
        print(f"    torque RMSE {'/'.join(f'{v:.3f}' for v in s['torque_rmse'])}"
              f" N m/kg | power RMSE "
              f"{'/'.join(f'{v:.2f}' for v in s['power_rmse'])} W/kg "
              f"(n={s['n']})", flush=True)
    dest = Path(__file__).resolve().parents[2] / "reports" / "rbds_ablation.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


#: Cumulative adoption of their conventions, biggest one-at-a-time lever
#: first.  Step k applies every change up to and including its own.
RBDS_CUMULATIVE = [
    ("ours", {}),
    ("+ their power (τ·ω)", {"power": "scalar"}),
    ("+ their hip (Bell)", {"power": "scalar", "hip": "bell"}),
    ("+ their segments (Dempster)",
     {"power": "scalar", "hip": "bell", "segments": "dempster"}),
    ("+ their 10 Hz filters  = all",
     {"power": "scalar", "hip": "bell", "segments": "dempster",
      "lowpass": 10.0}),
]


def main_rbds_cumulative(argv=None):
    """`python -m boneid.validate rbds-cumulative [n]` — re-run the sweep at
    each CUMULATIVE step toward their conventions, keeping the ensemble mean
    curves (3.5 m/s, the speed every subject ran) so the convergence can be
    PLOTTED, not just tabulated."""
    import json
    argv = list(argv or [])
    sids = None
    if argv:
        import re
        index = rbds_file_index()
        all_sids = sorted({int(m.group(1)) for n in index
                           for m in [re.match(r"RBDS(\d+)static\.c3d$", n)]
                           if m})
        sids = all_sids[:int(argv[0])]
    out = {"steps": [], "theirs": None}
    for name, variant in RBDS_CUMULATIVE:
        res = rbds_sweep(sids, variant=variant, verbose=False)
        rows35 = [d for r in res["results"].values() if "error" not in r
                  for s, d in r.get("speeds", {}).items()
                  if s == "35" and "error" not in d]
        tq = np.mean([np.asarray(d["torque_ours"]) for d in rows35],
                     axis=0) * RBDS_TORQUE_SIGN
        pw = np.mean([np.asarray(d["power_ours"]) for d in rows35], axis=0)
        if out["theirs"] is None:
            out["theirs"] = {
                "torque": np.mean([np.asarray(d["torque_theirs"])
                                   for d in rows35], axis=0).tolist(),
                "power": np.mean([np.asarray(d["power_theirs"])
                                  for d in rows35], axis=0).tolist(),
            }
        s = rbds_ablation_summary(res)
        out["steps"].append({"name": name, "variant": variant,
                             "summary": s,
                             "torque_mean_35": tq.tolist(),
                             "power_mean_35": pw.tolist()})
        print(f"{name}: torque RMSE "
              f"{'/'.join(f'{v:.3f}' for v in s['torque_rmse'])} | power "
              f"{'/'.join(f'{v:.2f}' for v in s['power_rmse'])}", flush=True)
    dest = (Path(__file__).resolve().parents[2] / "reports"
            / "rbds_cumulative.json")
    dest.write_text(json.dumps(out))
    print(f"wrote {dest}")


def main_rbds(argv=None):
    """Compatibility alias for the notebook report build."""
    if argv:
        raise ValueError("Set subjects in notebooks/rbds_report.ipynb")
    from .notebook_reports import main as build_reports
    return build_reports(["rbds_report"])


def rbds_update_reference_stance(results: dict) -> None:
    """Write the sweep's measured mean stance DURATIONS (seconds, per speed)
    into data/rbds_running_reference.npz as ``stance_s_25``/``stance_s_35``,
    so the c3d report can draw the reference curves on a real time axis."""
    path = (Path(__file__).resolve().parents[2] / "data"
            / "rbds_running_reference.npz")
    if not path.exists():
        return
    z = dict(np.load(path, allow_pickle=False).items())
    for sp in ("25", "35"):
        vals = [d["stance_s"] for r in results["results"].values()
                if "error" not in r
                for s, d in r.get("speeds", {}).items()
                if s == sp and "error" not in d]
        if vals:
            z[f"stance_s_{sp}"] = np.array(float(np.mean(vals)))
    np.savez(path, **z)




def _rbds_cumulative_html(cum=None) -> str:
    """Figures for the CUMULATIVE convergence run (rbds_cumulative.json):
    RMSE vs adoption step, and the ensemble mean + error curves at every
    step — the visual evidence behind the ablation table.  Empty string when
    the json has not been produced yet."""
    import json
    from .report import FAINT, SURFACE, figure_block
    import matplotlib.pyplot as plt

    if cum is None:
        return ""
    steps = cum["steps"]
    n_steps = len(steps)
    joints = ("ankle", "knee", "hip")
    grid = np.linspace(0, 100, 101)
    theirs_tq = np.asarray(cum["theirs"]["torque"])
    theirs_pw = np.asarray(cum["theirs"]["power"])
    shades = plt.cm.viridis(np.linspace(0.85, 0.1, n_steps))
    blocks = []

    # --- convergence lines: median RMSE vs cumulative step
    fig, axes = new_grid(1, 2, height=3.4, width=9.2)
    x = np.arange(n_steps)
    for k, (key, unit) in enumerate((("torque_rmse", "N m/kg"),
                                     ("power_rmse", "W/kg"))):
        ax = axes[0, k]
        for j, joint in enumerate(joints):
            ax.plot(x, [s["summary"][key][j] for s in steps],
                    color=JOINT_COLORS[j], lw=1.6, marker="o", ms=4,
                    label=joint)
        ax.set_xticks(x)
        ax.set_xticklabels([s["name"] for s in steps], rotation=35,
                           ha="right", fontsize=8)
        ax.set_ylim(bottom=0.0)
        style_axes(ax, "", f"median {key.split('_')[0]} RMSE ({unit})")
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5)
    blocks.append((fig, (
        "Convergence as their conventions are adopted CUMULATIVELY, left to "
        "right (each step keeps every change before it; medians over all "
        "101 subject-speeds).  The torque panel is nearly flat — no "
        "convention choice explains the torque differences, which are the "
        "between-implementation floor.  The power panel drops exactly once, "
        "at the power-definition step.")))

    # --- ensemble mean curves + error curves at every step (3.5 m/s)
    fig, axes = new_grid(4, 3, height=8.6, width=9.2, sharex=True)
    for j, joint in enumerate(joints):
        for row, (theirs, key, ylab) in enumerate((
                (theirs_tq, "torque_mean_35", "torque (N m/kg)"),
                (theirs_pw, "power_mean_35", "power (W/kg)"))):
            ax_c = axes[2 * row, j]        # curves
            ax_e = axes[2 * row + 1, j]    # ours - theirs
            ax_c.plot(grid, theirs[:, j], color=NEUTRAL, lw=3.0, alpha=0.45,
                      label="theirs (V3D)" if j == 0 else None)
            for si, s in enumerate(steps):
                ours = np.asarray(s[key])[:, j]
                ax_c.plot(grid, ours, color=shades[si], lw=1.1,
                          label=s["name"] if j == 0 else None)
                ax_e.plot(grid, ours - theirs[:, j], color=shades[si],
                          lw=1.1)
            ax_e.axhline(0.0, color=FAINT, lw=0.9)
            if row == 0 and j <= 2:
                ax_c.set_title(joint, color=NEUTRAL, fontsize=10)
            style_axes(ax_c, "", ylab if j == 0 else "")
            style_axes(ax_e, "stance (%)" if row == 1 else "",
                       "ours − theirs" if j == 0 else "")
    axes[0, 0].legend(frameon=False, fontsize=7)
    blocks.append((fig, (
        "The same convergence as curves (3.5 m/s, the speed all 39 subjects "
        "ran; ensemble means).  Rows 1 &amp; 3: their Visual3D ensemble "
        "mean (thick grey) with ours at every cumulative step (light "
        "&rarr; dark as conventions are adopted).  Rows 2 &amp; 4: the "
        "error, ours &minus; theirs, same colours — where the steps "
        "actually move the curves.  For POWER the successive lines collapse "
        "toward zero at the power-definition step (hip most of all).  For "
        "TORQUE the error curves barely move at any step: the remaining "
        "ankle/knee shape difference (early-stance lead, slightly lower "
        "knee peak) is not explained by hip, segments, filtering or power "
        "conventions.")))

    out = ""
    for k, (fig, caption) in enumerate(blocks):
        out += figure_block(fig_svg(fig), 90 + k, caption).replace(
            f"<b>Figure {90 + k}.</b>", "<b>Convergence "
            f"{'A' if k == 0 else 'B'}.</b>")
    return out


def rbds_report_html(results: dict, *, ablations=None, cumulative=None) -> str:
    """The RBDS cross-validation report: their raw data, our pipeline,
    scored against their own Visual3D processing."""
    from .report import FAINT, figure_block

    rows = []            # (sid, speed, dict)
    for sid, r in results["results"].items():
        if "error" in r:
            continue
        for sp, d in r["speeds"].items():
            if "error" not in d:
                rows.append((int(sid), sp, d))
    speeds = sorted({sp for _, sp, _ in rows})
    joints = ("ankle", "knee", "hip")
    grid = np.linspace(0, 100, 101)
    figs = []
    n = [0]

    def add(fig, caption):
        n[0] += 1
        figs.append(figure_block(fig_svg(fig), n[0], caption))

    sign = np.array([1.0, -1.0, -1.0])   # io_c3d local-y -> their MomZ

    for kind, unit in (("torque", "N m/kg"), ("power", "W/kg")):
        fig, axes = new_grid(len(speeds), 3, height=2.2 * len(speeds),
                             width=9.2, sharex=True)
        for si, sp in enumerate(speeds):
            ours = np.mean([np.asarray(d[f"{kind}_ours"])
                            for _, s, d in rows if s == sp], axis=0)
            theirs = np.mean([np.asarray(d[f"{kind}_theirs"])
                              for _, s, d in rows if s == sp], axis=0)
            if kind == "torque":
                ours = ours * sign
            for j in range(3):
                ax = axes[si, j]
                ax.plot(grid, theirs[:, j], color=NEUTRAL, lw=2.6, alpha=0.45,
                        label="theirs (V3D)" if si == 0 and j == 0 else None)
                ax.plot(grid, ours[:, j], color=JOINT_COLORS[j], lw=1.4,
                        label="ours" if si == 0 and j == 0 else None)
                ax.axhline(0.0, color=FAINT, lw=0.7)
                if si == 0:
                    ax.set_title(joints[j], color=NEUTRAL, fontsize=10)
                style_axes(ax, "stance (%)" if si == len(speeds) - 1 else "",
                           f"{sp[0]}.{sp[1]} m/s\n{kind} ({unit})"
                           if j == 0 else "")
                if si == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=8)
        add(fig, f"Ensemble mean sagittal joint {kind} over stance: every "
                 f"subject's raw c3d through OUR pipeline (colour) vs the "
                 f"dataset's own Visual3D processing (grey), averaged over "
                 f"the same subjects, per speed.  {kind.capitalize()}s are "
                 f"mass-normalized; ours sign-aligned per joint "
                 f"(convention only).")

    # RMSE distributions
    fig, axes = new_grid(1, 2, height=3.0, width=9.2)
    for k, kind in enumerate(("torque", "power")):
        ax = axes[0, k]
        data, labels = [], []
        for j in range(3):
            for sp in speeds:
                data.append([d[f"{kind}_rmse"][j] for _, s, d in rows
                             if s == sp])
                labels.append(f"{joints[j]} {sp[0]}.{sp[1]}")
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(JOINT_COLORS[i // len(speeds)])
            box.set_alpha(0.4)
        ax.tick_params(axis="x", rotation=60)
        style_axes(ax, "", f"{kind} RMSE "
                   + ("(N m/kg)" if kind == "torque" else "(W/kg)"))
    add(fig, "Per-subject RMSE of our stance-normalized mean curve against "
             "theirs, by joint and speed — the spread across all "
             f"{len({sid for sid, _, _ in rows})} subjects.")

    med_tq = {j: float(np.median([d["torque_rmse"][ji] for _, _, d in rows]))
              for ji, j in enumerate(joints)}
    med_r = {j: float(np.median([d["torque_corr"][ji] for _, _, d in rows]))
             for ji, j in enumerate(joints)}
    imb = [d["imbalance_frac"] for _, _, d in rows]
    stats_row = "".join([
        stat("subjects (N)", f"{len({sid for sid, _, _ in rows})}", ""),
        stat("subject-speeds analysed", f"{len(rows)}", ""),
        stat("median torque RMSE (ankle/knee/hip)",
             "/".join(f"{med_tq[j]:.2f}" for j in joints), "N m/kg"),
        stat("median torque correlation",
             "/".join(f"{med_r[j]:.2f}" for j in joints), ""),
        stat("energy imbalance (median)",
             f"{100 * float(np.median(imb)):.1f}", "% of peak dE/dt",
             "pass" if np.median(imb) < 0.03 else "fail"),
    ])

    # Comparison data must come from this execution, never implicit saved JSON.
    ablation_html = ""
    if ablations is not None:
        abl = ablations
        VARIANT_LABELS = {
            "baseline": "baseline (ours as published)",
            "hip_bell": "their hip landmark (Bell 1990, per the .mdh)",
            "lp10": "their 10 Hz filters (markers + forces, per the .v3s)",
            "dempster": "their segment masses (Dempster, V3D defaults)",
            "power_scalar": "their power definition (τ·ω only, no F·Δv term)",
            "all": "all four together",
        }
        rows_html = "".join(
            "<tr><td>" + VARIANT_LABELS.get(k, k) + "</td>"
            + "".join(f"<td>{v:.3f}</td>"
                      for v in abl[k]["summary"]["torque_rmse"])
            + "".join(f"<td>{v:.2f}</td>"
                      for v in abl[k]["summary"]["power_rmse"])
            + "</tr>"
            for k in VARIANT_LABELS if k in abl)
        ablation_html = f"""
<h2>Swapping in their conventions, one at a time</h2>
<p>Their exact definitions are in the dataset's own Visual3D model and
pipeline files (<code>RBDSmodelV3D.mdh</code>,
<code>RBDSpipelineV3D.v3s</code>): hip = the Bell (1990) landmark; markers
AND analog low-passed at 10 Hz; default (Dempster-mass) hybrid segments;
power = <code>JOINT_POWER_SCALAR</code>, the full 3-D τ·ω<sub>rel</sub> dot
product (all three components, so not sagittal-only, but rotational-only:
no translational joint-force power, which my wrench power includes).
Each row swaps ONE of my conventions for theirs and re-runs every subject
and speed (medians over all subject-speeds).  Findings 2-4 all come out of
this one table:</p>
<div class="overflow"><table><thead>
<tr><th>variant</th><th>ankle τ</th><th>knee τ</th><th>hip τ</th>
<th>ankle P</th><th>knee P</th><th>hip P</th></tr>
<tr><th></th><th colspan="3">torque RMSE (N m/kg)</th>
<th colspan="3">power RMSE (W/kg)</th></tr>
</thead><tbody>{rows_html}</tbody></table></div>
<h2>2. Power: the gap is mostly about what "joint power" means</h2>
<p>The power definition is the big lever.  Drop the translational term (as
they do) and the hip power gap mostly goes away.  So most of our hip-power
difference was never a disagreement about the motion, just about what
"joint power" includes.</p>

<h2>3. Their 10 Hz filters and Dempster masses: barely matter</h2>
<p>Both move things only a little.  Torque agreement is insensitive to
either one.</p>

<h2>4. Their own hip landmark makes hip torque agree less, and nothing
closes the torque gap</h2>
<p>This one surprised me.  Using THEIR OWN Bell hip landmark makes my hip
torque agree LESS with their export than Harrington does.  So the leftover
hip difference isn't simply hip-centre placement; I think it's compensating
conventions (pelvis-frame construction, Cardan-resolved components) that
happen to cancel better under Harrington.  The convergence figures below
make the general point: adopt their conventions cumulatively and the torque
error barely moves.  That's the between-implementation floor.</p>
{_rbds_cumulative_html(cumulative)}"""

    body = f"""
<p class="eyebrow">boneid · validate · RBDS public running dataset</p>

<h2>1. Torques: about as close as two independent implementations get</h2>
<div class="stat-row">{stats_row}</div>

<p>Same raw markers and forces, independent everything else.  What
differs (and so bounds the agreement): my hip centre is Harrington (2007),
theirs is their own landmark; my segment parameters are de Leva (1996),
theirs are Visual3D defaults; I low-pass kinematics and forces at a matched
12 Hz; and my joint power is the full joint wrench power.  The residual and
energy audit run on every trial, same as everywhere else.  Figures 1 and 2
are ensemble means per speed; Figure 3 is the per-subject spread.</p>

{"".join(figs)}

{ablation_html}
"""
    return html_page("Their Data, Our Pipeline", "rbds", body)


if __name__ == "__main__":
    main()
