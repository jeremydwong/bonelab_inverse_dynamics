"""Execute notebooks, save outputs, and export those exact outputs to HTML."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports"


def source_digest():
    paths = [*ROOT.glob("src/boneid/*.py"),
             *ROOT.glob("matlab-inverse-dynamics/**/*.m"),
             *ROOT.glob("matlab-inverse-dynamics/**/*.json"),
             ROOT / "pyproject.toml", ROOT / "uv.lock"]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def notebook_paths(names=()):
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if names:
        requested = {Path(n).stem for n in names}
        paths = [p for p in paths if p.stem in requested]
        if requested != {p.stem for p in paths}:
            raise ValueError(f"Unknown notebooks: {requested - {p.stem for p in paths}}")
    return paths


def check(paths):
    digest = source_digest()
    for path in paths:
        nb = json.loads(path.read_text())
        meta = nb["metadata"]["boneid_report"]
        if meta.get("source_sha256") != digest:
            raise RuntimeError(f"Backend changed since execution: {path.name}")
        html = OUT / meta["html"]
        stamp = hashlib.sha256(path.read_bytes()).hexdigest()
        if not html.exists() or f'notebook-sha256: {stamp}' not in html.read_text():
            raise RuntimeError(f"HTML is not exported from the current notebook: {path.name}")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and cell["source"]:
                if cell.get("execution_count") is None:
                    raise RuntimeError(f"Unexecuted cell in {path.name}")
                if any(o["output_type"] == "error" for o in cell.get("outputs", [])):
                    raise RuntimeError(f"Failed cell in {path.name}")
    print(f"Freshness verified: {len(paths)} notebook/HTML pairs.", flush=True)


def build(paths):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager

    # Always execute with this command's Python, not a global Jupyter kernel.
    class ProjectKernel(KernelManager):
        def format_kernel_cmd(self, extra_arguments=None):
            return [sys.executable, "-m", "ipykernel_launcher", "-f",
                    self.connection_file, *(extra_arguments or [])]

    OUT.mkdir(exist_ok=True)
    digest = source_digest()
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        meta = nb.metadata.setdefault("boneid_report", {})
        meta.setdefault("html", path.stem + ".html")
        nb = NotebookClient(
            nb, timeout=1800, allow_errors=False,
            kernel_manager_class=ProjectKernel,
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        if source_digest() != digest:
            raise RuntimeError("Backend changed during execution; rerun the build.")
        meta = nb.metadata["boneid_report"]
        meta["source_sha256"] = digest
        meta["executed_at"] = datetime.now(timezone.utc).isoformat()
        nbformat.write(nb, path)
        # Export the saved notebook, so the HTML provenance refers to exact bytes.
        saved = nbformat.read(path, as_version=4)
        # Tutorials show their code (the code IS the lesson); reports hide it.
        html, _ = HTMLExporter(template_name="lab",
                               exclude_input=not meta.get("show_code", False)
                               ).from_notebook_node(saved)
        stamp = hashlib.sha256(path.read_bytes()).hexdigest()
        note = (f'<!-- notebook-sha256: {stamp} -->'
                '<aside style="padding:16px;background:#eef4ed;color:#18351c">'
                f'Exported from executed notebook <a href="../notebooks/{escape(path.name)}">'
                f'{escape(path.name)}</a>. Executed {escape(meta["executed_at"])}. '
                'Rebuild: <code>uv run python examples/regenerate_reports.py</code>.</aside>')
        body_at = html.index('>', html.index('<body')) + 1
        html = html[:body_at] + note + html[body_at:]
        (OUT / meta["html"]).write_text(html)
        print(f'Exported reports/{meta["html"]}', flush=True)
    write_toc()
    # Retire the duplicate ASICS URL without retaining a second report.
    (OUT / "p1_asics_notebook.html").write_text('<!doctype html><html><head>'
        '<meta http-equiv="refresh" content="0;url=p1_asics.html"></head><body>'
        '<a href="p1_asics.html">Canonical notebook-generated ASICS report</a></body></html>')
    # Retired or renamed pages forward to where their content lives now.
    for old, new, label in (
            ("c3d_quickstart_report", "asics_running_c3d", "Asics running report"),
            ("s2s_report", "tutorial_s2s_markers", "Squat-to-stand from c3d files"),
            ("joint_dof_comparison", "p1_report", "P1 report, 3-DoF vs 6-DoF joint work")):
        (OUT / f"{old}.html").write_text('<!doctype html><html><head>'
            f'<meta http-equiv="refresh" content="0;url={new}.html"></head><body>'
            f'<a href="{new}.html">{label}</a></body></html>')
    check(paths)


# ---------------------------------------------------------------------------
# The table of contents: a curriculum, not a directory listing
# ---------------------------------------------------------------------------
#
# One source for what each page is FOR.  `CHAPTERS` is the reading order,
# grouped so each chapter answers one question and hands its answer to the
# next: exact answers by hand -> the real pipeline on a known body -> one
# real subject -> many subjects vs Visual3D -> the Asics reanalysis and the
# legacy MATLAB.  Any notebook missing from every chapter is still linked
# (nothing is silently dropped) under "Uncatalogued", which is a prompt to
# write its entry.

DESTINATION = "quickstart_report"

CHAPTERS = [
    ("Known answers you can check by hand",
     "Two small problems where I can write down the exact answer, so every "
     "step of the method gets graded. This is where the ideas come from: "
     "markers to pose, Newton-Euler, the residual, the energy audit.",
     [
        ("tutorial_brick_2d",
         "One brick carried by a hand: the whole inverse-dynamics workflow on a problem with an exact answer",
         "A hand I never show you carries a brick around, and all I record is "
         "four markers on its corners. From those alone we fit the pose, "
         "difference it twice, and run Newton-Euler; the hand's force and "
         "torque come out as the residual at the top of the chain. Because the "
         "motion was scripted I know the exact answer, so every step gets "
         "graded. Start here if you've never done this before =)"),
        ("tutorial_two_bricks_2d",
         "Two bricks on a hinge: the first joint torque, and a residual whose right answer is zero",
         "Two bricks on a hinge, the bottom one bolted to the floor. Now "
         "there's a measured load at the bottom, one unknown in the middle "
         "(the hinge torque), and nothing touching the top, so for the first "
         "time the residual has a known right answer: zero. I get the hinge "
         "torque bottom-up and top-down and show the gap is the residual, then "
         "make the lower brick 20 % too heavy and watch the residual pick up "
         "exactly the invented weight. Along the way: why I keep the floor's "
         "load as a wrench about a fixed point, and how a joint centre rides "
         "the marker clusters as a virtual point."),
     ]),
    ("The real pipeline on an idealized person",
     "Same idea, but now the simulated person is written out as c3d files and "
     "run through exactly the calls we use on a real runner. If it's right "
     "here, what goes wrong on a real person is the person, not the code.",
     [
        ("tutorial_s2s_markers",
         "Squat-to-stand from c3d files: the real pipeline on a body where every answer is known",
         "The simulator writes the files a lab would give you (a static c3d "
         "and a squat c3d, Asics markerset plus torso markers) and I run the "
         "Asics notebook's cells on them, verbatim. Torques land on a "
         "top-down truth to a few hundredths of a N m against 100+ N m peaks, "
         "the residual is basically zero, and the energy audit closes. Then I "
         "turn the old MATLAB's inertia error back on (about 1 N m, planar or "
         "not) and add 1 mm of marker noise, to show why there's a 12 Hz "
         "filter."),
        ("tutorial_6dof",
         "Squat-to-stand with a sliding knee: the evidence for reporting 6-DoF joint power",
         "Same idealized squat, same pipeline, but now the femur slides up to "
         "1 cm on the tibia, so the knee's force does work too. The pipeline "
         "gets that translational work right (3.214 J vs a true 3.213 J), the "
         "energy audit only closes when it's included, and if I move the knee "
         "centre 2 cm the 3-DoF work jumps from 246 to 268 J while the 6-DoF "
         "work sits at 249.6 J. That last bit is why we report 6-DoF power: on "
         "real people the joint centres are guesses =)"),
     ]),
    ("One real subject, walking",
     "Subject P1 on a dual-belt treadmill, whole body, twelve segments. First "
     "how to run it, then how far to trust each number, then what each "
     "analysis choice does to them.",
     [
        ("quickstart_report",
         "Quickstart on P1 walking: the ten steps you'd run on your own study",
         "Real data now: P1 walking, whole body, one call per step. This is the "
         "page I'd hand you with your own data: edit the config cell and the "
         "ten steps stay the same. Mass comes from the ground force (85.2 kg), "
         "the energy audit closes to 0.85 %, the residual averages +0.9 N, "
         "and ankle/knee torques land within about 3-4 N m of Visual3D's (the "
         "hip is 16-17 N m off, which I think is mostly where we each put the "
         "hip centre)."),
        ("p1_report",
         "Subject P1 walking, every check we have: how far to trust each whole-body number",
         "Same subject, but now I ask how much to believe it. Each number gets "
         "checked against something independent: Visual3D's torques on all 28 "
         "trials, the upper body's weight coming back through L5/S1, the "
         "energy books, and the literature on residuals. Ankles and knees "
         "agree to a few percent, hips less so; the residual averages to "
         "nothing but runs about 14 % of body weight RMS, and I say why. Plus "
         "3-DoF vs 6-DoF joint work, whole-body power, and how much the COP "
         "plane and hip-centre choices move things."),
        ("p1_playground",
         "P1 with every knob exposed: how much each analysis choice moves the torques and the residual",
         "Same trial as the quickstart, but every analysis choice is a "
         "variable. Break things on purpose and watch torque RMS vs Visual3D, "
         "the residual and the energy audit respond. The star knob is the "
         "force-to-mocap time offset: swept from -50 to +50 ms, the residual "
         "bottoms out at -5 ms, but so shallowly that I wouldn't trust it to "
         "find a single-digit-ms lag. This is where I go to poke at things =)"),
     ]),
    ("Does it agree with Visual3D?",
     "One subject could be luck. Here the same pipeline, with the same "
     "settings and no hand-tuning, goes against Visual3D on many people: the "
     "nine walkers, then 39 runners from someone else's lab.",
     [
        ("validation_report",
         "Nine subjects walking vs Visual3D: checking that P1's agreement wasn't a fluke",
         "Same pipeline, same settings, no hand-tuning, over all nine walkers: "
         "500 trial x leg comparisons at 0.65-1.94 m/s. Median torque RMS "
         "against Visual3D is 2.48 / 3.75 / 11.30 N m at ankle / knee / hip, "
         "and as a fraction of peak torque it stays flat with speed. The hip "
         "is worst for every subject, and I think that's where you put the "
         "hip centre, not the dynamics."),
        ("rbds_report",
         "RBDS running, 39 runners vs their own Visual3D: our c3d pipeline on someone else's raw data",
         "A fairer test, I think: a public dataset that ships the raw c3d "
         "files and the authors' own Visual3D answers. I push 39 runners' raw "
         "data through exactly the c3d calls we use for Asics and check the "
         "right leg against theirs. Torques land within 0.23 / 0.26 / 0.11 "
         "N m/kg, and most of the power gap turns out to be a definition "
         "(they leave out the translational term), not a disagreement about "
         "the motion =)"),
     ]),
    ("The Asics reanalysis and the legacy MATLAB",
     "The point of all this: one Asics runner reanalysed from raw c3d, the "
     "old lab MATLAB run on the same files, and why the two disagree.",
     [
        ("asics_running_c3d",
         "One Asics runner from raw c3d: our reanalysis, and how far the old MATLAB was from it",
         "The actual reanalysis. One Asics runner, raw c3d in, torques out. "
         "Then the same two files go through the old lab MATLAB, ported line "
         "by line and checked against the real thing to 1e-11 N m. The shapes "
         "agree (r >= 0.93) but the old code under-reads the knee and hip "
         "peaks, and a ladder that fixes one thing at a time shows how much of "
         "that is the equations and how much is the anthropometrics."),
        ("legacy_error_math",
         "Four legacy MATLAB errors, derived: the math behind each fix in the reanalysis",
         "The Asics page shows the old MATLAB and our pipeline disagree; this "
         "is the why. Four equation-level mistakes: a one-sided inertia "
         "product, a missing gyroscopic term, an angular velocity from the "
         "wrong matrix product, and a free moment in the wrong column. For "
         "each I derive the right statement, point at the line where the old "
         "code departs from it, and run a small check. Two of them surprised "
         "me (one only hides if your lab axes happen to line up =)"),
        ("p1_asics",
         "Two legacy physics mistakes on P1 walking: what each one costs per joint, with nothing else changed",
         "I switch the two verified legacy physics mistakes on inside the "
         "correct pipeline on P1 walking and change nothing else. The "
         "one-sided inertia costs about 8 N m at the hip, the gyroscopic term "
         "barely registers, and the energy audit flags both without any "
         "reference. The angular-velocity bug isn't here; that's on the Asics "
         "running page."),
     ]),
    ("What-if studies and muscles",
     "Same pipeline, same walkers, but now I change something on purpose (the "
     "skeleton) or push past net joint loads into muscles, and ask how much of "
     "it I'd actually believe.",
     [
        ("aff_bowed_femur",
         "Bowed femurs and atypical femoral fracture: what bowing our walkers does to knee and femoral-shaft loads",
         "Atypical femoral fracture is a stress fracture of the lateral femoral "
         "shaft, much more common in East and Southeast Asian women. I read the "
         "literature, then bend the van der Zee walkers' legs on the computer: "
         "a varus knee raises the knee adduction moment ~7 % per degree, and a "
         "bowed shaft raises lateral-tension bending ~10 % per 10 mm of bow. "
         "Wider steps take some of it back (one subject). Plus some honest "
         "speculation about what might help."),
        ("muscle_force_validation",
         "Do optimized muscle forces match measured ones: Prilutsky's cats, the knee Grand Challenge and Edwards' femoral strain",
         "I don't really believe optimized muscle forces, so I went looking for "
         "who has checked them. Individual muscles: only in cats, shape right, "
         "size off by 26-52 % at best. Joint contact: the knee Grand Challenge, "
         "about 0.3-0.5 BW, with the lateral side poorly predicted. Edwards' "
         "static-vs-dynamic femoral strain difference has no measured truth at "
         "all, and swapping the criterion matters more than static vs dynamic."),
        ("p1_muscle_forces",
         "Subject P1 walking, muscle forces: which muscles carry each joint torque",
         "P1's walking through OpenSim, with static optimization and Moco's "
         "dynamic optimization (IPOPT), synced to the animation and the 3x3 "
         "scope, with the muscle model laid out. Tick muscles on and off and "
         "watch whether what's left still adds up to the joint torque. Then knee "
         "contact forces (about 2x what instrumented knees measure, which says "
         "something) and a varus sweep that lifts the lateral condyle off at "
         "about 6 degrees."),
        ("varus_intervention",
         "Step frequency at a fixed walking speed, measured and time-rescaled: "
         "can quicker, shorter steps unload a (varus) knee?",
         "P1 at 1.25 m/s with a metronome at 75 to 135 steps/min (the "
         "experiment is already in the van der Zee data), next to my cheap "
         "version: the preferred walk time-rescaled to f0 +/- 10 steps/min. "
         "Knee contact from OpenSim per step and per km, the knee adduction "
         "moment with the knee pushed into varus, and two animated walkers with "
         "a slider over cadence. The rescale turns out not to be legal, and "
         "per km the knee doesn't care much."),
     ]),
]

TOC_CSS = """
  :root { --bg:#242424; --surface:#28292c; --border:#3d3e42; --text:#cccccc;
          --muted:#999999; --accent:rgb(108,159,213); --gold:#ffc107; }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body { background: var(--bg); color: var(--text); margin: 0 auto; max-width: 780px;
         padding: 40px 20px 80px; line-height: 1.6;
         font-family: "Raleway", "HelveticaNeue", Helvetica, Arial, sans-serif; }
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
  h1.docs-header { text-transform: uppercase; letter-spacing: .2rem; font-weight: 600;
                   font-size: 1.9rem; line-height: 1.15; color: var(--text); margin: 0 0 6px; }
  h1.docs-header .accent { color: var(--accent); }
  h2 { text-transform: uppercase; letter-spacing: .14rem; font-weight: 600; font-size: 1.05rem;
       color: var(--muted); margin: 44px 0 10px; }
  .sub { color: var(--muted); margin-bottom: 30px; max-width: 64ch; }
  .sub p, .prose p { margin: 0 0 12px; }
  .prose { max-width: 64ch; color: var(--text); }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .card { display:block; background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 18px 22px; margin: 14px 0; transition: border-color .15s;
          color: inherit; }
  a.card:hover { border-color: var(--accent); text-decoration: none; }
  .card .num { color: var(--accent); font-weight: 700; margin-right: 10px;
               font-family: "IBM Plex Mono", monospace; }
  .card .t { color: #ececec; font-weight: 600; font-size: 1.08em; }
  .card .d { color: var(--muted); margin-top: 6px; font-size: .95em; }
  .card .src { color: #777; margin-top: 8px; font-size: .8em;
               font-family: "IBM Plex Mono", monospace; }
  .card.destination { border-color: var(--accent); }
  .card.destination .t::after { content: " ← destination"; color: var(--accent);
                                font-weight: 400; font-size: .85em; }
  .note { background: var(--surface); border-left: 3px solid var(--accent);
          padding: 10px 16px; margin: 26px 0; color: var(--muted); font-size: .93em;
          border-radius: 0 8px 8px 0; }
  .foot { margin-top: 44px; color: #777; font-size: .9em; border-top: 1px solid var(--border);
          padding-top: 18px; }
  code { font-family: "IBM Plex Mono", monospace; font-size: .9em; }
  .toggle-container { position: relative; height: 22px; width: 44px; flex: 0 0 auto; margin-top: 6px; }
  .toggle-container input { position:absolute; display:none; width:0; height:0; visibility:hidden; }
  .toggle-container label { position:absolute; width:40px; height:20px; left:1px; top:50%;
      transform: translateY(-50%); display:block; background-color: var(--surface);
      border-radius:20px; cursor:pointer;
      box-shadow: inset 0 5px 15px rgba(0,0,0,.4), inset 0 -5px 15px rgba(255,255,255,.4); }
  .toggle-container label:after { content:""; width:18px; height:18px; position:absolute; top:1px;
      left:1px; background: linear-gradient(180deg,#5a7db8,#486c9a); border-radius:20px;
      box-shadow: inset 0 5px 10px rgba(0,0,0,.2); }
  .toggle-container input:checked + label:after { left:3px; transform: translateX(100%);
      background: linear-gradient(180deg,#ffeb3b,#ffc107); }
  .toggle-container label, .toggle-container label:after { transition: .3s; }
  body.light-mode { background:#ffffff; color:#222; }
  body.light-mode h1.docs-header { color:#1a1a1a; }
  body.light-mode .sub, body.light-mode .card .d, body.light-mode h2 { color:#666; }
  body.light-mode .prose { color:#222; }
  body.light-mode .card { background:#f6f7f9; border-color:#e1e1e1; }
  body.light-mode .card.destination { border-color: var(--accent); }
  body.light-mode .card .t { color:#1a1a1a; }
  body.light-mode .note { background:#f6f7f9; color:#555; }
  body.light-mode .foot { color:#888; }
"""


def _card(stem, title, blurb, number=None, destination=False):
    num = f'<span class="num">{number:02d}</span>' if number is not None else ""
    cls = "card destination" if destination else "card"
    return (f'<a class="{cls}" href="{escape(stem)}.html">{num}'
            f'<span class="t">{escape(title)}</span>'
            f'<div class="d">{escape(blurb)}</div>'
            f'<div class="src">notebooks/{escape(stem)}.ipynb</div></a>')


def write_toc():
    """Write reports/toc.html: the reading order, with a paragraph on what
    each page teaches, in the style of the essay indexes at
    jeremydwong.github.io.  Only pages whose HTML exists are linked."""
    have = {p.stem for p in notebook_paths()
            if (OUT / (json.loads(p.read_text())["metadata"]
                       .get("boneid_report", {}).get("html", p.stem + ".html"))).exists()}
    listed = {stem for _, _, pages in CHAPTERS for stem, _, _ in pages}
    chapters, n = [], 0
    for k, (name, intro, pages) in enumerate(CHAPTERS, start=1):
        cards = ""
        for stem, title, blurb in pages:
            if stem in have:
                n += 1
                cards += _card(stem, title, blurb, number=n,
                               destination=(stem == DESTINATION))
        chapters.append(f'<h2>Chapter {k} · {escape(name)}</h2>'
                        f'<div class="prose"><p>{escape(intro)}</p></div>{cards}')
    chapters = "\n".join(chapters)
    extra = "".join(_card(stem, stem, "Uncatalogued notebook — write its entry in "
                          "boneid.notebook_reports.")
                    for stem in sorted(have - listed))
    executed = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inverse Dynamics, from the Ground Up</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600;700&family=IBM+Plex+Mono:wght@600;700&display=swap" rel="stylesheet">
<style>{TOC_CSS}</style></head><body>
<div class="topbar"><div>
<h1 class="docs-header">Inverse Dynamics, <span class="accent">from the Ground Up</span></h1>
</div><div class="toggle-container">
<input id="mode-toggle" type="checkbox" aria-label="Toggle light mode"><label for="mode-toggle"></label>
</div></div>

<div class="sub">
<p>This is my attempt at a from-scratch walk through inverse dynamics,
starting with a brick you could hold in one hand and ending with a person
running on a treadmill. The question is always the same: <em>what forces
made this thing move the way it did?</em> And (this surprised me a bit) the
ten steps that answer it don't really change either. The body just gets more
complicated, and the data gets less honest =).</p>
<p>Every page is a Jupyter notebook that was run top to bottom and exported
with its outputs, and the tutorials show their code too. So the numbers you
see are the numbers the code actually produced the last time it was built. I
didn't paste anything in.</p>
</div>

<h2>The path</h2>
<div class="prose">
<p>Inverse dynamics is basically Newton's second law run backwards: you watch
the motion and you work out the force. The catch is that the forces we really
care about, the ones muscles make across joints, never get measured directly.
So we rebuild them from the things we can measure: where the markers went,
and what the floor pushed back with.</p>
<p>I think there are four decisions that make this trustworthy, and the
pages are ordered so you pick them up one at a time. First, work with
<b>wrenches about a fixed point</b> instead of centres of pressure. You don't
lose anything, and you never divide by a tiny vertical force. Second, read the
<b>residual</b>, the load that would have to act at the free end of the chain
to make the books balance. It's a result, not something embarrassing to hide.
Third, do an <b>energy audit</b> on every trial. It's computed by a completely
different route than the torque recursion, so it can catch mistakes the
recursion can't see. Fourth, report <b>6-DoF joint power</b> (the joint
torque's power <em>and</em> the joint force's), not just the torque's. Our
joint centres are always guesses, and 6-DoF power comes out the same wherever
you put them, while the torque-only (3-DoF) number doesn't
(<a href="tutorial_6dof.html">the 6-DoF tutorial</a> shows it on a body with a
known answer).</p>
<p>The other thing I care about is going from <b>idealized to messy with the
same code</b>. The simulated pages aren't toys with their own shortcuts: the
squat-to-stand and 6-DoF tutorials write c3d files and run them through
exactly the calls we use on a real runner. So when the answer comes out right
on a body where we know it, that's real evidence the pipeline is right, and
whatever goes wrong on a real person comes from the person (skin, noise,
joint-centre guesses), not from the code.</p>
<p>If you've never done this before, start with the bricks. If you have data
waiting, feel free to jump straight to the destination.</p>
</div>

<div class="note">▶ Every page is just a normal notebook: <code>uv sync</code>
once, then <code>uv run jupyter lab notebooks/&lt;name&gt;.ipynb</code>. The
simulated pages (chapters 1 and 2) don't need any data at all.</div>

{chapters}
{extra}

<div class="foot">
Built from executed notebooks on {executed} · rebuild with
<code>./generate-reports.sh</code> (or <code>uv run python examples/regenerate_reports.py</code>) ·
<a href="https://github.com/jeremydwong/bonelab_inverse_dynamics">source</a> ·
page style after the essay indexes at <a href="https://jeremydwong.github.io">jeremydwong.github.io</a>
</div>
<script>
  var cb = document.getElementById('mode-toggle');
  cb.addEventListener('change', function () {{
    document.body.classList.toggle('light-mode', cb.checked);
  }});
</script>
</body></html>
"""
    (OUT / "toc.html").write_text(body)
    print("Exported reports/toc.html", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook stems; default: all")
    parser.add_argument("--check", action="store_true", help="Fail if notebooks/HTML are stale")
    args = parser.parse_args(argv)
    paths = notebook_paths(args.notebooks)
    check(paths) if args.check else build(paths)


if __name__ == "__main__":
    main()
