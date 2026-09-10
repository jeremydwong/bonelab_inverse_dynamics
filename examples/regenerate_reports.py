"""Rebuild local HTML reports; reuse saved RBDS validation sweep results.

Run: uv run python examples/regenerate_reports.py
Requires the local P1–P9 MAT exports; Asics S1 AM recordings are in data/asics.
"""
from pathlib import Path
import json
import runpy
import time

from boneid import io_v3d, report, validate, viz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'


def main():
    OUT.mkdir(exist_ok=True)
    failures = []

    # Save standalone viewers from the same renders embedded in the reports.
    original_p1_viewer = report.p1_viewer_html
    original_static = viz.render_static_html

    def p1_viewer(*args, **kwargs):
        html = original_p1_viewer(*args, **kwargs)
        (OUT / 'p1_viewer.html').write_text(html)
        return html

    def static_viewer(*args, **kwargs):
        html = original_static(*args, **kwargs)
        (OUT / 's2s_viewer.html').write_text(html)
        return html

    def s2s():
        viz.render_static_html = static_viewer
        try:
            report.main_s2s()
        finally:
            viz.render_static_html = original_static

    def example(name):
        runpy.run_path(str(ROOT / 'examples' / name))['main']()

    def rbds():
        results = json.loads((OUT / 'rbds_sweep.json').read_text())
        (OUT / 'rbds_report.html').write_text(validate.rbds_report_html(results))

    def comparison():
        trial = io_v3d.load_v3d_trial(str(ROOT / 'data/p1_trial13_sample.mat'), 0)
        *_, audit = report.whole_body_run(trial, report.p1_params())
        names = ['R ankle', 'R knee', 'R hip', 'L ankle', 'L knee', 'L hip',
                 'L5/S1', 'R shoulder', 'R elbow', 'L shoulder', 'L elbow']
        (OUT / 'joint_dof_comparison.html').write_text(report.html_page(
            'Joint work: 3-DOF versus 6-DOF', '6dof',
            '<h1>P1 trial 13 — joint work comparison</h1>'
            + report.joint_dof_comparison_html(audit, names)))

    report.p1_viewer_html = p1_viewer
    jobs = [('s2s', s2s), ('p1', report.main_v3d),
            ('asics comparison', report.main_asics),
            ('quickstart', lambda: example('quickstart.py')),
            ('c3d quickstart', lambda: example('quickstart_c3d.py')),
            ('playground', report.main_playground),
            ('validation', lambda: validate.main([])),
            ('RBDS (saved sweep)', rbds), ('6-DOF comparison', comparison)]
    try:
        for name, fn in jobs:
            start = time.monotonic()
            print(f'Building {name}', flush=True)
            try:
                fn()
                print(f'Finished {name} in {time.monotonic() - start:.1f}s', flush=True)
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f'FAILED {name}: {exc}', flush=True)
    finally:
        report.p1_viewer_html = original_p1_viewer

    pages = [('joint_dof_comparison.html', '3-DOF versus 6-DOF joint work'),
             ('p1_report.html', 'P1 whole-body analysis'),
             ('quickstart_report.html', 'Quickstart walkthrough'),
             ('c3d_quickstart_report.html', 'Static-calibrated C3D running'),
             ('p1_asics.html', 'Legacy physics comparison'),
             ('p1_playground.html', 'Interactive P1 playground'),
             ('validation_report.html', 'P1–P9 validation'),
             ('rbds_report.html', 'RBDS validation (saved sweep)'),
             ('s2s_report.html', 'Simulated squat-to-stand'),
             ('p1_viewer.html', 'P1 viewer'), ('s2s_viewer.html', 'Squat-to-stand viewer')]
    (OUT / 'toc.html').write_text(report.html_page('Inverse dynamics reports', 'index',
        '<h1>Inverse dynamics reports</h1><ul>' + ''.join(
            f'<li><a href="{file}">{title}</a></li>' for file, title in pages)
        + '</ul><p>RBDS HTML uses the saved validation sweep; other analysis reports '
          'are recomputed from local recordings or simulation.</p>'))
    if failures:
        raise RuntimeError(f'Report builds failed: {failures}')
    print('All 12 HTML files rebuilt.', flush=True)


if __name__ == '__main__':
    main()
