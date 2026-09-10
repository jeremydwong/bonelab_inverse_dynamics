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
        html, _ = HTMLExporter(template_name="lab", exclude_input=True).from_notebook_node(saved)
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
    pages = []
    for path in notebook_paths():
        meta = json.loads(path.read_text())["metadata"].get("boneid_report", {})
        if meta.get("html") and (OUT / meta["html"]).exists():
            pages.append(f'<li><a href="{escape(meta["html"])}">{escape(path.stem)}</a></li>')
    (OUT / "toc.html").write_text('<!doctype html><html><head><meta charset="utf-8">'
        '<title>Inverse dynamics reports</title></head><body><h1>Inverse dynamics reports</h1>'
        '<p>All analysis pages are exported from executed notebooks. '
        'Rebuild with <code>uv run python examples/regenerate_reports.py</code>.</p>'
        '<ul>' + ''.join(pages) + '</ul></body></html>')
    # Retire the duplicate ASICS URL without retaining a second report.
    (OUT / "p1_asics_notebook.html").write_text('<!doctype html><html><head>'
        '<meta http-equiv="refresh" content="0;url=p1_asics.html"></head><body>'
        '<a href="p1_asics.html">Canonical notebook-generated ASICS report</a></body></html>')
    check(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook stems; default: all")
    parser.add_argument("--check", action="store_true", help="Fail if notebooks/HTML are stale")
    args = parser.parse_args(argv)
    paths = notebook_paths(args.notebooks)
    check(paths) if args.check else build(paths)


if __name__ == "__main__":
    main()
