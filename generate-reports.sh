#!/bin/sh
# Execute the canonical notebook-to-HTML report builder from any directory.
set -eu
report_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$report_root"
exec uv run python examples/regenerate_reports.py "$@"
