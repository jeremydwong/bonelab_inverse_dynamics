"""Canonical report build: execute notebooks, then export their saved outputs.

uv run python examples/regenerate_reports.py          # rebuild all
uv run python examples/regenerate_reports.py p1_asics # rebuild one
uv run python examples/regenerate_reports.py --check  # verify freshness
"""
from boneid.notebook_reports import main

if __name__ == "__main__":
    main()
