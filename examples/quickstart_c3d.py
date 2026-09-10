"""Compatibility entrypoint; the analysis lives in notebooks/c3d_quickstart_report.ipynb."""
from boneid.notebook_reports import main as build_reports


def main():
    build_reports(["c3d_quickstart_report"])


if __name__ == "__main__":
    main()
