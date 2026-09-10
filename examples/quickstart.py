"""Compatibility entrypoint; the analysis lives in notebooks/quickstart_report.ipynb."""
from boneid.notebook_reports import main as build_reports


def main():
    build_reports(["quickstart_report"])


if __name__ == "__main__":
    main()
