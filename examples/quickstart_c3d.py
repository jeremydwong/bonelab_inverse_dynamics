"""Compatibility entrypoint; the analysis lives in notebooks/asics_running_c3d.ipynb."""
from boneid.notebook_reports import main as build_reports


def main():
    build_reports(["asics_running_c3d"])


if __name__ == "__main__":
    main()
