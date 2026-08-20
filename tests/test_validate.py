"""Light smoke test of the cross-subject validation sweep.

The full sweep (nine subjects) takes minutes; this runs the same code path on
the first three trials of p1 and checks the shape of what comes out plus the
one number that matters — the RMS against Visual3D.
"""

from pathlib import Path

import numpy as np
import pytest

from boneid import validate

MAT = Path("/Users/jeremy/Dropbox/Public/inverse-dynamics-test-data/"
           "p1_5StridesData.mat")

pytestmark = pytest.mark.skipif(not MAT.exists(),
                                reason="validation .mat not available")


@pytest.fixture(scope="module")
def result(monkeypatch_module=None):
    """`sweep_subject` on p1, truncated to the first three trials."""
    import boneid.io_v3d as io
    real = io.load_v3d_trials
    try:
        io.load_v3d_trials = lambda path, **kw: real(path, **kw)[:3]
        return validate.sweep_subject(MAT)
    finally:
        io.load_v3d_trials = real


def test_structure(result):
    assert result["subject"] == "p1"
    assert result["n_trials"] == 3
    assert result["failures"] == []
    assert len(result["rows"]) == 6            # 3 trials x 2 legs
    for row in result["rows"]:
        assert set(row) == {"subject", "trial", "leg", "speed", "stride_time",
                            "body_mass", "flag", "rms", "peak", "nrms"}
        assert row["flag"] is None
        assert row["leg"] in ("R", "L")
        assert row["rms"].shape == row["nrms"].shape == row["peak"].shape == (3,)
        assert 0.3 < row["speed"] < 3.0
        assert 30.0 < row["body_mass"] < 150.0


def test_rms_finite_and_small(result):
    rms = np.array([r["rms"] for r in result["rows"]])
    assert np.isfinite(rms).all()
    assert np.median(rms) < 30.0
    assert (rms >= 0).all()


def test_report_assembles(result):
    results = {"subjects": ["p1"], "per_subject": {"p1": result},
               "rows": result["rows"], "failures": result["failures"],
               "n_trials": result["n_trials"]}
    html = validate.validation_report_html(results)
    assert "Cross-Subject Validation" in html
    assert "data:image/svg+xml;base64," in html
    names, med = validate.per_subject_medians(results)
    assert names == ["p1"]
    assert med.shape == (1, 6)
    assert np.isfinite(med).all()
