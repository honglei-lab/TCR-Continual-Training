import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize import curve_metrics, summarize_repeats


def test_weighted_auc_and_observed_threshold():
    metrics = curve_metrics([0, 500, 1500], [40, 60, 80], 70)
    assert metrics["nAUC"] == pytest.approx(190 / 3)
    assert metrics["T_tau"] == 1500
    assert curve_metrics([0, 500], [80, 85], 75)["T_tau"] == 0
    assert curve_metrics([0, 500], [40, 60], 75)["T_tau"] == ">500"
    assert curve_metrics([0, 500], [40, 60], None)["T_tau"] is None


def test_suite_changes_are_paired_and_censoring_is_preserved():
    repeats = []
    for before, after in [(10, 30), (20, 40), (30, 50)]:
        repeats.append({"S0": before, "SB": after, "nAUC": (before + after) / 2,
                        "T_tau": ">500", "per_suite": {"libero_goal": [before, after]}})
    result = summarize_repeats(repeats, ["libero_goal"])
    assert result["summary"]["S0"] == {"mean": 20, "std": 10}
    assert result["delta_success_pp"] == {"mean": 20, "std": 0}
    assert result["per_suite_summary"]["libero_goal"]["delta_pp"] == {"mean": 20, "std": 0}
    assert result["T_tau_per_repeat"] == [">500", ">500", ">500"]
