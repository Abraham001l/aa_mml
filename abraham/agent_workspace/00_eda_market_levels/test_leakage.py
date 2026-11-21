from pathlib import Path
import json


def test_eda_no_holdout() -> None:
    summary_path = Path(__file__).resolve().parent / "outputs" / "eda_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    assert s["max_ym_analyzed"] <= 202407, f"EDA leaked into holdout: {s['max_ym_analyzed']}"
    print("PASS: 00_eda_market_levels strictly constrained to YM <= 202407.")


if __name__ == "__main__":
    test_eda_no_holdout()
