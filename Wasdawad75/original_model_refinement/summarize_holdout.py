"""Print four comparable windows from saved original-evaluator metric reports."""
import argparse
import json
from pathlib import Path


def summarize(refined, baseline):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))["subwindows"]
    current, reference = read(refined), read(baseline)
    windows = [
        ("Year 1", "full_24m", "market_dir_year1"),
        ("Clean 17 months", "clean_through_2025_17m", "market_dir_overall"),
        ("Pre-anomaly 21 months", "clean_pre_anomaly_21m", "market_dir_overall"),
        ("Full 24 months", "full_24m", "market_dir_overall"),
    ]
    lines = ["| Window | Current model WAPE | Reference WAPE |", "|---|---:|---:|"]
    for label, window, level in windows:
        a, b = current[window][level]["wape"], reference[window][level]["wape"]
        lines.append(f"| {label} | {a:.6f} | {b:.6f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.refined, args.baseline))
