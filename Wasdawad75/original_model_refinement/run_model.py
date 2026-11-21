"""Run the frozen model_v10 candidate against the team's unchanged evaluator."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
COMPONENTS = [
    "03_damped_trend_seasonal", "14_multilevel_native_reconciled",
    "12_xgboost_l1_direct_anchor", "20b_cross_sectional_xgb_over_03",
]


def verify_sources(workspace):
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for relative, expected in manifest["sha256_normalized_utf8"].items():
        path = workspace / relative
        digest = hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Original implementation changed: {relative}")


def configured_source(workspace, name, config):
    path = workspace / name / "run.py"
    code = path.read_text(encoding="utf-8-sig")
    for old, new in config.get("replacements", {}).get(name, []):
        if code.count(old) != 1:
            raise RuntimeError(f"Expected one source match in {name}: {old}")
        code = code.replace(old, new)
    return path, code


def prepare_predictor(workspace, output, config, workers):
    import xgboost as xgb
    original_train = xgb.train
    active_component = [None]
    sequence = [0]

    def train(params, *args, **kwargs):
        gc.collect()
        count = 1 if active_component[0] == COMPONENTS[3] else workers
        booster = original_train(dict(params, nthread=count), *args, **kwargs)
        sequence[0] += 1
        folder = output / "fitted_boosters"
        folder.mkdir(parents=True, exist_ok=True)
        booster.save_model(folder / f"booster_{sequence[0]:03d}.ubj")
        return booster

    xgb.train = train
    path = workspace / "21_diverse_family_ensemble" / "run.py"
    spec = importlib.util.spec_from_file_location("personal_original_ensemble", path)
    ensemble = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble)
    ensemble.COMPONENT_WEIGHTS = list(zip(COMPONENTS, config["weights"]))

    def load_component(name, save_source=True):
        path, code = configured_source(workspace, name, config)
        module = types.ModuleType(f"personal_{name}")
        module.__file__ = str(path)
        exec(compile(code, str(path), "exec"), module.__dict__)

        def predict(fold, history, skeleton, airports, directory):
            folder = output / "model_source"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}.py").write_text(code, encoding="utf-8")
            print(f"Fitting {name}: {fold.name}", flush=True)
            gc.collect()
            active_component[0] = name
            try:
                return module.predict_fold(fold, history, skeleton, airports, directory)
            finally:
                active_component[0] = None
                gc.collect()
        return predict

    ensemble._load_predict_fn = load_component
    scale = float(config.get("prediction_scale", 1.0))

    def predict_fold(*args, **kwargs):
        result = ensemble.predict_fold(*args, **kwargs)
        if scale != 1.0:
            result = result.copy()
            result["y_pred"] *= scale
        return result

    return predict_fold, load_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parents[1] / "abraham/agent_workspace")
    parser.add_argument("--config", type=Path, default=ROOT / "best_config.json")
    parser.add_argument("--mode", choices=["backtest", "holdout"], default="backtest")
    parser.add_argument("--data-dir", type=Path, help="Local pipe-delimited CSVs with original export index")
    parser.add_argument("--external-dir", type=Path, help="Local prepared public feature parquet files")
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs/model_v10_backtest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Validate sources, configuration and imports without training")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    verify_sources(workspace)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    weights = config["weights"]
    if len(weights) != 4 or not all(0 <= w <= 1 for w in weights) or abs(sum(weights) - 1) > 1e-10:
        raise ValueError("Expected four nonnegative component weights summing to one")
    if not 0 < float(config.get("prediction_scale", 1)) < 2 or args.workers < 1:
        raise ValueError("Invalid prediction scale or worker count")
    sys.path.insert(0, str(workspace))
    from shared_eval import data_loader, external_feature_store
    from shared_eval.runner import run_backtest_suite, run_final_holdout_once
    if args.data_dir:
        data_loader.PAX_FORECASTING_PATH = args.data_dir.resolve() / "pax_forecasting.csv"
        data_loader.AIRPORT_INFO_PATH = args.data_dir.resolve() / "airport_info.csv"
    if args.external_dir:
        external_feature_store.EXT_DIR = args.external_dir.resolve()
    predictor, load_component = prepare_predictor(workspace, args.output.resolve(), config, args.workers)
    if args.check_only:
        for name in COMPONENTS:
            load_component(name, save_source=False)
        print("PASS: original sources, frozen configuration and all component imports validated; no training performed")
        return
    for path in [data_loader.PAX_FORECASTING_PATH, data_loader.AIRPORT_INFO_PATH,
                 *[external_feature_store.EXT_DIR / n for n in ["airport_external_annual.parquet", "pair_migration_2015.parquet", "macro_monthly_clean.parquet"]]]:
        if not path.is_file():
            raise FileNotFoundError(f"Required local input missing: {path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    evaluator = run_backtest_suite if args.mode == "backtest" else run_final_holdout_once
    evaluator(output, predictor)
    verify_sources(workspace)


if __name__ == "__main__":
    main()
