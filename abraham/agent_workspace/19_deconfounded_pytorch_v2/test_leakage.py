from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_eval.leakage_verifier import verify_external_model_leakage

if __name__ == "__main__":
    verify_external_model_leakage(Path(__file__).resolve().parent, include_macro=False, atol=1e-4)
