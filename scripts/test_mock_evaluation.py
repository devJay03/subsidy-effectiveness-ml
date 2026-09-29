"""Test new mock evaluations with the saved model; no training or metrics."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.excel_evaluation import predict_excel


def main():
    model = joblib.load(ROOT / "models/subsidy_effectiveness_model.joblib")
    original = pd.read_excel(ROOT / "data/mock_evaluation_data_corrected.xlsx", engine="openpyxl")
    results = predict_excel(original, model)
    output = ROOT / "outputs/mock_evaluation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results["summary"])
    print(f"\nResults saved:\n{output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Mock evaluation stopped: {error}") from None
