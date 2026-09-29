"""Run with python -m src.train."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURES, QUESTIONS, DERIVED, TARGET, FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LABEL_MAP_PATH = ROOT / "data" / "label_mapping.json"


def validate(frame):
    required = [*FEATURES.values(), TARGET]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame[required].copy()
    if frame[TARGET].isna().any():
        raise ValueError("Missing target labels; fix the source dataset.")
    for column in FEATURES.values():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & numeric.isna()).any() or np.isinf(numeric).any():
            raise ValueError(f"Invalid numeric values in {column}")
        frame[column] = numeric
        if numeric.isna().all():
            raise ValueError(f"No observed values in {column}")
        values = numeric.dropna()
        if column in QUESTIONS:
            valid = values.between(1, 5)
        elif column in ("Planting Type", "Subsidy Received"):
            valid = values.isin([0, 1])
        elif column == "Farm Size (ha)":
            valid = values > 0
        else:
            valid = values >= 0
        if not valid.all():
            raise ValueError(f"Out-of-range values in {column}")
    duplicates = int(frame.duplicated().sum())
    # Exact duplicate evaluations must not cross the train/test boundary.
    frame = frame.drop_duplicates()
    if frame.duplicated(subset=list(FEATURES.values()), keep=False).any():
        raise ValueError("Identical inputs have conflicting target labels.")
    return frame, duplicates


def target_names(target):
    names = {"Low", "Moderate", "High"}
    if set(target.unique()) == names:
        return target, {v: v for v in sorted(names)}
    if not LABEL_MAP_PATH.exists():
        raise ValueError(
            "Numeric target codes have no documented label key. Confirm their meaning, "
            "then create data/label_mapping.json mapping code strings to Low, Moderate, High. "
            "No model was saved."
        )
    mapping = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    codes = target.map(lambda v: str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v))
    if set(mapping) != set(codes.unique()) or set(mapping.values()) != names or len(mapping) != 3:
        raise ValueError("Label mapping must match all dataset codes and the three target names.")
    return codes.map(mapping), mapping


def main():
    frame, duplicates = validate(pd.read_excel(ROOT / "data/subsidy_dataset_ml_ready.xlsx"))
    y, mapping = target_names(frame[TARGET])
    X = frame.drop(columns=TARGET)
    excluded = {}
    if X["Subsidy Received"].value_counts(normalize=True).max() >= 0.98:
        excluded["Subsidy Received"] = "Constant or nearly constant (at least 98%)."
    # Never include the target or identifiers. Check low-cardinality direct proxies.
    for column in X:
        if X[column].nunique() == 3 and frame.groupby(column)[TARGET].nunique().max() == 1:
            excluded[column] = "Three-valued feature directly identifies target classes."
    active = [column for column in X if column not in excluded]
    categorical = [c for c in ["Planting Type", "Subsidy Received"] if c in active]
    numeric = [c for c in active if c not in categorical] + DERIVED
    counts = y.value_counts()
    if counts.min() < 5:
        raise ValueError("Each target class needs at least five records for a stratified 80/20 split.")
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    results, fitted = {}, {}
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    print(f"Dataset: {len(frame)} records")
    for name, estimator in candidates.items():
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if name == "Logistic Regression":
            numeric_steps.append(("scale", StandardScaler()))
        preprocess = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ])
        pipeline = Pipeline([("features", FeatureBuilder()), ("preprocess", preprocess), ("model", estimator)])
        pipeline.fit(train_x, train_y)
        predicted = pipeline.predict(test_x)
        precision, recall, f1, _ = precision_recall_fscore_support(test_y, predicted, average="macro", zero_division=0)
        results[name] = {
            "accuracy": float(accuracy_score(test_y, predicted)), "precision": float(precision),
            "recall": float(recall), "f1_score": float(f1),
            "confusion_matrix": confusion_matrix(test_y, predicted, labels=pipeline.classes_).tolist(),
            "confusion_matrix_labels": pipeline.classes_.tolist(),
        }
        fitted[name] = pipeline
        print(f"\n{name}\nAccuracy: {results[name]['accuracy']:.4f}\nF1: {f1:.4f}")
    selected = max(results, key=lambda name: (results[name]["f1_score"], results[name]["accuracy"]))
    pipeline = fitted[selected]
    metadata = {
        "model_name": selected, "model_version": "1.0", "input_features": list(FEATURES.values()),
        "api_feature_mapping": FEATURES, "selected_features": active, "derived_features": DERIVED,
        "excluded_features": excluded, "target_classes": pipeline.classes_.tolist(), "label_mapping": mapping,
        **results[selected], "metric_average": "macro", "dataset_size": len(frame),
        "training_size": len(train_x), "test_size": len(test_x), "random_state": 42,
        "training_date": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__,
        "validation": {"duplicates_removed": duplicates, "missing_values": frame.isna().sum().to_dict(),
                       "target_distribution": counts.to_dict()}, "model_comparison": results,
        "evaluation_note": "Metrics use the held-out 20%; selected pipeline remains fitted on the 80% training split.",
    }
    pipeline.model_version = metadata["model_version"]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "subsidy_effectiveness_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSelected: {selected}\nModel saved successfully.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Training stopped: {error}") from None
