"""Inference only for evaluation workbooks; never fits or scores a model."""

import numpy as np
import pandas as pd

from src.features import DERIVED, FEATURES, QUESTIONS, derive_features

LABELS = ("Low", "Moderate", "High")
ALIASES = {
    "Farm Size": FEATURES["farm_size"],
    "Crop Yield": FEATURES["crop_yield"],
    "Average Selling Price": FEATURES["average_selling_price"],
    "Average Yield": FEATURES["average_yield"],
}


def evaluation_features(original):
    if original.empty:
        raise ValueError("The workbook contains no evaluation records.")
    frame = original.rename(columns=ALIASES)
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate or ambiguous column names.")
    missing = sorted(set(FEATURES.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    # Explicit selection excludes identifiers, labels, and supplied derived values.
    frame = frame[list(FEATURES.values())].copy()
    for column in frame:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or np.isinf(values).any():
            raise ValueError(f"Invalid numeric values in {column}")
        observed = values.dropna()
        if column in QUESTIONS:
            valid = observed.between(1, 5)
        elif column in ("Planting Type", "Subsidy Received"):
            valid = observed.isin([0, 1])
        elif column == FEATURES["farm_size"]:
            valid = observed > 0
        else:
            valid = observed >= 0
        if not valid.all():
            raise ValueError(f"Out-of-range values in {column}")
        frame[column] = values
    return frame


def predict_excel(original, model):
    frame = evaluation_features(original)
    derived = derive_features(frame)
    predictions = model.predict(frame)
    probabilities = pd.DataFrame(
        model.predict_proba(frame), columns=model.classes_, index=frame.index
    )
    # Select by class name, never by a presumed probability column position.
    means = probabilities[list(LABELS)].mean()
    overall = str(means.idxmax())
    total = len(frame)
    confidence = round(float(means[overall]) * 100, 2)
    average_yield = round(float(derived[DERIVED[0]].mean()), 2)
    average_score = round(float(derived[DERIVED[1]].mean()), 2)
    counts = pd.Series(predictions).value_counts()
    distribution = {
        label: {
            "count": int(counts.get(label, 0)),
            "percentage": round(int(counts.get(label, 0)) / total * 100, 2),
        }
        for label in LABELS
    }
    description = {
        "Low": "Low Effectiveness",
        "Moderate": "Moderately Effective",
        "High": "Highly Effective",
    }[overall]
    return {
        "evaluation_summary": {
            "total_respondents": total,
            "overall_effectiveness": overall,
            "overall_confidence": confidence,
            "average_yield_per_hectare": average_yield,
            "average_questionnaire_score": average_score,
        },
        "model_distribution": {
            label: round(float(means[label]) * 100, 2) for label in LABELS
        },
        "prediction_distribution": distribution,
        "summary": (
            f"The seedling subsidy evaluation was classified as {description} "
            f"based on the combined evaluation records of {total} respondents. "
            f"The model produced an overall confidence of {confidence:.2f}% "
            f"for the {overall.lower()} effectiveness classification. "
            f"Respondents recorded an average questionnaire score of {average_score:.2f} "
            f"and an average yield per hectare of {average_yield:.2f} tons/ha. "
            f"Of the {total} evaluation records, {distribution['Moderate']['count']} "
            f"were classified as moderate, {distribution['High']['count']} as high, "
            f"and {distribution['Low']['count']} as low effectiveness."
        ),
    }
