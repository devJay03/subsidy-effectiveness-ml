"""One feature definition shared by training and inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "Effectiveness Label"
FEATURES = {
    "farm_size": "Farm Size (ha)",
    "planting_type": "Planting Type",
    "average_yield": "Average Yield (tons/ha)",
    "crop_yield": "Crop Yield (tons)",
    "average_selling_price": "Average Selling Price (₱/kg)",
    **{f"q{i}": f"Q{i}" for i in range(1, 11)},
    "subsidy_received": "Subsidy Received",
}
QUESTIONS = [f"Q{i}" for i in range(1, 11)]
DERIVED = ["yield_per_hectare", "questionnaire_average"]


def derive_features(frame):
    result = frame.copy()
    size = result[FEATURES["farm_size"]]
    crop = result[FEATURES["crop_yield"]]
    result[DERIVED[0]] = crop.where(crop >= 0) / size.where(size > 0)
    result[DERIVED[1]] = result[QUESTIONS].mean(axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        return derive_features(X)
