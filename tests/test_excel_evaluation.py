import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from app.main import app, FarmerEvaluation, health, predict, predict_batch, BatchRequest
from app.main import predict_excel as endpoint
from src.excel_evaluation import predict_excel
from src.features import FEATURES


class Model:
    classes_ = np.array(["High", "Low", "Moderate"])
    model_version = "test"

    def predict_proba(self, frame):
        return np.array([[0.45, 0.11, 0.44], [0.45, 0.11, 0.44], [0, 0, 1]])[:len(frame)]

    def predict(self, frame):
        return self.classes_[self.predict_proba(frame).argmax(axis=1)]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(farm_size=2.0, crop_yield=10.0, average_yield=3.0,
                           average_selling_price=20.0, planting_type=1,
                           subsidy_received=1, **{f"q{i}": 3.5 for i in range(1, 11)})
        self.frame = pd.DataFrame([self.record] * 3).rename(columns=FEATURES)

    def test_mean_probability_wins_over_majority_and_maps_classes(self):
        result = predict_excel(self.frame, Model())
        self.assertEqual(result["evaluation_summary"], {
            "total_respondents": 3, "overall_effectiveness": "Moderate",
            "overall_confidence": 62.67, "average_yield_per_hectare": 5.0,
            "average_questionnaire_score": 3.5,
        })
        self.assertEqual(result["model_distribution"], {"Low": 7.33, "Moderate": 62.67, "High": 30.0})
        self.assertEqual(result["prediction_distribution"], {
            "Low": {"count": 0, "percentage": 0.0},
            "Moderate": {"count": 1, "percentage": 33.33},
            "High": {"count": 2, "percentage": 66.67},
        })
        self.assertIn("Moderately Effective", result["summary"])

    def test_invalid_rows_reject_whole_evaluation(self):
        for value in [None, "bad", 0, -1, float("inf")]:
            with self.subTest(value=value):
                frame = self.frame.astype(object)
                frame.loc[1, FEATURES["farm_size"]] = value
                with self.assertRaises(ValueError):
                    predict_excel(frame, Model())
        with self.assertRaises(ValueError):
            predict_excel(self.frame.iloc[:0], Model())
        with self.assertRaises(ValueError):
            predict_excel(self.frame.drop(columns="Q1"), Model())

    def test_upload_and_existing_endpoints(self):
        stream = io.BytesIO()
        self.frame.to_excel(stream, index=False)
        stream.seek(0)
        with patch.object(app.state, "model", Model(), create=True):
            result = endpoint(UploadFile(file=stream, filename="evaluation.xlsx"))
            self.assertEqual(set(result), {"evaluation_summary", "model_distribution", "prediction_distribution", "summary"})
            self.assertTrue(stream.closed)
            self.assertEqual(health()["status"], "ok")
            record = FarmerEvaluation(**self.record)
            self.assertEqual(predict(record).predicted_effectiveness, "High")
            self.assertEqual(len(predict_batch(BatchRequest(records=[record]))), 1)
        with patch.object(app.state, "model", None, create=True):
            with self.assertRaises(HTTPException) as caught:
                endpoint(UploadFile(file=io.BytesIO(), filename="evaluation.xlsx"))
            self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
