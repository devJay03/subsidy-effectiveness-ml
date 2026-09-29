"""Load the saved pipeline once; never train inside the API."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, UploadFile
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features import FEATURES, DERIVED, derive_features
from src.excel_evaluation import predict_excel as evaluate_excel

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/subsidy_effectiveness_model.joblib"
logger = logging.getLogger(__name__)
Score = Annotated[float, Field(ge=1, le=5, allow_inf_nan=False, strict=True)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
BinaryCode = Annotated[int, Field(ge=0, le=1, strict=True)]


class FarmerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    farm_size: Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
    planting_type: BinaryCode
    average_yield: Nonnegative
    crop_yield: Nonnegative
    average_selling_price: Nonnegative
    q1: Score
    q2: Score
    q3: Score
    q4: Score
    q5: Score
    q6: Score
    q7: Score
    q8: Score
    q9: Score
    q10: Score
    subsidy_received: BinaryCode


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[FarmerEvaluation] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    predicted_effectiveness: Literal["Low", "Moderate", "High"]
    probabilities: dict[str, float] | None = None
    derived_values: dict[str, float | None]
    model_version: str


@asynccontextmanager
async def lifespan(application):
    application.state.model = None
    try:
        model = joblib.load(MODEL_PATH)
        if set(model.classes_) != {"Low", "Moderate", "High"}:
            raise ValueError("Unexpected target classes")
        if list(model.feature_names_in_) != list(FEATURES.values()):
            raise ValueError("Model input schema does not match API")
        if not isinstance(model.model_version, str):
            raise ValueError("Missing model version")
        application.state.model = model
    except Exception:
        logger.exception("Model unavailable; train the model before starting the API")
    yield
    application.state.model = None


app = FastAPI(title="Seedling Subsidy Effectiveness ML API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail={"status": "unavailable", "model_loaded": False, "model_version": None})
    return {"status": "ok", "model_loaded": True, "model_version": model.model_version}


def predict_records(records):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Train the model and restart the API.")
    try:
        frame = pd.DataFrame([record.model_dump() for record in records]).rename(columns=FEATURES)
        frame = frame[list(FEATURES.values())]
        predictions = model.predict(frame)
        probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None
        # Use the same feature function as the saved pipeline for response diagnostics.
        derived = derive_features(frame)[DERIVED]
        return [Prediction(
            predicted_effectiveness=str(prediction),
            probabilities={str(label): float(value) for label, value in zip(model.classes_, probabilities[i])}
            if probabilities is not None else None,
            derived_values={column: None if pd.isna(value) else float(value) for column, value in derived.iloc[i].items()},
            model_version=model.model_version,
        ) for i, prediction in enumerate(predictions)]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(record: FarmerEvaluation):
    return predict_records([record])[0]


@app.post("/predict/batch", response_model=list[Prediction], response_model_exclude_none=True)
def predict_batch(request: BatchRequest):
    return predict_records(request.records)


@app.post("/predict/excel")
def predict_excel(file: UploadFile):
    try:
        if not file.filename or Path(file.filename).suffix.lower() != ".xlsx":
            raise HTTPException(status_code=422, detail="Upload an .xlsx evaluation file.")
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model unavailable.")
        try:
            original = pd.read_excel(file.file, engine="openpyxl")
        except Exception:
            raise HTTPException(status_code=422, detail="Cannot read the uploaded .xlsx workbook.") from None
        try:
            result = evaluate_excel(original, model)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception:
            logger.exception("Excel prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed.") from None
        return result
    finally:
        file.file.close()
