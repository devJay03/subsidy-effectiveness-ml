# Seedling Subsidy Effectiveness ML API

Python **3.12**. From the project directory (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dataset: `data/subsidy_dataset_ml_ready.xlsx`. Confirmed target mapping in
`data/label_mapping.json`: 0 = Low, 1 = Moderate, 2 = High.
Planting type uses the dataset's numeric codes (0 or 1); their descriptive names
are not supplied. Questionnaire scores accept decimals from 1 to 5.
Farm size is hectares, crop yield is tons, average yield is tons/ha, and selling
price is pesos/kg. Subsidy received uses 0 or 1 and is retained because it varies
in more than 10% of rows. All request fields are required; invalid inputs return 422.

Train and start:

```powershell
python -m src.train
uvicorn app.main:app --reload
```

The saved pipeline includes feature engineering and preprocessing. Metrics and
confusion matrices for both candidates are in `models/model_metadata.json`.
Predictions classify recorded effectiveness; they do not establish causal impact.
Missing or unloadable models return 503; prediction errors return 500.

Swagger: http://127.0.0.1:8000/docs

Test health, single prediction, and batch prediction (maximum 1,000 records):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$record = @{
    farm_size = 2.0
    planting_type = 1
    average_yield = 12.0
    crop_yield = 9.464
    average_selling_price = 20.05
    q1 = 2.7; q2 = 2.9; q3 = 3.1; q4 = 2.0; q5 = 2.1
    q6 = 3.2; q7 = 1.9; q8 = 2.8; q9 = 1.8; q10 = 1.7
    subsidy_received = 1
}
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body ($record | ConvertTo-Json)
Invoke-RestMethod http://127.0.0.1:8000/predict/batch -Method Post -ContentType 'application/json' -Body (@{records = @($record, $record)} | ConvertTo-Json -Depth 3)
```

Batch returns a JSON array in request order. Probability keys follow the saved
model's classes. Derived values are computed internally.

### Complete Excel evaluation

`POST /predict/excel` accepts multipart form field `file` containing an `.xlsx`
workbook. One upload is one complete program evaluation; rows are respondents.
Use the training workbook's input column headings (short aliases `Farm Size`,
`Crop Yield`, `Average Yield`, and `Average Selling Price` are also accepted).
Identifiers and supplied labels or derived values are ignored. Missing required
columns, blank required values, invalid numbers, and out-of-range values reject
the entire evaluation with 422; no respondent rows are silently excluded.

The response contains:

- `evaluation_summary`: total respondents, overall effectiveness (`Low`,
  `Moderate`, or `High`), overall confidence, average yield per hectare (tons/ha),
  and average questionnaire score.
- `model_distribution`: mean model probabilities for Low, Moderate, and High,
  expressed as percentages.
- `prediction_distribution`: respondent prediction counts and percentages,
  supplied only as supporting information.
- `summary`: a descriptive, non-causal summary suitable for display in Laravel.

The overall class has the highest mean probability across all respondents;
confidence is that class's mean probability as a percentage. Class columns are
mapped using `model.classes_`. Exact ties use Low, Moderate, High order.
Percentages and averages are rounded to two decimal places; clients should format
numbers to two places when displaying trailing zeros. Average yield per hectare
is the mean of each respondent's crop yield divided by farm size. Questionnaire
score is the mean of respondent Q1–Q10 averages. Individual predictions are kept
internal. The saved model and its preprocessing are used without retraining.

Run regression checks with `python -m unittest discover -s tests`.
`python scripts/test_mock_evaluation.py` evaluates the existing mock workbook and
writes `outputs/mock_evaluation_summary.json`.
