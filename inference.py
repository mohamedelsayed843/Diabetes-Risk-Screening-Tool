"""Inference utilities for the Diabetes Risk Screening Tool.

This module is the single source of truth for:
  * the feature schema (names, order, allowed values),
  * input validation,
  * loading the saved pipeline + threshold,
  * producing predictions.

The notebook and the Streamlit app both import it, so training-time and
serving-time feature handling cannot drift apart.

IMPORTANT: the model was trained on self-reported CDC BRFSS 2015 survey data.
It is a screening-style research/portfolio model, NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

PIPELINE_FILENAME = "diabetes_pipeline.joblib"
CONFIG_FILENAME = "model_config.json"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent

YES_NO = {0: "No", 1: "Yes"}

AGE_GROUPS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
    8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80 or older",
}
GENERAL_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_LEVELS = {
    1: "Never attended school / kindergarten only",
    2: "Grades 1-8 (elementary)",
    3: "Grades 9-11 (some high school)",
    4: "Grade 12 or GED (high school graduate)",
    5: "College 1-3 years (some college / technical school)",
    6: "College 4+ years (college graduate)",
}
INCOME_LEVELS = {
    1: "Less than $10,000", 2: "$10,000 to < $15,000", 3: "$15,000 to < $20,000",
    4: "$20,000 to < $25,000", 5: "$25,000 to < $35,000", 6: "$35,000 to < $50,000",
    7: "$50,000 to < $75,000", 8: "$75,000 or more",
}


def _binary(label: str) -> dict[str, Any]:
    return {"kind": "binary", "label": label, "min": 0, "max": 1, "options": YES_NO}


# Order matches the column order of the original BRFSS 2015 CSV (target excluded).
FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "HighBP": _binary("High blood pressure (ever told by a health professional)"),
    "HighChol": _binary("High cholesterol (ever told by a health professional)"),
    "CholCheck": _binary("Cholesterol check within the past 5 years"),
    "BMI": {"kind": "continuous", "label": "Body Mass Index (kg/m²)", "min": 10.0, "max": 100.0},
    "Smoker": _binary("Smoked at least 100 cigarettes in lifetime"),
    "Stroke": _binary("Ever told you had a stroke"),
    "HeartDiseaseorAttack": _binary("Coronary heart disease or myocardial infarction"),
    "PhysActivity": _binary("Physical activity in the past 30 days (outside of work)"),
    "Fruits": _binary("Eats fruit at least once per day"),
    "Veggies": _binary("Eats vegetables at least once per day"),
    "HvyAlcoholConsump": _binary("Heavy alcohol consumption"),
    "AnyHealthcare": _binary("Has any kind of health care coverage"),
    "NoDocbcCost": _binary("Could not see a doctor in the past 12 months because of cost"),
    "GenHlth": {"kind": "ordinal", "label": "General health", "min": 1, "max": 5, "options": GENERAL_HEALTH},
    "MentHlth": {"kind": "continuous", "label": "Days of poor mental health (past 30 days)", "min": 0, "max": 30},
    "PhysHlth": {"kind": "continuous", "label": "Days of poor physical health (past 30 days)", "min": 0, "max": 30},
    "DiffWalk": _binary("Serious difficulty walking or climbing stairs"),
    "Sex": {"kind": "binary", "label": "Sex", "min": 0, "max": 1, "options": {0: "Female", 1: "Male"}},
    "Age": {"kind": "ordinal", "label": "Age group", "min": 1, "max": 13, "options": AGE_GROUPS},
    "Education": {"kind": "ordinal", "label": "Education level", "min": 1, "max": 6, "options": EDUCATION_LEVELS},
    "Income": {"kind": "ordinal", "label": "Annual household income", "min": 1, "max": 8, "options": INCOME_LEVELS},
}
FEATURE_COLUMNS: list[str] = list(FEATURE_SPECS)


class InputValidationError(ValueError):
    """Raised when incoming data does not match the training schema."""


def validate_input(data: Mapping[str, Any] | list[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Validate raw input and return a DataFrame with exactly FEATURE_COLUMNS, in order.

    Accepts a single record (mapping), a list of records, or a DataFrame.
    Raises InputValidationError listing every problem found.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = pd.DataFrame([dict(data)])
    elif isinstance(data, list) and data and all(isinstance(item, Mapping) for item in data):
        frame = pd.DataFrame([dict(item) for item in data])
    else:
        raise InputValidationError("Input must be a mapping, a non-empty list of mappings, or a DataFrame.")

    if frame.empty:
        raise InputValidationError("Input contains no rows.")

    errors: list[str] = []
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    unexpected = [col for col in frame.columns if col not in FEATURE_COLUMNS]
    if missing:
        errors.append(f"Missing features: {missing}")
    if unexpected:
        errors.append(f"Unexpected features: {unexpected}")
    if errors:
        raise InputValidationError("; ".join(errors))

    frame = frame[FEATURE_COLUMNS]
    for column, spec in FEATURE_SPECS.items():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            errors.append(f"'{column}' must be numeric and not missing.")
            continue
        if ((numeric < spec["min"]) | (numeric > spec["max"])).any():
            errors.append(f"'{column}' must be between {spec['min']} and {spec['max']}.")
            continue
        if spec["kind"] in ("binary", "ordinal") and (numeric % 1 != 0).any():
            errors.append(f"'{column}' must be a whole number.")
            continue
        frame[column] = numeric.astype("float64")

    if errors:
        raise InputValidationError("; ".join(errors))
    return frame


def load_artifacts(artifact_dir: str | Path | None = None):
    """Load the saved pipeline and its configuration (threshold, metrics, versions)."""
    directory = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
    pipeline_path = directory / PIPELINE_FILENAME
    config_path = directory / CONFIG_FILENAME
    for path in (pipeline_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact not found: {path}. Run the notebook first and place "
                f"'{PIPELINE_FILENAME}' and '{CONFIG_FILENAME}' in {directory}."
            )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if list(config["feature_columns"]) != FEATURE_COLUMNS:
        raise RuntimeError("Feature columns in model_config.json do not match inference.py.")

    # Pickled models can misbehave when library versions differ, so warn loudly.
    import sklearn

    trained_with = config.get("library_versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Model trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed. "
            "Install the versions in requirements.txt to avoid inconsistent predictions."
        )

    pipeline = joblib.load(pipeline_path)
    return pipeline, config


def predict_diabetes_risk(
    data: Mapping[str, Any] | list[Mapping[str, Any]] | pd.DataFrame,
    pipeline=None,
    config: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
):
    """Validate input, score it, and apply the saved decision threshold.

    Returns a dict for a single mapping, otherwise a list of dicts, each containing:
      predicted_class : 1 if score >= threshold else 0
      predicted_label : human-readable (non-diagnostic) wording
      probability     : model score for class 1 (see note below)
      threshold       : decision threshold selected on validation data

    Note: when class weights or SMOTE are used, the score is NOT a calibrated
    probability of having diabetes. Treat it as a ranking/screening score.
    """
    if pipeline is None or config is None:
        pipeline, config = load_artifacts(artifact_dir)

    frame = validate_input(data)
    scores = pipeline.predict_proba(frame)[:, 1]
    threshold = float(config["threshold"])

    results = []
    for score in scores:
        predicted_class = int(score >= threshold)
        results.append({
            "predicted_class": predicted_class,
            "predicted_label": (
                "Pattern associated with self-reported prediabetes/diabetes"
                if predicted_class == 1
                else "Pattern not associated with self-reported prediabetes/diabetes"
            ),
            "probability": float(score),
            "threshold": threshold,
        })
    return results[0] if isinstance(data, Mapping) else results
