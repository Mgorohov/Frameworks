import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

_model = None
_model_path = os.path.join(os.path.dirname(__file__), "credit_model.pkl")

if os.path.exists(_model_path):
    try:
        _model = joblib.load(_model_path)
        logger.info("ML model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load ML model: {e}. Falling back to heuristic.")

_columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country"
]

def _education_to_num(edu: str) -> int:
    mapping = {'no': 1, 'high school': 5, 'bachelor': 10, 'master': 13, 'doctorate': 16}
    return mapping.get(edu.lower(), 5)

def calculate_credit_score(age: int, income: float, education: str, married: str) -> float:
    if _model is not None:
        input_dict = {
            "age": age, "workclass": "Private", "fnlwgt": 150000,
            "education": education, "education-num": _education_to_num(education),
            "marital-status": "Married-civ-spouse" if married.lower() == "yes" else "Never-married",
            "occupation": "Prof-specialty", "relationship": "Husband" if married.lower() == "yes" else "Not-in-family",
            "race": "White", "sex": "Male", "capital-gain": 0, "capital-loss": 0,
            "hours-per-week": 40, "native-country": "United-States"
        }
        input_df = pd.DataFrame([input_dict])[_columns]
        proba = _model.predict_proba(input_df)[0, 1]
        return round(min(max(proba * 100, 0), 100), 2)
    else:
        base = 50
        if education.lower() in ['bachelor', 'master', 'doctorate']:
            base += 15
        if married.lower() == 'yes':
            base += 10
        if income > 50000:
            base += 15
        if age < 25:
            base -= 10
        elif age > 60:
            base += 5
        return min(max(base + (age - 30) * 0.5, 0), 100)

def calculate_risk(metadata: dict) -> float:
    score = calculate_credit_score(
        metadata.get('age', 30), metadata.get('income', 50000),
        metadata.get('education', 'bachelor'), metadata.get('married', 'yes')
    )
    return round(1.0 - (score / 100.0), 2)