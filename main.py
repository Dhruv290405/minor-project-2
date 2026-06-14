# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# import shap

# # Initialize the FastAPI application
# app = FastAPI(title="Diabetes Risk ML Engine")

# print("Loading XGBoost Model and SHAP Explainer into memory...")
# # Load the brain globally so it only happens once when the server starts
# model = joblib.load('xgboost_diabetes_model.pkl')
# explainer = shap.TreeExplainer(model)

# class PatientVitals(BaseModel):
#     Pregnancies: int
#     Glucose: float
#     BloodPressure: float
#     SkinThickness: float
#     Insulin: float
#     BMI: float
#     Pedigree: float
#     Age: int

# @app.post("/predict")
# async def assess_risk(vitals: PatientVitals):
#     vitals_dict = vitals.dict()
#     df_format = {key: [value] for key, value in vitals_dict.items()}
#     patient_df = pd.DataFrame(df_format)

#     # FIX: Wrap the prediction in float() to strip away the Numpy data type
#     raw_probability = model.predict_proba(patient_df)[0][1] * 100
#     risk_probability = float(raw_probability)

#     shap_values = explainer(patient_df)
#     impact_scores = shap_values.values[0]
    
#     risk_factors = []
#     protective_factors = []
    
#     for i, feature in enumerate(patient_df.columns):
#         # FIX: Ensure the impact score is also cast to a standard float
#         impact = float(impact_scores[i])
#         value = df_format[feature][0]
        
#         if impact > 0.1:
#             risk_factors.append(f"{feature} of {value} significantly increases risk.")
#         elif impact < -0.1:
#             protective_factors.append(f"{feature} of {value} lowers risk.")

#     return {
#         "status": "success",
#         "risk_percentage": round(risk_probability, 2),
#         "explanation": {
#             "primary_risk_factors": risk_factors,
#             "protective_factors": protective_factors
#         }
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import shap

# The deployed model is trained with a custom cost-sensitive objective (see
# train_cost_sensitive_model.py). We import its sigmoid so serving can convert raw margins
# to probabilities itself — XGBoost won't, because it has no link function for a custom loss.
from custom_objectives import sigmoid

# Columns where a value of 0 means "missing", not a real reading. Must match the
# preprocessing in train_model.py so inference sees data the same way as training.
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Initialize the FastAPI application
app = FastAPI(title="Diabetes Risk ML Engine")

print("Loading XGBoost Model and SHAP Explainer into memory...")
# Load the brain globally
model = joblib.load('xgboost_diabetes_model.pkl')
explainer = shap.TreeExplainer(model)

class PatientVitals(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    Pedigree: float
    Age: int

@app.post("/predict")
async def assess_risk(vitals: PatientVitals):
    # Parse incoming JSON
    vitals_dict = vitals.model_dump()
    df_format = {key: [value] for key, value in vitals_dict.items()}
    patient_df = pd.DataFrame(df_format)

    # Apply the same 0-as-missing transform used during training so the model
    # treats an unmeasured reading as NaN rather than a literal value of 0.
    patient_df[ZERO_AS_MISSING] = patient_df[ZERO_AS_MISSING].replace(0, np.nan)

    # IMPORTANT: with a custom objective XGBoost's predict_proba is unreliable and
    # `predict` returns the RAW MARGIN, not a probability. We take the margin explicitly
    # and apply the sigmoid ourselves to recover the positive-class probability.
    # Note: this is a cost-sensitive *risk score*, deliberately skewed toward catching
    # diabetics, not a calibrated probability — see PAPER.md (calibration caveat).
    margin = float(model.predict(patient_df, output_margin=True)[0])
    risk_probability = float(sigmoid(margin)) * 100

    # Generate the SHAP Explanation
    shap_values = explainer(patient_df)
    impact_scores = shap_values.values[0]
    
    risk_factors = []
    protective_factors = []
    
    for i, feature in enumerate(patient_df.columns):
        impact = float(impact_scores[i])
        value = df_format[feature][0]

        # A 0 in these columns was treated as a missing reading, so describe it
        # as such instead of reporting a misleading literal value of 0.
        if feature in ZERO_AS_MISSING and value == 0:
            descriptor = f"{feature} (not measured)"
        else:
            descriptor = f"{feature} of {value}"

        if impact > 0.1:
            risk_factors.append(f"{descriptor} significantly increases risk.")
        elif impact < -0.1:
            protective_factors.append(f"{descriptor} lowers risk.")

    return {
        "status": "success",
        "risk_percentage": round(risk_probability, 2),
        "explanation": {
            "primary_risk_factors": risk_factors,
            "protective_factors": protective_factors
        }
    }