import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

print("Initializing the AI Interrogator (SHAP)...")

# ---------------------------------------------------------
# STEP 1: Load the Pre-Trained Brain
# ---------------------------------------------------------
# We load the .pkl file you generated in Phase 1. 
# In your final app, your FastAPI server will load this into memory once when it starts.
model = joblib.load('xgboost_diabetes_model.pkl')

# ---------------------------------------------------------
# STEP 2: Simulate an Incoming API Request from Node.js
# ---------------------------------------------------------
# Imagine your React frontend sent this JSON payload, and Node.js forwarded it here.
# This represents a single patient's health vitals.
patient_data = {
    'Pregnancies': [2],
    'Glucose': [175],        # Very high glucose
    'BloodPressure': [80],
    'SkinThickness': [25],
    'Insulin': [0],
    'BMI': [33.6],           # High BMI
    'Pedigree': [0.25],
    'Age': [22]              # Young age
}
# Convert the dictionary to a Pandas DataFrame (which XGBoost expects)
patient_df = pd.DataFrame(patient_data)

# Let's see what the black box predicts first
risk_probability = model.predict_proba(patient_df)[0][1] * 100
print(f"\n[BLACK BOX OUTPUT] Model predicts a {risk_probability:.2f}% risk of Diabetes.")

# ---------------------------------------------------------
# STEP 3: The Interrogation (Running SHAP)
# ---------------------------------------------------------
# We initialize the SHAP explainer with our specific model
explainer = shap.TreeExplainer(model)

# We calculate the SHAP values for our specific patient
shap_values = explainer(patient_df)

# ---------------------------------------------------------
# STEP 4: Translating Math to Medical Rules
# ---------------------------------------------------------
print("\n[EXPLAINABILITY ENGINE] Breaking down the decision...")

# The "base value" is the average risk for everyone in the dataset before knowing any details.
# The "values" array shows how much each specific feature pushed the risk up or down.
feature_names = patient_df.columns
impact_scores = shap_values.values[0]

risk_factors = []
protective_factors = []

# Loop through each feature to see what it did
for i, feature in enumerate(feature_names):
    impact = impact_scores[i]
    value = patient_data[feature][0]
    
    if impact > 0:
        # A positive SHAP value means this pushed the risk HIGHER
        risk_factors.append(f"{feature} ({value}) increased risk by +{impact:.2f} points.")
    elif impact < 0:
        # A negative SHAP value means this pushed the risk LOWER (protective)
        protective_factors.append(f"{feature} ({value}) decreased risk by {impact:.2f} points.")

print("\nPRIMARY RISK FACTORS (Why the score is high):")
for factor in sorted(risk_factors, key=lambda x: float(x.split('+')[1].split(' ')[0]), reverse=True):
    print(f"  -> {factor}")

print("\nPROTECTIVE FACTORS (What kept the score from being even higher):")
for factor in sorted(protective_factors, key=lambda x: float(x.split('by ')[1].split(' ')[0])):
    print(f"  -> {factor}")

# ---------------------------------------------------------
# STEP 5: Generate the Visual Evidence
# ---------------------------------------------------------
# We generate a "Waterfall Plot". This visually charts how we get from the base average risk 
# to this patient's specific high risk. 
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig('patient_explanation.png')
print("\nVisual explanation saved as 'patient_explanation.png'")