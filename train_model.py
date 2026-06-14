# STANDARD BASELINE trainer (control group).
# This trains a plain `binary:logistic` XGBoost — the symmetric-loss baseline that the
# Cost-Sensitive Medical XGBoost is compared against in compare_models.py / PAPER.md.
# To train and deploy the safer, false-negative-penalizing model, use
# train_cost_sensitive_model.py instead.
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Columns where a value of 0 is physiologically impossible and therefore
# represents a MISSING reading in the Pima dataset (not a real measurement).
# 'Pregnancies' is excluded because 0 is a legitimate value.
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("Starting the ML Build Process...")

# ---------------------------------------------------------
# STEP 1: Load the Data (The Database Query equivalent)
# ---------------------------------------------------------
# We are pulling the Pima dataset directly from a public raw CSV for now.
# 'Outcome' is our target: 1 means Diabetic, 0 means Non-Diabetic.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

# Treat impossible 0 readings as missing. XGBoost handles NaN natively: during
# training it learns the best default branch direction for missing values, so we
# don't need to impute. The serving code (main.py) applies this same transform.
data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)

# ---------------------------------------------------------
# STEP 2: Prepare the Data (The Payload separation)
# ---------------------------------------------------------
# 'X' represents the input features (the JSON payload the user will eventually send).
# 'y' represents the target answer (what the model is trying to learn to predict).
X = data.drop('Outcome', axis=1) 
y = data['Outcome']

# We split the data: 80% to train the model, 20% to test if it actually learned anything.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# STEP 3: Initialize and Train the Model (The Compilation)
# ---------------------------------------------------------
# We create an instance of the XGBoost Classifier. 
# Think of this as instantiating an empty class with complex internal logic.
model = xgb.XGBClassifier(
    objective='binary:logistic', # We want a binary outcome (0 or 1)
    eval_metric='logloss',       # How the model measures its own errors during training
    seed=42
)

print("Training the XGBoost model...")
# The .fit() method is where the actual math happens. 
# It builds the decision trees based on the training data.
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: Evaluate the Model (The Unit Test)
# ---------------------------------------------------------
# We ask the model to predict outcomes for the 20% of data it hasn't seen yet.
predictions = model.predict(X_test)

# We compare its predictions against the actual true answers.
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete. Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# STEP 5: Save the Model (The Export)
# ---------------------------------------------------------
# We serialize (pickle) the trained model into a file. 
# Your future FastAPI service will load this file to make real-time predictions.
filename = 'xgboost_diabetes_model.pkl'
joblib.dump(model, filename)
print(f"Model successfully saved as {filename}")