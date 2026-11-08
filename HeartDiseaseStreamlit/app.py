from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️")

# These files must be kept beside app.py in the deployed repository.
APP_DIR = Path(__file__).resolve().parent
model = joblib.load(APP_DIR / "KNN_heart.pkl")
scaler = joblib.load(APP_DIR / "scaler.pkl")
expected_columns = joblib.load(APP_DIR / "Expected_columns.pkl")

st.title("Heart Disease Risk Prediction")
st.caption("Enter the requested health measurements to view the model's prediction.")
st.warning("For educational purposes only. This app is not a medical diagnosis or a substitute for professional care.")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest pain type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
    max_hr = st.slider("Maximum heart rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)
    st_slope = st.selectbox("ST slope", ["Down", "Flat", "Up"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build the same drop-first one-hot features used when training the model.
    input_values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": int(sex == "M"),
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA": int(chest_pain == "TA"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST"),
        "ExerciseAngina_Y": int(exercise_angina == "Y"),
        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up"),
    }

    input_df = pd.DataFrame([[input_values[column] for column in expected_columns]], columns=expected_columns)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts a higher likelihood of heart disease. Please consult a qualified healthcare professional.")
    else:
        st.success("The model predicts a lower likelihood of heart disease. This result does not rule out a health condition.")



