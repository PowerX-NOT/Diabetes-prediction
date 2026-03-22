import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AYUSH Diabetes Prediction", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("ayush_diabetes_model.pkl")
    df = pd.read_csv("ayush_ehr_synthetic.csv")
    df = df.drop(columns=["patient_id", "diabetes_mellitus"])
    return model, df


def build_defaults(df_features: pd.DataFrame):
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df_features.columns if c not in numeric_cols]

    defaults = {}
    for c in numeric_cols:
        defaults[c] = float(df_features[c].median())
    for c in cat_cols:
        mode = df_features[c].mode(dropna=True)
        defaults[c] = mode.iloc[0] if len(mode) else ""
    return defaults


def predict(model, input_dict):
    X = pd.DataFrame([input_dict])
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    return pred, proba


st.title("Diabetes Prediction")
st.caption("Predicts `diabetes_mellitus` using `ayush_diabetes_model.pkl`.")

try:
    model, df_features = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.info("Train first: `python3 train_ayush_diabetes_model.py`")
    st.stop()

defaults = build_defaults(df_features)

st.sidebar.header("Patient Inputs")
st.sidebar.caption("This UI exposes common fields; remaining fields use dataset medians/modes.")


def num_input(col, label, step=1.0):
    col_min = float(np.nanmin(df_features[col]))
    col_max = float(np.nanmax(df_features[col]))
    v = float(defaults[col])
    return st.sidebar.number_input(
        label,
        min_value=col_min,
        max_value=col_max,
        value=v,
        step=float(step),
    )


def cat_input(col, label):
    options = [x for x in df_features[col].dropna().unique().tolist()]
    options = sorted(options, key=lambda x: str(x))
    default = defaults[col]
    if len(options) == 0:
        return ""
    if default not in options:
        default = options[0]
    return st.sidebar.selectbox(
        label,
        options=options,
        index=options.index(default),
    )


st.sidebar.subheader("Demographics")
age = num_input("age", "Age", step=1)
sex = cat_input("sex", "Sex")
ethnicity = cat_input("ethnicity", "Ethnicity")
region = cat_input("region", "Region")

st.sidebar.subheader("Anthropometrics")
height_cm = num_input("height_cm", "Height (cm)", step=0.1)
weight_kg = num_input("weight_kg", "Weight (kg)", step=0.1)
bmi = num_input("bmi", "BMI", step=0.1)
waist = num_input("waist_circumference_cm", "Waist circumference (cm)", step=0.1)

st.sidebar.subheader("Vitals")
sbp = num_input("systolic_bp_mmhg", "Systolic BP (mmHg)", step=1)
dbp = num_input("diastolic_bp_mmhg", "Diastolic BP (mmHg)", step=1)
hr = num_input("heart_rate_bpm", "Heart rate (bpm)", step=1)

st.sidebar.subheader("Labs")
fasting_glucose = num_input("fasting_glucose_mg_dl", "Fasting glucose (mg/dL)", step=0.1)
hba1c = num_input("hba1c_percent", "HbA1c (%)", step=0.1)

st.sidebar.subheader("Lifestyle")
smoking = cat_input("smoking_status", "Smoking status")
alcohol = cat_input("alcohol_consumption", "Alcohol consumption")
activity = cat_input("physical_activity_level", "Physical activity")
sleep = num_input("sleep_hours_per_night", "Sleep hours/night", step=0.1)
stress = num_input("stress_level", "Stress level", step=1)

st.sidebar.subheader("Comorbidities")
hypertension_status = st.sidebar.selectbox("Hypertension status", options=[0, 1], index=int(defaults["hypertension_status"]))
ckd = st.sidebar.selectbox("Chronic kidney disease", options=[0, 1], index=int(defaults["chronic_kidney_disease"]))
obesity = st.sidebar.selectbox("Obesity", options=[0, 1], index=int(defaults["obesity"]))
dyslipidemia = st.sidebar.selectbox("Dyslipidemia", options=[0, 1], index=int(defaults["dyslipidemia"]))

st.sidebar.markdown("---")
do_predict = st.sidebar.button("Predict", type="primary", use_container_width=True)

input_row = dict(defaults)
input_row.update(
    {
        "age": age,
        "sex": sex,
        "ethnicity": ethnicity,
        "region": region,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "waist_circumference_cm": waist,
        "systolic_bp_mmhg": sbp,
        "diastolic_bp_mmhg": dbp,
        "heart_rate_bpm": hr,
        "fasting_glucose_mg_dl": fasting_glucose,
        "hba1c_percent": hba1c,
        "smoking_status": smoking,
        "alcohol_consumption": alcohol,
        "physical_activity_level": activity,
        "sleep_hours_per_night": sleep,
        "stress_level": stress,
        "hypertension_status": int(hypertension_status),
        "chronic_kidney_disease": int(ckd),
        "obesity": int(obesity),
        "dyslipidemia": int(dyslipidemia),
    }
)

if do_predict:
    pred, proba = predict(model, input_row)

    st.subheader("Prediction")
    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("Diabetes: YES")
        else:
            st.success("Diabetes: NO")

    with col2:
        if proba is not None:
            st.metric("Probability (diabetes)", f"{proba * 100:.1f}%")
        else:
            st.info("Model does not provide probability.")

    st.markdown("---")
    st.subheader("Input snapshot")
    st.dataframe(pd.DataFrame([input_row]).T.rename(columns={0: "value"}))
else:
    st.info("Enter inputs in the sidebar and click Predict.")
