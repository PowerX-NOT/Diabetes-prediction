# Diabetes Prediction System (AYUSH EHR)

This project provides an end-to-end Machine Learning pipeline for predicting **Diabetes Mellitus** based on Electronic Health Records (EHR) that include both clinical measurements and traditional AYUSH (Ayurveda, Yoga, Unani, Siddha, and Homeopathy) features.

## 🚀 Live Demo
The application is deployed and can be accessed at:
**[diabete-prediction0.streamlit.app](https://diabete-prediction0.streamlit.app/)**

---

## 📋 Table of Contents
1. [Introduction](#-introduction)
2. [Data Processing](#-data-processing)
3. [Model Development](#-model-development)
4. [Application Features](#-application-features)
5. [Installation & Usage](#-installation--usage)

---

## 🏥 Introduction
The dataset used in this project is `ayush_ehr_synthetic.csv`, which contains 2,000 patient records with 86 distinct features. Unlike standard diabetes datasets (like Pima), this dataset includes:
- **Clinical Vitals**: BP, Heart Rate, Glucose, HbA1c, Lipid profile.
- **AYUSH Features**: Prakriti (Dominant Dosha), Vikriti, Nadi type/rate/quality, Agni status, and more.
- **Lifestyle Factors**: Diet, physical activity, sleep, and stress levels.

---

## 🧹 Data Processing
The data cleaning and preparation are handled automatically by a Scikit-Learn `Pipeline` to ensure consistency between training and inference.

1. **Feature Selection**:
   - **Target**: `diabetes_mellitus` (Binary: 0 or 1).
   - **Identifiers**: `patient_id` is dropped as it contains no predictive value.
2. **Handling Numeric Data**:
   - **Imputation**: Missing values are filled using the **Median** of the column to remain robust against outliers.
   - **Scaling**: All numeric features are scaled using **StandardScaler** to have a mean of 0 and variance of 1.
3. **Handling Categorical Data**:
   - **Imputation**: Missing values are filled with the **Most Frequent** (mode) value.
   - **Encoding**: Categorical strings are converted into numeric format using **OneHotEncoder** (ignoring unknown categories during inference).

---

## 🤖 Model Development
The model is built using a **Logistic Regression** classifier wrapped in a multi-stage pipeline.

- **Class Imbalance**: Since the dataset is highly imbalanced (~4% positive cases), the model uses `class_weight='balanced'` to give more importance to the minority class.
- **Evaluation**: The model achieved high performance during testing:
  - **Accuracy**: 98%
  - **ROC-AUC**: 0.99
  - **F1-Score**: 0.76 (balanced recall/precision for the positive class).

---

## 💻 Application Features
The Streamlit app (`app.py`) provides an intuitive interface for risk assessment:

- **Interactive Inputs**: Sidebar sliders and dropdowns for the most critical patient data (Age, BMI, Vitals, Labs).
- **Smart Defaults**: The app automatically fills the remaining 60+ technical features (like Nadi types or Morbidity codes) using dataset medians and modes, making it usable even with partial data.
- **Probability Scoring**: Shows the exact confidence percentage of the prediction.
- **Input Snapshot**: Displays a transparent view of all data passed to the model.

---

## ⚙️ Installation & Usage

### 1. Requirements
Ensure you have Python installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To retrain the model on the latest data:
```bash
python train_ayush_diabetes_model.py
```
This generates `ayush_diabetes_model.pkl` and `ayush_diabetes_metrics.json`.

### 3. Running Locally
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
- `app.py`: Streamlit web application.
- `train_ayush_diabetes_model.py`: Training script with automated preprocessing.
- `ayush_ehr_synthetic.csv`: The underlying EHR dataset.
- `ayush_diabetes_model.pkl`: The serialized trained model.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Configured to manage large CSVs and model artifacts appropriately.
