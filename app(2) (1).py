import streamlit as st
import pandas as pd
import joblib
import os

# Load trained Random Forest model safely
@st.cache_resource
def load_model():
    model_path = "RandomForest_model.pkl"  # adjust if inside a folder
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# Streamlit app
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🌿 Diabetes Prediction App (Random Forest)")

st.write("This app uses your trained Random Forest model to predict diabetes based on patient details.")

# Input form
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("AGE", min_value=1, max_value=120, value=30)
    gender = st.selectbox("GENDER", ['F', 'M'])
    visit_type = st.selectbox("VISIT TYPE", ['R', 'F'])
    weight = st.number_input("WEIGHT(kg)", min_value=0.0, max_value=300.0, value=70.0)
with col2:
    height = st.number_input("HEIGHT(cm)", min_value=0.0, max_value=300.0, value=165.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    waist = st.number_input("WAIST CIRCUMFERENCE", min_value=0.0, max_value=200.0, value=90.0)
    blood_sugar = st.number_input("BLOOD SUGAR(mmol/L)", min_value=0.0, max_value=40.0, value=8.0)
with col3:
    htn = st.selectbox("HTN", [0.0, 1.0])
    systolic_bp = st.number_input("SYSTOLIC BP", min_value=0, max_value=300, value=120)
    diastolic_bp = st.number_input("DIASTOLIC BP", min_value=0, max_value=200, value=80)

# Build input dataframe
# MODIFICATION: Removed the redundant 'BP(mmHg)' column creation
input_dict = {
    'AGE': [age],
    'GENDER': [gender],
    'VISIT TYPE': [visit_type],
    'WEIGHT(kg)': [weight],
    'HEIGHT(cm)': [height],
    'BMI': [bmi],
    'WAIST CIRCUMFERENCE': [waist],
    'BLOOD SUGAR(mmol/L)': [blood_sugar],
    'HTN': [htn],
    'SYSTOLIC BP': [systolic_bp],  # Direct numeric input
    'DIASTOLIC BP': [diastolic_bp]  # Direct numeric input
}
input_df_processed = pd.DataFrame(input_dict)

# Preprocessing: One-Hot Encoding for categorical features
input_df_processed = pd.get_dummies(input_df_processed, columns=['GENDER', 'VISIT TYPE'], drop_first=True)

# Align columns with training data
X_train_cols = [
    'AGE', 'WEIGHT(kg)', 'HEIGHT(cm)', 'BMI', 'WAIST CIRCUMFERENCE',
    'BLOOD SUGAR(mmol/L)', 'HTN', 'SYSTOLIC BP', 'DIASTOLIC BP',
    'GENDER_M', 'VISIT TYPE_R'
]

# Ensure all expected columns are present (important for one-hot-encoded columns that might be zero)
for col in X_train_cols:
    if col not in input_df_processed.columns:
        # Cast to float to maintain numerical consistency with original training data
        input_df_processed[col] = 0.0

# Final feature selection and ordering
input_df_aligned = input_df_processed[X_train_cols]

# Predict
if st.button("Predict"):
    # Ensure the final DataFrame contains only numeric data
    input_df_aligned = input_df_aligned.apply(pd.to_numeric, errors='coerce')
    
    try:
        pred = model.predict(input_df_aligned)[0]
        # Predict_proba returns probabilities for all classes. We want the probability of class 1 (Diabetic).
        # We use .iloc[0, 1] for safety, as predict_proba returns a 2D array.
        prob = model.predict_proba(input_df_aligned).iloc[0, 1] 

        st.subheader("🔍 Prediction Result")
        outcome_mapping = {0: "Non-Diabetic", 1: "Diabetic"}
        st.success(f"**Outcome:** {outcome_mapping.get(pred, 'Unknown')}")
        st.info(f"**Probability of Diabetes:** {prob:.3f}")
        
    except Exception as e:
        # A model expects a certain shape or data type; a 'ValueError' is common here.
        st.error(f"⚠️ Error during prediction. Check your model's expected feature format: {e}")
