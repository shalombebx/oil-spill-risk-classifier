import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Oil Spill Risk Classifier", page_icon="🌊", layout="centered")

st.title("🌊 Oil Spill Risk Classifier")
st.write("Predictive Risk Classification of Oil Spill Severity in the Niger Delta.")

# Cache the model loading to prevent reloading on every UI interaction
@st.cache_resource
def load_components():
    model = joblib.load("models/xgboost_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    # Load target encoder if it was serialized separately for the severity label
    try:
        target_encoder = joblib.load("models/target_encoder.pkl")
    except FileNotFoundError:
        target_encoder = None
    return model, encoders, target_encoder

try:
    model, encoders, target_encoder = load_components()
except FileNotFoundError:
    st.error("Model files not found. Ensure `models/xgboost_model.pkl` and `models/label_encoders.pkl` exist in the root directory.")
    st.stop()

st.sidebar.header("Incident Attributes")
st.sidebar.write("Enter the details of the oil spill incident at the time of report.")

def get_user_input():
    user_data = {}
    
    # Dynamically generate dropdowns for all categorical features using the saved LabelEncoders
    for col, encoder in encoders.items():
        options = list(encoder.classes_)
        user_data[col] = st.sidebar.selectbox(f"{col.capitalize()}", options)
    
    # Manually add the engineered numeric temporal features
    user_data['incident_year'] = st.sidebar.number_input("Incident Year", min_value=2006, max_value=2030, value=2026)
    user_data['incident_month'] = st.sidebar.slider("Incident Month", min_value=1, max_value=12, value=10)
    
    return pd.DataFrame([user_data])

input_df = get_user_input()

st.subheader("Reported Incident Profile")
st.dataframe(input_df, hide_index=True)

if st.button("Predict Severity Risk", type="primary"):
    # Apply label encoding to the categorical user inputs
    encoded_df = input_df.copy()
    for col, le in encoders.items():
        if col in encoded_df.columns:
            encoded_df[col] = le.transform(encoded_df[col].astype(str))
            
    # REORDER COLUMNS: Force alignment with the XGBoost training features
    expected_features = model.feature_names_in_
    encoded_df = encoded_df[expected_features]
            
    # Generate numerical prediction from the XGBoost model
    prediction_encoded = model.predict(encoded_df)[0]
    
    # Translate numerical prediction back to human-readable severity
    if target_encoder:
        prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
    else:
        # Fallback mapping assuming standard alphabetical encoding
        severity_map = {0: "Major", 1: "Medium", 2: "Minor"}
        prediction_label = severity_map.get(prediction_encoded, "Minor")

    st.subheader("Predicted Outcome")
    if prediction_label.lower() == "major":
        st.error(f"🚨 **{prediction_label.upper()} SEVERITY** - Proactive resource allocation recommended.")
    elif prediction_label.lower() == "medium":
        st.warning(f"⚠️ **{prediction_label.upper()} SEVERITY**")
    else:
        st.success(f"✅ **{prediction_label.upper()} SEVERITY**")