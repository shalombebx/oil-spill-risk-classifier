from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Oil Spill Risk Classifier", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #202a35;
    }
    [data-testid="stSidebar"] {
        background: #f5f7f9;
        border-right: 1px solid #e2e7eb;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #1e2b37;
        letter-spacing: -0.025em;
    }
    .page-kicker {
        color: #526575;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }
    .page-intro {
        color: #5c6873;
        font-size: 1.05rem;
        margin-top: -0.35rem;
        margin-bottom: 2.25rem;
    }
    .severity-major {
        color: #9c3636;
    }
    .severity-medium {
        color: #90651d;
    }
    .severity-minor {
        color: #2d6a50;
    }
    div.stButton > button[kind="primary"] {
        background: #24465c;
        border: 1px solid #24465c;
        border-radius: 5px;
        min-height: 2.75rem;
        padding: 0.45rem 1.15rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #19384d;
        border-color: #19384d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-kicker">Incident assessment</div>', unsafe_allow_html=True)
logo_path = Path(__file__).parent / "assets" / "gemini-svg.svg"
st.image(logo_path.read_bytes(), width=420)
st.markdown(
    '<p class="page-intro">Assess the predicted severity of an oil spill incident in the Niger Delta.</p>',
    unsafe_allow_html=True,
)

# Cache the model loading to prevent reloading on every UI interaction
@st.cache_resource
def load_components():
    model = joblib.load("models/xgboost_weighted_model.pkl")
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
    st.error("Model files not found. Ensure `models/xgboost_weighted_model.pkl` and `models/label_encoders.pkl` exist in the root directory.")
    st.stop()

st.sidebar.title("Incident details")
st.sidebar.caption("Enter the reported incident attributes.")

def get_user_input():
    user_data = {}

    for col, encoder in encoders.items():
        if col.lower() == "severity":
            continue

        options = list(encoder.classes_)
        user_data[col] = st.sidebar.selectbox(f"{col.capitalize()}", options)

    user_data['incident_year'] = st.sidebar.number_input("Incident Year", min_value=2006, max_value=2030, value=2026)
    user_data['incident_month'] = st.sidebar.slider("Incident Month", min_value=1, max_value=12, value=10)

    return pd.DataFrame([user_data])

input_df = get_user_input()

st.subheader("Incident profile")
st.caption("Review the details before running the assessment.")
st.dataframe(input_df, hide_index=True, use_container_width=True)

if st.button("Assess severity", type="primary"):
    encoded_df = input_df.copy()
    for col, le in encoders.items():
        if col in encoded_df.columns:
            encoded_df[col] = le.transform(encoded_df[col].astype(str))

    expected_features = model.feature_names_in_
    encoded_df = encoded_df[expected_features]

    raw_prediction = int(model.predict(encoded_df)[0])

    severity_map = {0: "Major", 1: "Medium", 2: "Minor"}
    prediction_label = severity_map.get(raw_prediction, "Minor")

    st.subheader("Assessment")
    with st.container(border=True):
        severity_class = f"severity-{prediction_label.lower()}"
        st.markdown(
            f'<h3 class="{severity_class}">{prediction_label} severity</h3>',
            unsafe_allow_html=True,
        )
        if prediction_label == "Major":
            st.write("Proactive resource allocation is recommended.")
