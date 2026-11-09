
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()

# Load dataset
data = pd.read_csv("wine_quality.csv")

# Application title
st.title("🍷 Wine Quality Prediction System")
st.caption(
    "Section B classroom prototype for preliminary quality classification"
)

# Dashboard metrics
c1, c2, c3 = st.columns(3)

c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric(
    "Good-quality records",
    int((data["quality"] >= 6).sum())
)

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# CSV upload section
st.subheader("Predict Uploaded Wine Samples")

st.write(
    "Upload a CSV containing the same predictor columns "
    "as the training data."
)

# Download sample input
template = data.drop(columns=["quality"]).head(5)

st.download_button(
    "Download Sample Input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

# Upload CSV
uploaded = st.file_uploader(
    "Choose Wine Sample CSV",
    type=["csv"]
)

if uploaded is not None:

    new_data = pd.read_csv(uploaded)

    required_columns = list(model.feature_names_in_)

    missing = [
        c for c in required_columns
        if c not in new_data.columns
    ]

    if missing:
        st.error(
            "Missing required columns: " + ", ".join(missing)
        )

    else:
        predictions = model.predict(
            new_data[required_columns]
        )

        results = new_data.copy()

        results["Prediction"] = [
            "GOOD QUALITY" if value == 1
            else "NEEDS IMPROVEMENT"
            for value in predictions
        ]

        st.subheader("Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.metric(
            "Predicted Good-Quality Samples",
            int((predictions == 1).sum())
        )

        st.download_button(
            "Download Prediction Results",
            results.to_csv(index=False),
            "wine_quality_predictions.csv",
            "text/csv"
        )
