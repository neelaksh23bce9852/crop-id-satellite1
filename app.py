import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Crop ID from Satellite (Demo)", layout="wide")
st.title("🌾 Crop Identification using Satellite Observations — Demo")

st.markdown("""
This lightweight demo classifies crop type from multi-date spectral bands (Red, NIR, Green).
- Use the provided sample file or upload your own CSV with columns like `date1_red`, `date1_nir`, `date1_green`, ..., up to `date4_*`.
- The model is a small RandomForest trained on synthetic examples for wheat/maize/rice.
""")

MODEL_PATH = Path("src/model.joblib")
ENCODER_PATH = Path("src/label_encoder.joblib")
SAMPLE_CSV = Path("data/sample_pixels.csv")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return clf, le
    else:
        st.error("Model not found. Please run `python train.py` first to create src/model.joblib.")
        st.stop()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    for d in range(1,5):
        red = df[f"date{d}_red".format(d=d)]
        nir = df[f"date{d}_nir".format(d=d)]
        df[f"date{d}_ndvi".format(d=d)] = (nir - red) / (nir + red + 1e-6)
    ndvi_cols = [f"date{d}_ndvi".format(d=d) for d in range(1,5)]
    df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
    df["ndvi_max"] = df[ndvi_cols].max(axis=1)
    df["ndvi_min"] = df[ndvi_cols].min(axis=1)
    return df

clf, le = load_model()

st.sidebar.header("1) Get Data")
use_sample = st.sidebar.checkbox("Use bundled sample data", value=True)
if use_sample:
    df = pd.read_csv(SAMPLE_CSV).sample(100, random_state=42).reset_index(drop=True)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV to continue, or toggle 'Use bundled sample data'.")
        st.stop()
    df = pd.read_csv(up)

st.sidebar.header("2) Predict")
go = st.sidebar.button("Run prediction")

st.subheader("Input preview")
st.dataframe(df.head(10))

if go:
    df_feat = add_features(df.copy())
    feature_cols = [c for c in df_feat.columns if c.startswith("date") and any(b in c for b in ["red","nir","green","ndvi"])]
    preds = clf.predict(df_feat[feature_cols].values)
    df_out = df.copy()
    df_out["pred_label"] = le.inverse_transform(preds)

    st.subheader("Predictions")
    st.dataframe(df_out.head(20))

    # Simple counts
    st.subheader("Class distribution")
    counts = df_out["pred_label"].value_counts().reset_index()
    counts.columns = ["label","count"]
    st.bar_chart(counts.set_index("label"))

    # Download
    st.download_button("Download predictions as CSV", data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
