import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("src/model.joblib")
ENCODER_PATH = Path("src/label_encoder.joblib")

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

def main(input_csv: str, output_csv: str):
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    df = pd.read_csv(input_csv)
    df = add_features(df)
    feature_cols = [c for c in df.columns if c.startswith("date") and any(b in c for b in ["red","nir","green","ndvi"])]
    preds = clf.predict(df[feature_cols].values)
    df["pred_label"] = le.inverse_transform(preds)
    df.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV with spectral bands")
    ap.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV")
    args = ap.parse_args()
    main(args.input, args.output)
