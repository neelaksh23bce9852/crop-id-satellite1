import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

DATA_CSV = Path("data/sample_pixels.csv")
MODEL_PATH = Path("src/model.joblib")
ENCODER_PATH = Path("src/label_encoder.joblib")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute NDVI and simple stats across dates
    for d in range(1,5):
        red = df[f"date{d}_red".format(d=d)]
        nir = df[f"date{d}_nir".format(d=d)]
        df[f"date{d}_ndvi".format(d=d)] = (nir - red) / (nir + red + 1e-6)
    # Temporal summaries
    ndvi_cols = [f"date{d}_ndvi".format(d=d) for d in range(1,5)]
    df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
    df["ndvi_max"] = df[ndvi_cols].max(axis=1)
    df["ndvi_min"] = df[ndvi_cols].min(axis=1)
    return df

def main():
    df = pd.read_csv(DATA_CSV)
    df = add_features(df)
    feature_cols = [c for c in df.columns if c.startswith("date") and any(b in c for b in ["red","nir","green","ndvi"])]
    X = df[feature_cols].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Saved model to {MODEL_PATH} and encoder to {ENCODER_PATH}")

if __name__ == "__main__":
    main()
