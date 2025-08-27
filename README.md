# Crop Identification using Satellite Observations

> **Tagline:** Mapping crops from space.

This repo contains a complete, minimal demo pipeline for crop identification from multi-date spectral bands (Red, NIR, Green).
It includes:
- `train.py` — trains a RandomForest on a small sample dataset.
- `predict.py` — runs batch predictions for a CSV.
- `app.py` — a Streamlit app so others can try your project interactively.
- `data/sample_pixels.csv` — a synthetic sample dataset (wheat/maize/rice).

## Try it locally

```bash
# 1) Create environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the model (creates src/model.joblib)
python train.py

# 4) Launch the demo app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501) and click **Run prediction**.

## Batch prediction

```bash
python predict.py --input data/sample_pixels.csv --output predictions.csv
```

## Where can people try it or see the code?

- **Try it:** Run the Streamlit app locally with `streamlit run app.py` (see steps above).
- **See the code:** This repository (zip in your hackathon submission) is self-contained. You can also push the folder to **GitHub** and enable **Streamlit Community Cloud** to host it publicly.

### Deploy to Streamlit Community Cloud (optional)
1. Push this folder to a GitHub repo.
2. Go to https://streamlit.io/cloud → *Deploy an app* → select your repo/branch → set `app.py` as the entry point.
3. Add Python version (if needed) and `requirements.txt`. Deploy.

## Data

For a quick demo we use a synthetic dataset in `data/sample_pixels.csv`. In a production workflow you would replace this with features derived from actual satellite imagery (e.g., Sentinel‑2) and compute indices such as:
\[ NDVI = \frac{NIR - Red}{NIR + Red} \]

## License
MIT
