# Exoplanet Habitability Explorer

An end-to-end data-science & ML project that explores, ranks, and visualizes
thousands of confirmed exoplanets using data from the  
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu).

The project demonstrates data sourcing, feature engineering, interactive visualization,
and machine-learning classification in a single, deployable web app.

---

## Features
- **Live NASA data pull** – fetches the latest PSCompPars catalog from the
  NASA Exoplanet Archive via its TAP API (`src/fetch_data.py`)
- **Explainable habitability score** – a simple, heuristic composite
  based on physical parameters such as insolation, radius, distance,
  stellar temperature, etc.
- **Interactive web app (Streamlit)**  
  - Filter by radius, insolation, discovery year, distance, etc.  
  - **Weight presets** (Conservative HZ / Optimistic HZ / Observation-friendly) +
    sliders for custom scoring  
  - **Click-to-inspect planet details** with score-component breakdown  
  - **Compare tray** for side-by-side comparison of up to 3 planets  
  - Downloadable filtered table
- **ML classifier** – trains a Random-Forest model to predict
  “**optimistic habitable-zone candidate**” label from
  non-leaking astrophysical & engineered features
  (luminosity proxy, semi-major axis, estimated insolation, etc.)  
  - ROC-AUC / PR-AUC / F1 displayed in the app  
  - Optional toggle to show predicted probability & label in the UI
- **Clean architecture & reproducibility** – separate training script, 
  model artifacts in `/models`, Streamlit app in `/app.py`

---

## 🛠 Tech Stack
- **Python**: `pandas`, `numpy`, `requests`, `pyarrow`
- **Data science / ML**: `scikit-learn`, `joblib`
- **Web app / viz**: `Streamlit`, `Plotly`, `streamlit-plotly-events`
- **Data source**: [NASA Exoplanet Archive TAP API](https://exoplanetarchive.ipac.caltech.edu)  

## Typical Workflow
# 1. Create env & install deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Pull latest data
python src/fetch_data.py          # use --limit 500 for quick dev

# 3. (optional) Train / update ML model
python src/train_classifier.py

# 4. Run interactive app locally
streamlit run app.py