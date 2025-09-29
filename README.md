# Exoplanet Habitability Explorer 🌎✨

An interactive data-science project that ranks and visualizes exoplanets
based on simple, explainable habitability criteria using NASA’s Exoplanet Archive.

## Features
- Pulls latest **PSCompPars** catalog via NASA Exoplanet Archive API
- Computes heuristic habitability scores
- Interactive filters & plots (insolation, radius, distance, etc.)
- Hosted demo for live exploration

## Stack
- Python (pandas, numpy, requests)
- Plotly / Streamlit / or React + Plotly
- NASA Exoplanet Archive TAP API

## Project Layout
- `notebooks/` – EDA and experiments
- `src/` – scoring & utils
- `app/` – web UI

## Data Source
[NASA Exoplanet Archive – Planetary Systems Composite Parameters](https://exoplanetarchive.ipac.caltech.edu)  
Pulled via TAP API (see `src/fetch_data.py`)
