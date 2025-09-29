import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.scoring import compute_scores, DEFAULT_WEIGHTS

DATA_PARQUET = Path("data/latest.parquet")

st.set_page_config(
    page_title="Exoplanet Habitability Explorer",
    layout="wide"
)

st.title("Exoplanet Habitability Explorer")
st.caption("Heuristic, explainable scoring over NASA Exoplanet Archive (PSCompPars). Not for publication—just exploration!")

# Load data
@st.cache_data
def load_data(pq_path: Path):
    df = pd.read_parquet(pq_path)
    # clean names for display
    df.rename(columns={
        "pl_name": "name",
        "pl_rade": "radius_Re",
        "pl_bmasse": "mass_Me",
        "pl_orbper": "orbital_period_d",
        "pl_eqt": "eq_temp_K",
        "pl_insol": "insolation_Earth=1",
        "st_teff": "star_teff_K",
        "sy_dist": "distance_pc"
    }, inplace=True)
    return df

if not DATA_PARQUET.exists():
    st.error("No data file found. Please run: `python src/fetch_data.py --limit 500` first.")
    st.stop()

df = load_data(DATA_PARQUET)

# Sidebar filters
st.sidebar.header("Filters")
name_search = st.sidebar.text_input("Search planet name")
radius_min, radius_max = st.sidebar.slider("Radius (Earth radii)", 0.3, 5.0, (0.5, 2.5), 0.1)
insol_min, insol_max = st.sidebar.slider("Insolation (Earth = 1)", 0.01, 5.0, (0.35, 1.5), 0.01)
dist_max = st.sidebar.slider("Max distance (pc)", 1, 3000, 1000, 10)
year_min, year_max = st.sidebar.slider("Discovery year range", 1989, int(df["disc_year"].max() if "disc_year" in df else 2025), (2009, 2025), 1)

# Weights (optional tuning)
with st.sidebar.expander("Scoring weights (advanced)"):
    use_custom = st.checkbox("Tune weights", value=False)
    if use_custom:
        w = {k: st.slider(k, 0.0, 1.0, float(v), 0.01) for k, v in DEFAULT_WEIGHTS.items()}
    else:
        w = DEFAULT_WEIGHTS


# Apply filters
fdf = df.copy()
if name_search.strip():
    fdf = fdf[fdf["name"].str.contains(name_search.strip(), case=False, na=False)]

fdf = fdf[
    (fdf["radius_Re"].between(radius_min, radius_max, inclusive="both")) &
    (fdf["insolation_Earth=1"].between(insol_min, insol_max, inclusive="both")) &
    (fdf["distance_pc"].fillna(np.inf) <= dist_max)
]

if "disc_year" in fdf.columns:
    fdf = fdf[(fdf["disc_year"] >= year_min) & (fdf["disc_year"] <= year_max)]

# Compute scores
scored = compute_scores(fdf.rename(columns={
    "radius_Re": "pl_rade",
    "insolation_Earth=1": "pl_insol",
    "distance_pc": "sy_dist",
    "eq_temp_K": "pl_eqt",
    "star_teff_K": "st_teff"
}), weights=w)

# merge back original display names (compute_scores returned original col names)
scored.rename(columns={
    "pl_rade": "radius_Re",
    "pl_insol": "insolation_Earth=1",
    "sy_dist": "distance_pc",
    "pl_eqt": "eq_temp_K",
    "st_teff": "star_teff_K"
}, inplace=True)

# Main layout
left, right = st.columns((2, 1))

with left:
    st.subheader("Planet map")
    color_col = st.selectbox("Color by", ["score", "eq_temp_K", "radius_Re", "distance_pc", "disc_year"])
    size_col = st.selectbox("Point size by", ["radius_Re", "mass_Me", "score"])

    fig = px.scatter(
        scored,
        x="insolation_Earth=1",
        y="radius_Re",
        color=color_col,
        size=size_col,
        hover_data=["name", "eq_temp_K", "distance_pc", "star_teff_K", "discoverymethod", "disc_year"],
        labels={
            "insolation_Earth=1": "Insolation (Earth=1)",
            "radius_Re": "Radius (Earth radii)"
        },
        title="Insolation vs Radius"
    )
    fig.update_xaxes(type="log")  # log-x helps spread insolation
    st.plotly_chart(
    fig,
    width="stretch",                 # ← replaces use_container_width
    config={                          # ← new way to pass Plotly config
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["toImage"]
    }
)

with right:
    st.subheader("Top candidates")
    topk = st.slider("Show top N", 5, 100, 20, 5)
    show_cols = ["name", "score", "radius_Re", "insolation_Earth=1", "eq_temp_K", "distance_pc", "star_teff_K", "discoverymethod", "disc_year"]
    st.dataframe(scored.sort_values("score", ascending=False)[show_cols].head(topk), width="stretch")

    # Download
    st.download_button(
        label="Download filtered (CSV)",
        data=scored[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="exoplanets_filtered.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("**Notes**: Scores are heuristic and for exploration only. Real habitability depends on atmosphere, geology, magnetic field, etc.")
