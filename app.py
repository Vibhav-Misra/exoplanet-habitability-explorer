import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from src.scoring import compute_scores, DEFAULT_WEIGHTS
from pathlib import Path

import numpy as np
import pandas as pd

def build_model_features(view_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    cols_to_pull = ["name", "st_rad", "st_mass", "orbital_period_d", "star_teff_K", "discoverymethod"]
    base = view_df.merge(full_df[cols_to_pull], on="name", how="left")

    feat = pd.DataFrame(index=base.index)

    feat["pl_eqt"]     = base.get("eq_temp_K") 
    feat["st_teff"]    = base.get("star_teff_K")
    feat["st_rad"]     = base.get("st_rad")     
    feat["st_mass"]    = base.get("st_mass")    
    feat["sy_dist"]    = base.get("distance_pc")
    feat["pl_orbper"]  = base.get("orbital_period_d")
    feat["discoverymethod"] = base.get("discoverymethod")
    feat["lum_rel"] = (feat["st_rad"] ** 2) * ((feat["st_teff"] / 5772.0) ** 4)
    P_years = feat["pl_orbper"] / 365.25
    feat["a_AU"] = ((P_years ** 2) * feat["st_mass"]) ** (1/3)
    feat["insol_est"] = feat["lum_rel"] / (feat["a_AU"] ** 2)

    model_cols = [
        "pl_eqt", "st_teff", "st_rad", "st_mass", "sy_dist", "pl_orbper",
        "lum_rel", "a_AU", "insol_est", 
        "discoverymethod",
    ]
    return feat[model_cols]

PRESET_WEIGHTS = {
    "Conservative HZ": {
        "insolation_proximity": 0.45,
        "radius_proximity": 0.30,
        "proximity_bonus": 0.10,
        "temp_band_bonus": 0.12,
        "sunlike_bonus": 0.03,
    },
    "Optimistic HZ": {
        "insolation_proximity": 0.30,
        "radius_proximity": 0.20,
        "proximity_bonus": 0.20,
        "temp_band_bonus": 0.25,
        "sunlike_bonus": 0.05,
    },
    "Observation-friendly": {
        "insolation_proximity": 0.20,
        "radius_proximity": 0.15,
        "proximity_bonus": 0.45,  
        "temp_band_bonus": 0.15,
        "sunlike_bonus": 0.05,
    },
}

DATA_PARQUET = Path("data/latest.parquet")

st.set_page_config(
    page_title="Exoplanet Habitability Explorer",
    layout="wide"
)

st.title("Exoplanet Habitability Explorer")
st.caption("Heuristic, explainable scoring over NASA Exoplanet Archive (PSCompPars). Not for publication—just exploration!")

@st.cache_resource
def load_model():
    mpath = Path("models/hz_rf.pkl")
    if mpath.exists():
        return joblib.load(mpath)
    return None

@st.cache_data
def load_metrics():
    mpath = Path("models/hz_rf_metrics.json")
    if mpath.exists():
        import json
        return json.load(open(mpath))
    return {}

@st.cache_data
def load_data(pq_path: Path):
    df = pd.read_parquet(pq_path)
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

model = load_model()
metrics = load_metrics()

def get_feature_importances(model):
    try:
        import numpy as np
        rf = model.named_steps["clf"]
        return rf.feature_importances_
    except Exception:
        return None

if not DATA_PARQUET.exists():
    st.error("No data file found. Please run: `python src/fetch_data.py --limit 500` first.")
    st.stop()

df = load_data(DATA_PARQUET)

if st.sidebar.button("Reset filters"):
    st.session_state.clear()
    st.rerun()

if "disc_year" in df.columns:
    disc_year_num = pd.to_numeric(df["disc_year"], errors="coerce")
    y_max = int(np.nanmax(disc_year_num)) if np.isfinite(np.nanmax(disc_year_num)) else 2025
else:
    y_max = 2025

year_min, year_max = st.sidebar.slider(
    "Discovery year range",
    1989, y_max,
    (max(1989, min(2009, y_max)), y_max),
    1
)

st.sidebar.header("Filters")
name_search = st.sidebar.text_input("Search planet name")
radius_min, radius_max = st.sidebar.slider(
    "Radius (Earth radii)", 0.3, 5.0, (0.5, 2.5), 0.1, key="radius_range"
)
insol_min, insol_max = st.sidebar.slider(
    "Insolation (Earth = 1)", 0.01, 5.0, (0.35, 1.5), 0.01, key="insolation_range"
)
dist_max = st.sidebar.slider(
    "Max distance (pc)", 1, 3000, 1000, 10, key="distance_max"
)
year_min, year_max = st.sidebar.slider(
    "Discovery year range",
    1989,
    int(df["disc_year"].max() if "disc_year" in df else 2025),
    (2009, 2025),
    1,
    key="disc_year_range"
)

st.sidebar.markdown("---")
use_ml = st.sidebar.checkbox("Show ML-based label (RF)", value=bool(model))
ml_threshold = st.sidebar.slider("ML threshold", 0.05, 0.95, 0.5, 0.05, disabled=not use_ml)

with st.sidebar.expander("Scoring weights"):
    if "current_weights" not in st.session_state:
        st.session_state.current_weights = DEFAULT_WEIGHTS.copy()

    preset = st.selectbox(
        "Preset",
        ["Custom"] + list(PRESET_WEIGHTS.keys()),
        index=0,
        help="Start from a preset, then tweak below."
    )

    if preset != "Custom":
        st.session_state.current_weights = PRESET_WEIGHTS[preset].copy()

    w = {}
    for k, v in st.session_state.current_weights.items():
        w[k] = st.slider(k, 0.0, 1.0, float(v), 0.01, key=f"weight_{k}")
    st.session_state.current_weights = w

    if st.button("Save current as My Preset"):
        PRESET_WEIGHTS["My Preset"] = st.session_state.current_weights.copy()
        st.success("Saved as 'My Preset' (temporary for this session).")

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

if fdf.empty:
    st.warning("No results match your filters. Try widening them.")
    st.stop()    

scored = compute_scores(fdf.rename(columns={
    "radius_Re": "pl_rade",
    "insolation_Earth=1": "pl_insol",
    "distance_pc": "sy_dist",
    "eq_temp_K": "pl_eqt",
    "star_teff_K": "st_teff"
}), weights=w)

if use_ml and model is not None:
    try:
        feat = build_model_features(scored, df) 
        proba = model.predict_proba(feat)[:, 1]
        scored["ml_prob"] = proba
        scored["ml_label"] = (scored["ml_prob"] >= ml_threshold).astype(int)
    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        scored["ml_prob"] = np.nan
        scored["ml_label"] = np.nan
        use_ml = False
else:
    scored["ml_prob"] = np.nan
    scored["ml_label"] = np.nan



scored.rename(columns={
    "pl_rade": "radius_Re",
    "pl_insol": "insolation_Earth=1",
    "sy_dist": "distance_pc",
    "pl_eqt": "eq_temp_K",
    "st_teff": "star_teff_K"
}, inplace=True)

left, right = st.columns((2, 1))

with left:
    st.subheader("Planet map")
    color_col = st.selectbox("Color by", ["score", "eq_temp_K", "radius_Re", "distance_pc", "disc_year"])
    size_col = st.selectbox("Point size by", ["radius_Re", "mass_Me", "score"])

    num_cols = ["insolation_Earth=1","radius_Re","score","mass_Me","eq_temp_K",
            "distance_pc","star_teff_K","disc_year"]

    for c in num_cols:
        if c in scored.columns:
            scored[c] = pd.to_numeric(scored[c], errors="coerce")

    plot_df = scored[scored["insolation_Earth=1"].notna() & scored["radius_Re"].notna()].copy()
    st.write("Rows with valid x/y for plotting:", len(plot_df))

    if size_col not in plot_df.columns:
        st.warning(f"'{size_col}' not available; falling back to 'radius_Re'.")
        size_col = "radius_Re"
    if color_col not in plot_df.columns:
        st.warning(f"'{color_col}' not available; falling back to 'score'.")
        color_col = "score" if "score" in plot_df.columns else None

    hover_cols = ["name", "eq_temp_K", "distance_pc", "star_teff_K", "discoverymethod", "disc_year"]
    if use_ml:
        hover_cols = ["name", "ml_prob", "ml_label"] + hover_cols
    for col in ["eq_temp_K", "distance_pc", "star_teff_K", "discoverymethod", "disc_year"]:
        if col in plot_df.columns:
            hover_cols.append(col)

    plot_df["insolation_Earth=1"] = pd.to_numeric(plot_df["insolation_Earth=1"], errors="coerce")
    plot_df["radius_Re"] = pd.to_numeric(plot_df["radius_Re"], errors="coerce")
    if size_col in plot_df:
        plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors="coerce")

    plot_df = plot_df[
        plot_df["insolation_Earth=1"].notna()
        & (plot_df["insolation_Earth=1"] > 0)              # required for log scale
        & plot_df["radius_Re"].notna()
    ].copy()

if plot_df.empty:
    st.warning("No rows available to plot after numeric/log filtering.")
    st.stop()

extra_color_opts = ["score", "eq_temp_K", "radius_Re", "distance_pc", "disc_year"]
if use_ml:
    extra_color_opts = ["ml_prob", "ml_label"] + extra_color_opts
color_col = st.selectbox(
    "Color by",
    extra_color_opts,
    index=0 if use_ml else 0,
    key="color_by_main"        
)


if use_ml:
    only_ml_pos = st.checkbox("Filter to ML-positive (label=1)", value=False)
    if only_ml_pos:
        scored = scored[scored["ml_label"] == 1]    

if size_col in plot_df:
    s = plot_df[size_col]
    lo, hi = s.quantile([0.05, 0.95])                  
    plot_df[size_col] = s.clip(lower=lo, upper=hi)

    fig = px.scatter(
    plot_df,
    x="insolation_Earth=1",
    y="radius_Re",
    color=color_col if color_col in plot_df.columns else None,
    size=size_col if size_col in plot_df.columns else None,
    size_max=24,                                        
    hover_data=hover_cols,
    custom_data=["name"] if "name" in plot_df.columns else None,
    labels={"insolation_Earth=1": "Insolation (Earth=1)", "radius_Re": "Radius (Earth radii)"},
    title="Insolation vs Radius",
    render_mode="webgl",                                
    )

    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[radius_min, radius_max])
    fig.update_xaxes(tickmode="array", tickvals=[0.01, 0.1, 1, 10, 100], ticktext=["0.01", "0.1", "1", "10", "100"])
    fig.update_traces(marker=dict(sizemode="area", opacity=0.6, sizemin=4))

    st.plotly_chart(fig, use_container_width=True)

    names = plot_df["name"].dropna().astype(str).unique().tolist()
    options = ["—"] + names
    st.markdown("### Select a planet")
    options = ["—"] + plot_df["name"].tolist()
    choice = st.selectbox("Planet", options, index=0, key="planet_select")

    if choice != "—":
        st.session_state.selected_planet = choice
    else:
        if "selected_planet" not in st.session_state and not plot_df.empty:
            top_name = plot_df.sort_values("score", ascending=False)["name"].iloc[0] if "score" in plot_df.columns else plot_df["name"].iloc[0]
            st.session_state.selected_planet = top_name

with right:
    tab_details, tab_top = st.tabs(["Planet Details", "Top candidates"])

    if use_ml and metrics:
        st.markdown("**Model Metrics (held-out test)**")
        st.write(f"ROC-AUC: {metrics.get('roc_auc', None):.3f} | PR-AUC: {metrics.get('pr_auc', None):.3f} | F1: {metrics.get('f1', None):.3f}")
        st.caption("Label: optimistic HZ (insolation 0.35–1.5 ⨁ & radius 0.5–1.75 Re). Features: radius, insolation, eq. temp, stellar temp, distance, period, discovery method.")

    with tab_details:
        st.caption("Click a point on the plot to see details and score breakdown.")

        if "selected_planet" not in st.session_state:
            st.session_state.selected_planet = None

        if st.session_state.selected_planet:
            pname = st.session_state.selected_planet
            prow = scored.loc[scored["name"] == pname].head(1)
            if len(prow):
                r = prow.iloc[0].to_dict()
                st.subheader(pname)
                st.write(
                    f"**Score:** {r.get('score', float('nan')):0.1f}  |  "
                    f"**Radius:** {r.get('radius_Re')} Re  |  "
                    f"**Insolation:** {r.get('insolation_Earth=1')} ⨁  |  "
                    f"**Eq. Temp:** {r.get('eq_temp_K')} K  |  "
                    f"**Distance:** {r.get('distance_pc')} pc"
                )

                part_cols = [c for c in prow.columns if c.startswith("part_")]
                if part_cols:
                    parts = (
                        prow[part_cols]
                        .iloc[0]  
                        .astype(float) 
                        .rename_axis("component") 
                        .reset_index(name="value")
                        .sort_values("value", ascending=False)
                    )
                    st.bar_chart(parts.set_index("component")["value"])
                else:
                    st.caption("No part_* columns found for breakdown.")
                search_url = f"https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*"
                st.markdown(
                    f"[Open NASA Exoplanet Archive search for **{pname}**](https://ui.adsabs.harvard.edu/search/q={pname.replace(' ', '%20')})"
                )

                if "compare" not in st.session_state:
                    st.session_state.compare = []

                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button("➕ Add to Compare"):
                        if pname not in st.session_state.compare:
                            if len(st.session_state.compare) < 3:
                                st.session_state.compare.append(pname)
                            else:
                                st.warning("Compare tray is full (max 3). Remove one first.")
                with col2:
                    if st.button("Clear Selection"):
                        st.session_state.selected_planet = None

        else:
            st.info("No planet selected. Click a point on the scatter plot.")

    with tab_top:
        topk = st.slider("Show top N", 5, 100, 20, 5, key="topN_slider")

        show_cols = ["name", "score", "radius_Re", "insolation_Earth=1", "eq_temp_K",
             "distance_pc", "star_teff_K", "discoverymethod", "disc_year"]
avail_cols = [c for c in show_cols if c in scored.columns]  

st.dataframe(scored.sort_values("score", ascending=False)[avail_cols].head(topk),
             use_container_width=True)
st.download_button("Download filtered (CSV)",
                   data=scored[avail_cols].to_csv(index=False).encode("utf-8"),
                   file_name="exoplanets_filtered.csv",
                   mime="text/csv")

st.markdown("---")
st.subheader("Compare (up to 3 planets)")

if "compare" not in st.session_state:
    st.session_state.compare = []

if st.session_state.compare:
    cols = st.columns(len(st.session_state.compare))
    for i, pname in enumerate(st.session_state.compare):
        with cols[i]:
            st.write(f"**{pname}**")
            if st.button(f"Remove {i+1}", key=f"remove_{i}"):
                st.session_state.compare.pop(i)
                st.rerun()
else:
    st.caption("Use 'Add to Compare' in the Details tab to fill this tray.")

if len(st.session_state.compare) >= 2:
    existing = set(scored["name"].astype(str))
    st.session_state.compare = [p for p in st.session_state.compare if p in existing]

    cdf = scored[scored["name"].isin(st.session_state.compare)].copy()

    compare_cols = [
        "name", "score", "radius_Re", "mass_Me", "insolation_Earth=1",
        "eq_temp_K", "distance_pc", "star_teff_K", "discoverymethod", "disc_year"
    ]
    compare_avail = [c for c in compare_cols if c in cdf.columns]

    if not compare_avail:
        st.warning("No comparable columns available in the current dataset.")
    else:
        sort_by = "score" if "score" in cdf.columns else (
            "radius_Re" if "radius_Re" in cdf.columns else compare_avail[0]
        )
        st.dataframe(
            cdf[compare_avail].sort_values(sort_by, ascending=False),
            use_container_width=True
        )

params = {
    "rmin": radius_min, "rmax": radius_max,
    "imin": insol_min, "imax": insol_max,
    "dmax": dist_max, "ymin": year_min, "ymax": year_max,
}

st.markdown(f"**Permalink params:** `{dict(st.query_params)}`")

if st.button("Copy current filters to URL"):
    st.query_params.update(**{k: str(v) for k, v in params.items()})
    st.success("URL updated with current filters.")

