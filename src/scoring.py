import numpy as np
import pandas as pd

# Reasonable/clippable ranges for normalization
RANGES = {
    "fpl_insol": (0.05, 4.0),     # Insolation flux: 1 ~ Earth
    "pl_rade":   (0.5, 3.0),      # Planet radius [Earth radii] (rocky/super-Earth-ish)
    "sy_dist":   (1.0, 2000.0),   # Distance [pc] (closer is easier to study)
    "pl_eqt":    (150.0, 1000.0), # Equilibrium temperature [K]
    "st_teff":   (3000.0, 7500.0) # Stellar temperature [K]; Sun ~ 5772 K
}

DEFAULT_WEIGHTS = {
    # Lower distance from Earth-like insolation is better
    "insolation_proximity": 0.35,
    # Prefer rocky/smaller (within band); radius closer to ~1.0 Re
    "radius_proximity": 0.25,
    # Closer system gets a bonus
    "proximity_bonus": 0.20,
    # Temperate equilibrium temperature (broad window)
    "temp_band_bonus": 0.15,
    # Sunlike star slight bonus
    "sunlike_bonus": 0.05
}

def _norm(val, lo, hi):
    if pd.isna(val):
        return np.nan
    v = np.clip(val, lo, hi)
    return (v - lo) / (hi - lo + 1e-9)

def insolation_proximity(f):
    """Score 0..1: closer to Earth=1 flux is better."""
    if pd.isna(f):
        return np.nan
    lo, hi = RANGES["fpl_insol"]
    f = np.clip(f, lo, hi)
    # distance from 1 on a log-ish scale
    d = abs(np.log10(f) - np.log10(1.0))
    # map d in [0, log10(hi)] to score [1..0]
    dmax = abs(np.log10(hi))
    return max(0.0, 1.0 - d / (dmax + 1e-9))

def radius_proximity(r):
    """Score 0..1: radius closer to 1 Re is better (with falloff)."""
    if pd.isna(r):
        return np.nan
    lo, hi = RANGES["pl_rade"]
    r = np.clip(r, lo, hi)
    d = abs(r - 1.0)
    dmax = max(1.0 - lo, hi - 1.0)
    return max(0.0, 1.0 - d / (dmax + 1e-9))

def proximity_bonus(dist_pc):
    """Closer is better; invert normalized distance."""
    if pd.isna(dist_pc):
        return np.nan
    lo, hi = RANGES["sy_dist"]
    n = _norm(dist_pc, lo, hi)
    return 1.0 - n  # smaller distance → higher score

def temp_band_bonus(eqt_k):
    """Broad 'temperate' window (approx): 200–350 K peaks."""
    if pd.isna(eqt_k):
        return np.nan
    # triangular window centered ~288 K
    center, left, right = 288.0, 200.0, 350.0
    if eqt_k <= left or eqt_k >= right:
        return 0.0
    # piecewise linear peak at center
    if eqt_k <= center:
        return (eqt_k - left) / (center - left)
    else:
        return (right - eqt_k) / (right - center)

def sunlike_bonus(teff_k):
    """Bonus for Sunlike star ~5772 K, gentle falloff."""
    if pd.isna(teff_k):
        return np.nan
    mu = 5772.0
    sigma = 1200.0
    # Gaussian bump 0..1
    return float(np.exp(-0.5 * ((teff_k - mu) / sigma) ** 2))

def compute_score_row(row, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    parts = {
        "insolation_proximity": insolation_proximity(row.get("fpl_insol")),
        "radius_proximity":     radius_proximity(row.get("pl_rade")),
        "proximity_bonus":      proximity_bonus(row.get("sy_dist")),
        "temp_band_bonus":      temp_band_bonus(row.get("pl_eqt")),
        "sunlike_bonus":        sunlike_bonus(row.get("st_teff")),
    }

    # Weighted sum ignoring NaNs (rescale by sum of present weights)
    score = 0.0
    wsum = 0.0
    for k, v in parts.items():
        w = weights.get(k, 0.0)
        if v is not None and not np.isnan(v):
            score += w * v
            wsum += w
    score = (score / wsum) if wsum > 0 else np.nan
    return score * 100.0, parts

def compute_scores(df, weights=None):
    scores = []
    breakdown = []
    for _, row in df.iterrows():
        s, parts = compute_score_row(row, weights=weights)
        scores.append(s)
        breakdown.append(parts)
    out = df.copy()
    out["score"] = scores
    # Optional: add part columns for transparency
    for k in DEFAULT_WEIGHTS.keys():
        out[f"part_{k}"] = [b.get(k, np.nan) if b is not None else np.nan for b in breakdown]
    return out
