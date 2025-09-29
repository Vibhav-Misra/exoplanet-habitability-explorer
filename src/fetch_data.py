import argparse
import io
import os
import sys
from urllib.parse import urlencode

import pandas as pd
import requests


# Columns to fetch (names from PSCompPars)
COLUMNS = [
    "pl_name",
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_eqt",
    "pl_insol",        # <-- fixed
    "st_teff",
    "st_rad",
    "st_mass",
    "sy_dist",
    "ra", "dec",
    "discoverymethod",
    "disc_year"
]

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def build_query(limit=None):
    cols = ", ".join(COLUMNS)
    if limit:
        # ADQL-friendly
        return f"SELECT TOP {int(limit)} {cols} FROM pscomppars ORDER BY disc_year DESC"
    return f"SELECT {cols} FROM pscomppars"

def fetch_pscomppars(limit=None, timeout=60):
    query = build_query(limit=limit)
    params = {
        "query": query,
        "format": "csv",
    }
    # Alternatively, you can skip TOP and instead do: if limit: params["maxrec"] = int(limit)
    if limit:
        params["maxrec"] = int(limit)  # harmless alongside TOP; or remove TOP and rely on this
    resp = requests.get(TAP_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric columns are numeric (coerce to NaN on errors)
    numeric_cols = [c for c in df.columns if c not in ("pl_name", "discoverymethod")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser(description="Fetch PSCompPars from NASA Exoplanet Archive")
    ap.add_argument("--out", default="data/latest.parquet", help="Output parquet file")
    ap.add_argument("--csv", default="data/latest.csv", help="Optional CSV dump alongside parquet")
    ap.add_argument("--limit", type=int, default=None, help="Row limit for fast dev (e.g., 200)")
    args = ap.parse_args()

    print("Fetching PSCompPars…")
    df = fetch_pscomppars(limit=args.limit)
    print(f"Fetched {len(df):,} rows.")
    df = coerce_types(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved parquet → {args.out}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved CSV     → {args.csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
