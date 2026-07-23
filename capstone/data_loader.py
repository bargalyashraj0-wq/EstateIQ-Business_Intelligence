"""
Shared data-loading utilities for EstateIQ.

Source datasets (from the dsmp-capstone-project notebooks):
- gurgaon_properties_cleaned_v1.csv  -> output of merge + level-2 preprocessing.
  Has the RAW/text columns: society, facing, additionalRoom, furnishDetails,
  features (amenities), nearbyLocations.
- gurgaon_properties_cleaned_v2.csv  -> same pipeline, but with feature-engineering
  applied: split area columns (super_built_up_area/built_up_area/carpet_area),
  binary room flags (study room/servant room/store room/pooja room/others),
  furnishing_type (KMeans cluster id) and luxury_score.
- latlong.csv -> sector -> "lat° N, lon° E" string, used for the geo map.

Both v1 and v2 come from the SAME row-ordered pipeline (verified: identical
price column, sector values match except for minor v2 sub-sector cleanup e.g.
"sector 37c" -> "sector 37"), so we align them by position and take v2's
sector labels (cleaner) plus v1's text columns (features/nearbyLocations)
that v2 dropped.
"""

import ast
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = "datasets"


def _parse_list_string(value):
    """Turn a stringified python list (e.g. "['Park', 'Lift(s)']") into a real list."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_coordinates(coord_str):
    """"28.3663° N, 76.9456° E" -> (28.3663, 76.9456)."""
    if pd.isna(coord_str):
        return None, None
    nums = re.findall(r"[-+]?\d*\.\d+", str(coord_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


@st.cache_data(show_spinner="Loading property data...")
def load_analytics_data():
    """Returns the merged, analysis-ready dataframe plus the sector->lat/lon table."""
    v1 = pd.read_csv(f"{DATA_DIR}/gurgaon_properties_cleaned_v1.csv")
    v2 = pd.read_csv(f"{DATA_DIR}/gurgaon_properties_cleaned_v2.csv")
    latlong = pd.read_csv(f"{DATA_DIR}/latlong.csv")

    # Positional merge: v2's engineered columns + v1's text columns v2 doesn't have.
    df = v2.copy()
    df["facing"] = v1["facing"].values
    df["features_list"] = v1["features"].apply(_parse_list_string).values
    df["nearby_list"] = v1["nearbyLocations"].apply(_parse_list_string).values

    # Numeric coercion (scraped data can hide stray strings in numeric columns).
    numeric_cols = [
        "price", "price_per_sqft", "bedRoom", "bathroom", "floorNum",
        "built_up_area", "super_built_up_area", "carpet_area", "luxury_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Human-readable labels for the raw cluster ids.
    df["furnishing_label"] = df["furnishing_type"].map(
        {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
    )
    # Quantile-based luxury buckets (33rd / 66th percentile split).
    q1, q2 = df["luxury_score"].quantile([0.33, 0.66])
    df["luxury_category"] = pd.cut(
        df["luxury_score"],
        bins=[-1, q1, q2, df["luxury_score"].max()],
        labels=["Low", "Medium", "High"],
    )
    df["property_type"] = df["property_type"].str.title()

    # Coordinates.
    latlong[["lat", "lon"]] = latlong["coordinates"].apply(
        lambda x: pd.Series(_parse_coordinates(x))
    )

    return df, latlong


def get_sector_geo_stats(df: pd.DataFrame, latlong: pd.DataFrame) -> pd.DataFrame:
    """Sector-level aggregates joined to coordinates, for the map view."""
    sector_stats = (
        df.groupby("sector")
        .agg(
            avg_price=("price", "mean"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            median_price=("price", "median"),
            listing_count=("price", "size"),
        )
        .reset_index()
    )
    merged = sector_stats.merge(latlong[["sector", "lat", "lon"]], on="sector", how="inner")
    return merged.dropna(subset=["lat", "lon"])


def get_amenity_frequencies(df: pd.DataFrame) -> Counter:
    all_features = [f for feats in df["features_list"] for f in feats]
    return Counter(all_features)


def get_nearby_location_frequencies(df: pd.DataFrame) -> Counter:
    all_locations = [loc for locs in df["nearby_list"] for loc in locs]
    return Counter(all_locations)


def clip_for_viz(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Clip a numeric series to its 1st-99th percentile so a handful of bad
    scraped rows (e.g. area=58228 sqft for a 2BHK) don't blow out chart axes.
    Only used for plotting - never mutates the underlying counts/KPIs."""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)
