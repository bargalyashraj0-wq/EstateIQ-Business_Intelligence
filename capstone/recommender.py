import ast
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Recommender | EstateIQ", page_icon="🏠", layout="wide")
st.title("🏠 Property Recommender")
st.caption(
    "Pick a property you like, and we'll find similar ones based on "
    "facilities, price/area, and nearby locations."
)

# A property with no distance to a given landmark is treated as "far away"
# rather than 0, so it doesn't look artificially close to other properties.
MISSING_DISTANCE_M = 54000


# ----------------------------------------------------------------------------
# BUILD SIMILARITY MATRICES
# Wrapped in one cached function: Streamlit reruns this whole script on every
# click, and this step (TF-IDF + one-hot + cosine similarity) is the slow
# part. Caching means it only actually runs once, not on every interaction.
# ----------------------------------------------------------------------------
@st.cache_data
def build_similarity_data(path="appartments.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["PropertyName"]).drop_duplicates(subset="PropertyName")
    df = df.reset_index(drop=True)

    # ---- 1. Facilities similarity (TF-IDF over the facilities text) ----
    def get_facilities_text(raw):
        # TopFacilities is a stringified list, e.g. "['Lift', 'Gym']"
        facilities = re.findall(r"'(.*?)'", raw) if isinstance(raw, str) else []
        return " ".join(facilities)

    facilities_text = df["TopFacilities"].apply(get_facilities_text)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(facilities_text)
    sim_facilities = cosine_similarity(tfidf_matrix)

    # ---- 2. Price/area similarity (per-BHK numeric features) ----
    def parse_price_row(raw):
        # PriceDetails is a stringified dict, e.g.
        # "{'2 BHK': {'building_type': 'Apartment', 'area': '1,370 sq.ft.',
        #             'price-range': '₹ 2 - 2.4 Cr'}}"
        # A couple of rows in real scraped data are just plain broken
        # (e.g. one row here literally contains the text "PriceDetails"
        # instead of a dict) — this try/except lets us skip those rows
        # instead of crashing the whole app.
        try:
            details = ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError):
            return {}

        row = {}
        for bhk, info in details.items():
            if not isinstance(info, dict):
                continue
            row[f"type_{bhk}"] = info.get("building_type") or "NA"

            area_text = re.sub(r"sq\.?\s?ft\.?", "", str(info.get("area", "")), flags=re.IGNORECASE)
            area_nums = [float(re.sub(r"[^\d.]", "", p)) for p in area_text.split("-") if re.search(r"\d", p)]
            if area_nums:
                row[f"area_{bhk}"] = sum(area_nums) / len(area_nums)  # midpoint

            price_nums = []
            for p in str(info.get("price-range", "")).split("-"):
                digits = re.sub(r"[^\d.]", "", p)
                if not digits:
                    continue
                value = float(digits)
                if "L" in p:  # Lakh -> Crore
                    value /= 100
                price_nums.append(value)
            if price_nums:
                row[f"price_{bhk}"] = sum(price_nums) / len(price_nums)  # midpoint

        return row

    price_df = pd.DataFrame([parse_price_row(v) for v in df["PriceDetails"]])
    type_cols = [c for c in price_df.columns if c.startswith("type_")]
    price_df = pd.get_dummies(price_df, columns=type_cols, drop_first=True).fillna(0)
    price_scaled = StandardScaler().fit_transform(price_df)
    sim_price = cosine_similarity(price_scaled)

    # ---- 3. Nearby-location similarity (distance to landmarks) ----
    def distance_to_meters(text):
        match = re.search(r"([\d.]+)", str(text))
        if not match:
            return None
        value = float(match.group(1))
        return value * 1000 if "km" in str(text).lower() else value

    def parse_location_row(raw):
        try:
            landmarks = ast.literal_eval(raw) if isinstance(raw, str) else {}
        except (ValueError, SyntaxError):
            return {}
        return {k: distance_to_meters(v) for k, v in landmarks.items()}

    location_df = pd.DataFrame([parse_location_row(v) for v in df["LocationAdvantages"]])
    location_df = location_df.fillna(MISSING_DISTANCE_M)
    location_scaled = StandardScaler().fit_transform(location_df)
    sim_location = cosine_similarity(location_scaled)

    return df["PropertyName"].tolist(), sim_facilities, sim_price, sim_location


try:
    property_names, sim_facilities, sim_price, sim_location = build_similarity_data()
except FileNotFoundError:
    st.error(
        "Couldn't find `appartments.csv` in the project folder. "
        "Add it alongside your other data files to use this module."
    )
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Recommend Based On")
selected_property = st.sidebar.selectbox("Choose a property", sorted(property_names))
top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

st.sidebar.markdown("**Weight each signal**")
w_facilities = st.sidebar.slider("Facilities & amenities", 0.0, 1.0, 0.5, 0.05)
w_price = st.sidebar.slider("Price & area", 0.0, 1.0, 0.3, 0.05)
w_location = st.sidebar.slider("Nearby locations", 0.0, 1.0, 0.2, 0.05)
total_weight = max(w_facilities + w_price + w_location, 0.01)  # avoid /0

# ----------------------------------------------------------------------------
# RECOMMEND
# ----------------------------------------------------------------------------
idx = property_names.index(selected_property)
combined_sim = (
    w_facilities * sim_facilities + w_price * sim_price + w_location * sim_location
) / total_weight

scores = [(i, score) for i, score in enumerate(combined_sim[idx]) if i != idx]
scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

results = pd.DataFrame({
    "PropertyName": [property_names[i] for i, _ in scores],
    "SimilarityScore": [round(s, 3) for _, s in scores],
})

st.subheader(f"Properties similar to: {selected_property}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Ranked list**")
    st.dataframe(results, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Similarity scores**")
    fig = px.bar(results.sort_values("SimilarityScore"), x="SimilarityScore", y="PropertyName", orientation="h")
    fig.update_layout(yaxis_title="", xaxis_title="Similarity Score")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "How it works: each property is compared on three signals — listed "
    "facilities (text similarity), price/area by BHK config, and distance "
    "to nearby landmarks — then blended using the weights above."
)