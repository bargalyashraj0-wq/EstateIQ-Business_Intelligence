import ast
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

# set_page_config can only be called ONCE, and must be the very first
# Streamlit command - so it lives here, not inside any individual module.
st.set_page_config(page_title="EstateIQ", page_icon="🏠", layout="wide")


# ============================================================================
# SHARED HELPERS (previously in data_loader.py — inlined so this is one file)
# ============================================================================
def clip_for_viz(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Clip a numeric series to its 1st-99th percentile so a handful of bad
    scraped rows (e.g. area=58228 sqft for a 2BHK) don't blow out chart axes."""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


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


# ============================================================================
# MODULE 1 — ANALYTICS
# ============================================================================
def analytics_module():
    st.title("📊 Analytics Module")
    st.caption("Explore Gurgaon's real-estate market: prices, sectors, amenities, and trends.")

    df_raw = pd.read_csv("gurgaon_properties_cleaned_v2.csv")
    latlong = pd.read_csv("latlong.csv")
    latlong["coordinates"] = latlong["coordinates"].str.replace("°", "", regex=False)
    latlong[["lat", "lon"]] = (
        latlong["coordinates"]
        .str.extract(r"([-+]?\d*\.?\d+)\D+([-+]?\d*\.?\d+)")
        .astype(float)
    )
    df1_raw = pd.read_csv("gurgaon_properties_cleaned_v1.csv")

    st.sidebar.header("🔎 Analytics Filters")

    sectors = sorted(df_raw["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", sectors, default=[])

    property_types = sorted(df_raw["property_type"].dropna().unique())
    selected_types = st.sidebar.multiselect("Property Type", property_types, default=property_types)

    bed_min, bed_max = int(df_raw["bedRoom"].min()), int(df_raw["bedRoom"].max())
    bed_range = st.sidebar.slider("Bedrooms (BHK)", bed_min, bed_max, (bed_min, bed_max))

    price_min, price_max = float(df_raw["price"].min()), float(df_raw["price"].max())
    price_range = st.sidebar.slider("Price (₹ Crore)", price_min, price_max, (price_min, price_max))

    df = df_raw.copy()
    if selected_sectors:
        df = df[df["sector"].isin(selected_sectors)]
    if selected_types:
        df = df[df["property_type"].isin(selected_types)]
    df = df[df["bedRoom"].between(*bed_range) & df["price"].between(*price_range)]

    df1 = df1_raw.copy()
    if selected_sectors:
        df1 = df1[df1["sector"].isin(selected_sectors)]
    if selected_types:
        df1 = df1[df1["property_type"].isin(selected_types)]
    df1 = df1[df1["bedRoom"].between(*bed_range) & df1["price"].between(*price_range)]

    st.sidebar.markdown(f"**{len(df):,}** properties match your filters")

    if df.empty:
        st.warning("No properties match the current filters. Try widening your selection.")
        st.stop()

    tab_overview, tab_geo, tab_price, tab_features, tab_amenities, tab_corr = st.tabs(
        [" Overview", " Geography", " Price Trends", " Property Features",
         " Amenities & Location", " Correlations"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Listings", f"{len(df):,}")
        c2.metric("Avg. Price", f"₹{df['price'].mean():.2f} Cr")
        c3.metric("Avg. Price/Sqft", f"₹{df['price_per_sqft'].mean():,.0f}")
        c4.metric("Sectors Covered", df["sector"].nunique())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Property Type Split**")
            type_counts = df["property_type"].value_counts().reset_index()
            type_counts.columns = ["property_type", "count"]
            fig = px.pie(type_counts, names="property_type", values="count", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 10 Sectors by Listing Count**")
            top_sectors = df["sector"].value_counts().head(10).reset_index()
            top_sectors.columns = ["sector", "count"]
            fig = px.bar(top_sectors, x="count", y="sector", orientation="h")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with tab_geo:
        geo_metric = st.radio(
            "Color by", ["avg_price", "avg_price_per_sqft", "listing_count"],
            format_func=lambda x: {
                "avg_price": "Avg. Price (Cr)",
                "avg_price_per_sqft": "Avg. Price/Sqft",
                "listing_count": "Listing Count",
            }[x],
            horizontal=True,
        )
        geo_df = get_sector_geo_stats(df, latlong)
        if geo_df.empty:
            st.info("None of the filtered sectors have matching coordinates in latlong.csv.")
        else:
            fig = px.scatter_mapbox(
                geo_df, lat="lat", lon="lon", size="listing_count", color=geo_metric,
                hover_name="sector",
                hover_data={"avg_price": ":.2f", "avg_price_per_sqft": ":.0f",
                            "listing_count": True, "lat": False, "lon": False},
                color_continuous_scale="Viridis", size_max=30, zoom=10,
                mapbox_style="carto-positron", height=550,
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Showing {len(geo_df)} of {df['sector'].nunique()} filtered sectors — "
                "sectors without a 'sector N' style name (e.g. colony names) aren't in latlong.csv."
            )

    with tab_price:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Price Distribution (₹ Crore)**")
            fig = px.histogram(df, x="price", nbins=50, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Price per Sqft Distribution**")
            viz_series = clip_for_viz(df["price_per_sqft"].dropna())
            fig = px.histogram(viz_series, nbins=50, marginal="box")
            fig.update_layout(xaxis_title="Price per Sqft (clipped at 1st/99th pct)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Price by Property Type**")
        fig = px.box(df, x="property_type", y="price", color="property_type", points=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Price by BHK**")
        fig = px.box(df.sort_values("bedRoom"), x="bedRoom", y="price", points=False)
        fig.update_layout(xaxis_title="Bedrooms (BHK)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Price Comparison Across Top 10 Sectors (by listing count)**")
        top10 = df["sector"].value_counts().head(10).index
        fig = px.box(df[df["sector"].isin(top10)], x="sector", y="price", points=False)
        fig.update_layout(xaxis_title="Sector")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Built-up Area vs Price**")
        scatter_df = df.dropna(subset=["built_up_area", "price"]).copy()
        scatter_df["built_up_area"] = clip_for_viz(scatter_df["built_up_area"])
        fig = px.scatter(
            scatter_df, x="built_up_area", y="price", color="bedRoom",
            hover_data=["sector", "society"], opacity=0.6,
        )
        fig.update_layout(xaxis_title="Built-up Area (sqft, clipped)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_features:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Luxury Score Distribution**")
            fig = px.histogram(
                df, x="luxury_score", nbins=20,
                title="Distribution of Luxury Scores",
                labels={"luxury_score": "Luxury Score", "count": "Number of Properties"},
            )
            fig.update_layout(width=500, height=500, autosize=False)
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            st.markdown("**Furnishing Status**")
            furn_counts = df["furnishing_type"].value_counts().reset_index()
            furn_counts.columns = ["furnishing", "count"]
            fig = px.pie(
                furn_counts, names="furnishing", values="count", hole=0.4,
                labels={"furnishing": "Category", "count": "Properties"},
            )
            fig.update_traces(textinfo="percent+label", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("`0` Unfurnished &nbsp;&nbsp;|&nbsp;&nbsp; `1` Semi-Furnished &nbsp;&nbsp;|&nbsp;&nbsp; `2` Furnished")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Facing Direction**")
            facing_counts = df["facing"].value_counts().reset_index()
            facing_counts.columns = ["facing", "count"]
            fig = px.bar(facing_counts, x="facing", y="count")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("**Extra Room Availability**")
            room_cols = ["study room", "servant room", "store room", "pooja room"]
            room_pct = (df[room_cols].sum() / len(df) * 100).reset_index()
            room_pct.columns = ["room_type", "pct_of_listings"]
            fig = px.bar(room_pct, x="room_type", y="pct_of_listings")
            fig.update_layout(yaxis_title="% of listings with this room")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Luxury Score vs Price per Sqft**")
        fig = px.scatter(
            df.dropna(subset=["luxury_score", "price_per_sqft"]),
            x="luxury_score", y="price_per_sqft", color="property_type", opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_amenities:
        st.markdown("**Most Common Amenities**")
        amenity_freq = Counter(
            f for feats in df1["features"].fillna("[]").apply(ast.literal_eval) for f in feats
        )
        if amenity_freq:
            wc = WordCloud(width=1000, height=400, background_color="white",
                            colormap="viridis").generate_from_frequencies(amenity_freq)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            top_amenities = pd.DataFrame(amenity_freq.most_common(15), columns=["amenity", "count"])
            fig2 = px.bar(top_amenities, x="count", y="amenity", orientation="h")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No amenity data available for the current filter selection.")

        st.markdown("**Most Frequently Mentioned Nearby Landmarks**")
        nearby_freq = Counter(
            loc for locs in df1["nearbyLocations"].fillna("[]").apply(ast.literal_eval) for loc in locs
        )
        if nearby_freq:
            top_nearby = pd.DataFrame(nearby_freq.most_common(15), columns=["landmark", "mentions"])
            fig3 = px.bar(top_nearby, x="mentions", y="landmark", orientation="h")
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No nearby-location data available for the current filter selection.")

    with tab_corr:
        st.markdown("**Correlation Heatmap (numeric features)**")
        numeric_cols = ["price", "price_per_sqft", "bedRoom", "bathroom", "built_up_area",
                         "luxury_score", "floorNum"]
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Values close to +1 or -1 indicate strong relationships. "
            "See the **Insights** module for model-driven feature importance."
        )


# ============================================================================
# MODULE 2 — RECOMMENDER
# ============================================================================
def recommender_module():
    st.title("🏠 Property Recommender")
    st.caption(
        "Pick a property you like, and we'll find similar ones based on "
        "facilities, price/area, and nearby locations."
    )

    MISSING_DISTANCE_M = 54000

    @st.cache_data
    def build_similarity_data(path="appartments.csv"):
        df = pd.read_csv(path)
        df = df.dropna(subset=["PropertyName"]).drop_duplicates(subset="PropertyName")
        df = df.reset_index(drop=True)

        def get_facilities_text(raw):
            facilities = re.findall(r"'(.*?)'", raw) if isinstance(raw, str) else []
            return " ".join(facilities)

        facilities_text = df["TopFacilities"].apply(get_facilities_text)
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(facilities_text)
        sim_facilities = cosine_similarity(tfidf_matrix)

        def parse_price_row(raw):
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
                    row[f"area_{bhk}"] = sum(area_nums) / len(area_nums)
                price_nums = []
                for p in str(info.get("price-range", "")).split("-"):
                    digits = re.sub(r"[^\d.]", "", p)
                    if not digits:
                        continue
                    value = float(digits)
                    if "L" in p:
                        value /= 100
                    price_nums.append(value)
                if price_nums:
                    row[f"price_{bhk}"] = sum(price_nums) / len(price_nums)
            return row

        price_df = pd.DataFrame([parse_price_row(v) for v in df["PriceDetails"]])
        type_cols = [c for c in price_df.columns if c.startswith("type_")]
        price_df = pd.get_dummies(price_df, columns=type_cols, drop_first=True).fillna(0)
        price_scaled = StandardScaler().fit_transform(price_df)
        sim_price = cosine_similarity(price_scaled)

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

    st.sidebar.header("🔎 Recommend Based On")
    selected_property = st.sidebar.selectbox("Choose a property", sorted(property_names))
    top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

    st.sidebar.markdown("**Weight each signal**")
    w_facilities = st.sidebar.slider("Facilities & amenities", 0.0, 1.0, 0.5, 0.05)
    w_price = st.sidebar.slider("Price & area", 0.0, 1.0, 0.3, 0.05)
    w_location = st.sidebar.slider("Nearby locations", 0.0, 1.0, 0.2, 0.05)
    total_weight = max(w_facilities + w_price + w_location, 0.01)

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


# ============================================================================
# MODULE 3 — INSIGHTS
# ============================================================================
def insights_module():
    st.title("📈 Insights Module")
    st.caption(
        "A data-driven report on Gurgaon's real-estate market: what actually moves "
        "price, which amenities are worth paying for, and where the value pockets are."
    )

    try:
        df_raw = pd.read_csv("gurgaon_properties_cleaned_v2.csv")
        df1_raw = pd.read_csv("gurgaon_properties_cleaned_v1.csv")
        model_raw = pd.read_csv("gurgaon_properties_post_feature_selection_v2.csv")
    except FileNotFoundError as e:
        st.error(
            f"Couldn't find one of the required data files ({e.filename}). "
            "Make sure `gurgaon_properties_cleaned_v1.csv`, `gurgaon_properties_cleaned_v2.csv`, "
            "and `gurgaon_properties_post_feature_selection_v2.csv` all sit alongside the other "
            "project data files."
        )
        st.stop()

    model_df_base = model_raw.drop(
        columns=[c for c in ["store room", "floor_category", "balcony"] if c in model_raw.columns]
    )

    st.sidebar.header("🔎 Insights Filters")

    sectors = sorted(df_raw["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", sectors, default=[])

    property_types = sorted(df_raw["property_type"].dropna().unique())
    selected_types = st.sidebar.multiselect("Property Type", property_types, default=property_types)

    bed_min, bed_max = int(df_raw["bedRoom"].min()), int(df_raw["bedRoom"].max())
    bed_range = st.sidebar.slider("Bedrooms (BHK)", bed_min, bed_max, (bed_min, bed_max))

    price_min, price_max = float(df_raw["price"].min()), float(df_raw["price"].max())
    price_range = st.sidebar.slider("Price (₹ Crore)", price_min, price_max, (price_min, price_max))

    def filter_descriptive(data):
        out = data.copy()
        if selected_sectors:
            out = out[out["sector"].isin(selected_sectors)]
        if selected_types:
            out = out[out["property_type"].isin(selected_types)]
        return out[out["bedRoom"].between(*bed_range) & out["price"].between(*price_range)]

    df = filter_descriptive(df_raw)
    df1 = filter_descriptive(df1_raw)

    # property_type in post_feature_selection_v2.csv is text ('flat'/'house'),
    # same as the other CSVs - no numeric-code translation needed here.
    model_df = model_df_base.copy()
    if selected_sectors:
        model_df = model_df[model_df["sector"].isin(selected_sectors)]
    if selected_types:
        model_df = model_df[model_df["property_type"].isin(selected_types)]
    model_df = model_df[model_df["bedRoom"].between(*bed_range) & model_df["price"].between(*price_range)]

    st.sidebar.markdown(f"**{len(df):,}** properties match your filters")

    if df.empty or model_df.empty:
        st.warning("No properties match the current filters. Try widening your selection.")
        st.stop()

    tab_model, tab_amenities, tab_sector, tab_report = st.tabs(
        [" Price Drivers", " Amenity Value", " Sector Value Map", " Full Report"]
    )

    with tab_model:
        st.markdown("### What Actually Drives Price")
        st.caption(
            "A Ridge regression on log(price), trained on the currently filtered listings. "
            "Coefficients are on standardized features, so they're directly comparable — "
            "bigger magnitude = bigger effect on price."
        )

        required_cols = {
            "bedRoom", "bathroom", "built_up_area", "servant room", "property_type",
            "furnishing_type", "luxury_category", "sector", "agePossession", "price",
        }
        missing = required_cols - set(model_df.columns)
        r2_mean = r2_std = mae_cr = None
        coef_df = pd.DataFrame(columns=["feature", "coef", "abs_coef"])

        if missing:
            st.error(f"Model data is missing expected columns: {sorted(missing)}")
        elif len(model_df) < 40:
            st.info(
                "Not enough listings under the current filters to fit a reliable model "
                "(need at least 40). Widen the filters to see this section."
            )
        else:
            m = model_df.copy()
            m["luxury_category"] = m["luxury_category"].replace({"Low": 0, "Medium": 1, "High": 2})
            m["agePossession"] = m["agePossession"].replace({
                "Relatively New": "new", "New Property": "new",
                "Moderately Old": "old", "Old Property": "old",
                "Under Construction": "under construction",
            })

            encoded = pd.get_dummies(m, columns=["sector", "agePossession", "property_type"], drop_first=True)
            X = encoded.drop(columns=["price"])
            y_log = np.log1p(encoded["price"])

            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

            with st.spinner("Fitting model..."):
                kfold = KFold(n_splits=min(10, max(3, len(X) // 20)), shuffle=True, random_state=42)
                cv_scores = cross_val_score(Ridge(alpha=0.0001), X_scaled, y_log, cv=kfold, scoring="r2")
                r2_mean, r2_std = cv_scores.mean(), cv_scores.std()

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
                ridge = Ridge(alpha=0.0001)
                ridge.fit(X_train, y_train)
                pred_log = ridge.predict(X_test)
                mae_cr = mean_absolute_error(np.expm1(y_test), np.expm1(pred_log))

                ridge_full = Ridge(alpha=0.0001)
                ridge_full.fit(X_scaled, y_log)

            coef_df = (
                pd.DataFrame({"feature": X.columns, "coef": ridge_full.coef_})
                .assign(abs_coef=lambda d: d["coef"].abs())
                .sort_values("abs_coef", ascending=False)
            )
            coef_df["group"] = coef_df["feature"].apply(
                lambda f: "sector" if f.startswith("sector_")
                else "agePossession" if f.startswith("agePossession_")
                else f
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Cross-validated R²", f"{r2_mean:.2f}", help=f"±{r2_std:.2f} across folds")
            c2.metric("Typical Error (holdout)", f"₹{mae_cr:.2f} Cr")
            c3.metric("Features Used", f"{X.shape[1]}")

            top_features = coef_df[~coef_df["feature"].str.startswith(("sector_", "agePossession_"))].head(10)
            fig = px.bar(
                top_features.sort_values("coef"), x="coef", y="feature", orientation="h",
                color="coef", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                title="Top Non-Location Price Drivers (standardized coefficient)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Positive bars push price up, negative bars pull it down, for a 1 standard-"
                "deviation increase in that feature (with everything else held constant). "
                "Sector and possession-age dummies are excluded from this chart to keep it "
                "readable — see the Full Report tab for the sector effect summary."
            )

            with st.expander("Show statistical significance (OLS summary)"):
                X_const = sm.add_constant(X_scaled)
                ols_model = sm.OLS(y_log, X_const).fit()
                sig = (
                    pd.DataFrame({
                        "feature": ols_model.params.index,
                        "coef": ols_model.params.values,
                        "p_value": ols_model.pvalues.values,
                    })
                    .query("feature != 'const'")
                    .assign(significant=lambda d: d["p_value"] < 0.05)
                    .sort_values("p_value")
                )
                st.dataframe(sig, use_container_width=True, hide_index=True)
                st.caption(f"Model adjusted R² = {ols_model.rsquared_adj:.3f} · n = {int(ols_model.nobs)}")

    with tab_amenities:
        st.markdown("### Which Amenities Actually Carry a Price Premium")
        st.caption(
            "For each amenity, this compares the average price of listings that have it "
            "against listings that don't — a simple but concrete measure of what buyers "
            "are actually paying for, as opposed to just what's commonly advertised."
        )

        if "features" not in df1.columns or "price" not in df1.columns:
            st.info("Amenity/price data not available for the current filter selection.")
            lift_df = pd.DataFrame()
        else:
            feats_parsed = df1["features"].fillna("[]").apply(ast.literal_eval)
            overall_avg_price = df1["price"].mean()

            exploded = df1.assign(_feat=feats_parsed).explode("_feat").dropna(subset=["_feat"])
            lift = exploded.groupby("_feat")["price"].agg(["mean", "count"]).reset_index()
            lift.columns = ["amenity", "avg_price_with", "listing_count"]
            lift = lift[lift["listing_count"] >= max(15, int(len(df1) * 0.02))]
            lift["premium_pct"] = (lift["avg_price_with"] - overall_avg_price) / overall_avg_price * 100
            lift_df = lift.sort_values("premium_pct", ascending=False)

            if lift_df.empty:
                st.info("No amenity appears often enough under the current filters to compare reliably.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Biggest price premium**")
                    top = lift_df.head(10).sort_values("premium_pct")
                    fig = px.bar(top, x="premium_pct", y="amenity", orientation="h",
                                 labels={"premium_pct": "% above average price"})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("**Biggest price discount**")
                    bottom = lift_df.tail(10).sort_values("premium_pct")
                    fig = px.bar(bottom, x="premium_pct", y="amenity", orientation="h",
                                 labels={"premium_pct": "% vs average price"})
                    st.plotly_chart(fig, use_container_width=True)

                best_amenity = lift_df.iloc[0]
                worst_amenity = lift_df.iloc[-1]
                st.markdown(
                    f"Listings with **{best_amenity['amenity']}** average "
                    f"**{best_amenity['premium_pct']:+.0f}%** vs. the overall average price, "
                    f"the largest premium of any common amenity. Listings with "
                    f"**{worst_amenity['amenity']}** average **{worst_amenity['premium_pct']:+.0f}%**, "
                    "the weakest association — worth noting this is a correlation, not proof the "
                    "amenity itself causes the price gap (it may just cluster in cheaper sectors)."
                )

    with tab_sector:
        st.markdown("### Sector Value Map")
        st.caption(
            "Sectors plotted by average luxury score (x) vs. average price/sqft (y). "
            "Sectors below the trend line offer more luxury per rupee than the market average; "
            "sectors above it command a premium beyond what their luxury score would predict."
        )

        sector_stats = (
            df.groupby("sector")
            .agg(
                avg_price_per_sqft=("price_per_sqft", "mean"),
                avg_luxury_score=("luxury_score", "mean"),
                listing_count=("price", "size"),
            )
            .reset_index()
        )
        sector_stats = sector_stats[sector_stats["listing_count"] >= 3]

        if len(sector_stats) < 5:
            st.info("Not enough sectors with 3+ listings under the current filter to map value pockets.")
            best_value, most_premium = [], []
        else:
            fig = px.scatter(
                sector_stats, x="avg_luxury_score", y="avg_price_per_sqft", size="listing_count",
                hover_name="sector", trendline="ols", trendline_color_override="gray",
            )
            st.plotly_chart(fig, use_container_width=True)

            coeffs = np.polyfit(sector_stats["avg_luxury_score"], sector_stats["avg_price_per_sqft"], 1)
            predicted = np.polyval(coeffs, sector_stats["avg_luxury_score"])
            sector_stats["residual"] = sector_stats["avg_price_per_sqft"] - predicted

            best_value = sector_stats.nsmallest(3, "residual")["sector"].tolist()
            most_premium = sector_stats.nlargest(3, "residual")["sector"].tolist()

            st.markdown(
                f"**Best value:** {', '.join(best_value)} — priced below what their luxury "
                f"score would predict.  \n"
                f"**Biggest premium:** {', '.join(most_premium)} — priced above prediction, "
                "likely paying for location or brand rather than fit-out quality."
            )

    with tab_report:
        st.markdown("### Market Insights Report")
        st.caption(f"Generated from {len(df):,} filtered listings across {df['sector'].nunique()} sectors.")

        lines = [f"# EstateIQ Market Insights Report\n{len(df):,} listings, {df['sector'].nunique()} sectors\n"]

        avg_price, med_price = df["price"].mean(), df["price"].median()
        avg_psf = df["price_per_sqft"].mean()
        summary = (
            f"The filtered market covers **{len(df):,} listings** averaging **₹{avg_price:.2f} Cr** "
            f"(median ₹{med_price:.2f} Cr) at **₹{avg_psf:,.0f}/sqft**."
        )
        if r2_mean is not None:
            summary += (
                f" A regression model explains **{r2_mean:.0%}** of price variation "
                f"(R² = {r2_mean:.2f} ± {r2_std:.2f}), with a typical error of ₹{mae_cr:.2f} Cr — "
                "useful for understanding what matters, not for exact valuation."
            )
        st.markdown("**1. Executive Summary**  \n" + summary)
        lines.append("## 1. Executive Summary\n" + summary)

        if not coef_df.empty:
            top_drivers = coef_df[~coef_df["feature"].str.startswith(("sector_", "agePossession_"))].head(3)
            driver_text = ", ".join(f"{r.feature} ({r.coef:+.2f})" for r in top_drivers.itertuples())
            st.markdown(f"**2. Price Drivers**  \nTop factors: {driver_text}. See the Price Drivers tab for the full breakdown and statistical significance.")
            lines.append(f"## 2. Price Drivers\nTop factors: {driver_text}")

        if not lift_df.empty:
            amenity_text = (
                f"{best_amenity['amenity']} carries the largest measured premium "
                f"({best_amenity['premium_pct']:+.0f}%); {worst_amenity['amenity']} the weakest "
                f"({worst_amenity['premium_pct']:+.0f}%)."
            )
            st.markdown(f"**3. Amenity Value**  \n{amenity_text}")
            lines.append(f"## 3. Amenity Value\n{amenity_text}")

        if best_value:
            sector_text = f"Best value sectors: {', '.join(best_value)}. Highest premium sectors: {', '.join(most_premium)}."
            st.markdown(f"**4. Sector Value**  \n{sector_text}")
            lines.append(f"## 4. Sector Value\n{sector_text}")

        st.markdown(
            "**5. Recommendations**  \n"
            "- Value buyers: start with the best-value sectors above — comparable luxury "
            "for a lower price/sqft than the market trend implies.  \n"
            "- Don't overweight amenities that are now standard — check the discount list "
            "before assuming an amenity is a differentiator.  \n"
            "- Narrow the sidebar filters to your actual budget/BHK before trusting the "
            "model's error margin for a specific search."
        )
        lines.append(
            "## 5. Recommendations\n"
            "- Start with the best-value sectors identified above.\n"
            "- Don't overweight amenities that are now standard.\n"
            "- Narrow filters to your actual budget/BHK before trusting the model's error margin."
        )

        st.download_button(
            "⬇️ Download this report as Markdown",
            data="\n\n".join(lines),
            file_name="estateiq_insights_report.md",
            mime="text/markdown",
        )


# ============================================================================
# MAIN — module switcher
# ============================================================================
st.sidebar.title("🏠 EstateIQ")
module = st.sidebar.selectbox(
    "Choose a module",
    ["📊 Analytics", "🏠 Recommender", "📈 Insights"],
)
st.sidebar.markdown("---")

if module == "📊 Analytics":
    analytics_module()
elif module == "🏠 Recommender":
    recommender_module()
else:
    insights_module()
