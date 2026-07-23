import ast
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Insights | EstateIQ", page_icon="📈", layout="wide")
st.title("📈 Insights Module")
st.caption(
    "A data-driven report on Gurgaon's real-estate market: what actually moves "
    "price, which amenities are worth paying for, and where the value pockets are."
)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
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

# post_feature_selection_v2 is the model-ready table: numeric + ordinal-encoded
# columns only, no raw text. Drop the columns the notebook found redundant.
model_df_base = model_raw.drop(
    columns=[c for c in ["store room", "floor_category", "balcony"] if c in model_raw.columns]
)

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS — applied to both the descriptive data and the model data
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")


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

# property_type in the model table is coded 0/1 (flat/house); translate the
# string selections from the sidebar into that coding so the same filters apply.
type_code_map = {"flat": 0, "house": 1}
selected_type_codes = [type_code_map[t] for t in selected_types if t in type_code_map]

model_df = model_df_base.copy()
if selected_sectors:
    model_df = model_df[model_df["sector"].isin(selected_sectors)]
if selected_type_codes:
    model_df = model_df[model_df["property_type"].isin(selected_type_codes)]
model_df = model_df[model_df["bedRoom"].between(*bed_range) & model_df["price"].between(*price_range)]

st.sidebar.markdown(f"**{len(df):,}** properties match your filters")

if df.empty or model_df.empty:
    st.warning("No properties match the current filters. Try widening your selection.")
    st.stop()

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_model, tab_amenities, tab_sector, tab_report = st.tabs(
    [" Price Drivers", " Amenity Value", " Sector Value Map", " Full Report"]
)

# ============================================================================
# TAB 1 — REGRESSION MODEL: WHAT ACTUALLY DRIVES PRICE
# Mirrors the notebook's approach (log price, standardized features, OHE
# sector/agePossession, Ridge regression) but surfaces it as readable output.
# ============================================================================
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
        # property_type and furnishing_type are already numerically coded upstream;
        # only luxury_category still needs an ordinal mapping.
        m["luxury_category"] = m["luxury_category"].replace({"Low": 0, "Medium": 1, "High": 2})
        m["agePossession"] = m["agePossession"].replace({
            "Relatively New": "new", "New Property": "new",
            "Moderately Old": "old", "Old Property": "old",
            "Under Construction": "under construction",
        })

        encoded = pd.get_dummies(m, columns=["sector", "agePossession"], drop_first=True)
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

            # refit on full filtered set for the coefficient table shown to the user
            ridge_full = Ridge(alpha=0.0001)
            ridge_full.fit(X_scaled, y_log)

        coef_df = (
            pd.DataFrame({"feature": X.columns, "coef": ridge_full.coef_})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
        )
        # Collapse one-hot sector/agePossession dummies into readable group labels
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

# ============================================================================
# TAB 2 — AMENITY VALUE: WHICH AMENITIES CARRY A REAL PRICE PREMIUM
# ============================================================================
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
        lift = lift[lift["listing_count"] >= max(15, int(len(df1) * 0.02))]  # drop noisy rare amenities
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

# ============================================================================
# TAB 3 — SECTOR VALUE MAP
# ============================================================================
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

# ============================================================================
# TAB 4 — FULL REPORT (narrative synthesis + download)
# ============================================================================
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
        driver_text = ", ".join(
            f"{r.feature} ({r.coef:+.2f})" for r in top_drivers.itertuples()
        )
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
