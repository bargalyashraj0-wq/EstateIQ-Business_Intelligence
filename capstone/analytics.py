import ast
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud


from data_loader import (
    clip_for_viz,
    get_amenity_frequencies,
    get_nearby_location_frequencies,
    get_sector_geo_stats,
    load_analytics_data,
)

st.set_page_config(page_title="Analytics | EstateIQ", page_icon="📊", layout="wide")
st.title("📊 Analytics Module")
st.caption("Explore Gurgaon's real-estate market: prices, sectors, amenities, and trends.")

df_raw =pd.read_csv('gurgaon_properties_cleaned_v2.csv')
latlong = pd.read_csv('latlong.csv')
latlong['coordinates'] = latlong['coordinates'].str.replace('°', '', regex=False)
latlong[['lat', 'lon']] = (
    latlong['coordinates']
    .str.extract(r'([-+]?\d*\.?\d+)\D+([-+]?\d*\.?\d+)')
    .astype(float)
)
df1_raw=pd.read_csv('gurgaon_properties_cleaned_v1.csv')
# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

sectors = sorted(df_raw["sector"].dropna().unique())
selected_sectors = st.sidebar.multiselect("Sector", sectors, default=[])

property_types = sorted(df_raw["property_type"].dropna().unique())
selected_types = st.sidebar.multiselect("Property Type", property_types, default=property_types)

bed_min, bed_max = int(df_raw["bedRoom"].min()), int(df_raw["bedRoom"].max())
bed_range = st.sidebar.slider("Bedrooms (BHK)", bed_min, bed_max, (bed_min, bed_max))

price_min, price_max = float(df_raw["price"].min()), float(df_raw["price"].max())
price_range = st.sidebar.slider(
    "Price (₹ Crore)", price_min, price_max, (price_min, price_max)
)

# Apply filters
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

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_geo, tab_price, tab_features, tab_amenities, tab_corr = st.tabs(
    [" Overview", " Geography", " Price Trends", " Property Features",
     " Amenities & Location", " Correlations"]
)

# ---- OVERVIEW -----------------------------------------------------------
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

# ---- GEOGRAPHY ------------------------------------------------------------
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
    geo_df = get_sector_geo_stats(df,latlong)
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

# ---- PRICE TRENDS -----------------------------------------------------
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

# ---- PROPERTY FEATURES ---------------------------------------------------
with tab_features:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Luxury Score Distribution**")

        # Create the histogram
        fig = px.histogram(
            df, 
            x="luxury_score", 
            nbins=20,  # Adjust the number of bins to make it look smooth
            title="Distribution of Luxury Scores",
            labels={"luxury_score": "Luxury Score", "count": "Number of Properties"}
        )

        # Force the chart size to be a perfect square (e.g., 500x500 pixels)
        fig.update_layout(
            width=500,
            height=500,
            autosize=False
        )

        # Set use_container_width=False so Streamlit respects the custom 500x500 size
        st.plotly_chart(fig, use_container_width=False)
        
    st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Furnishing Status**")

        # Get counts
        furn_counts = df["furnishing_type"].value_counts().reset_index()
        furn_counts.columns = ["furnishing", "count"]
        

        # 1. Update the pie chart to display percentage labels on the slices
        fig = px.pie(
            furn_counts, 
            names="furnishing", 
            values="count", 
            hole=0.4,
            labels={"furnishing": "Category", "count": "Properties"}
        )

        # Force percent labels to show up clearly inside/outside the slices
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

# ---- AMENITIES & LOCATION ----------------------------------------------
with tab_amenities:
    st.markdown("**Most Common Amenities**")
    amenity_freq = Counter(
    f
    for feats in df1['features'].fillna('[]').apply(ast.literal_eval)
    for f in feats
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
    loc
    for locs in df1["nearbyLocations"]
        .fillna("[]")
        .apply(ast.literal_eval)
    for loc in locs
)
    if nearby_freq:
        top_nearby = pd.DataFrame(nearby_freq.most_common(15), columns=["landmark", "mentions"])
        fig3 = px.bar(top_nearby, x="mentions", y="landmark", orientation="h")
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No nearby-location data available for the current filter selection.")

# ---- CORRELATIONS --------------------------------------------------------
with tab_corr:
    st.markdown("**Correlation Heatmap (numeric features)**")
    numeric_cols = [
        "price", "price_per_sqft", "bedRoom", "bathroom", "built_up_area",
        "luxury_score", "floorNum",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Values close to +1 or -1 indicate strong relationships. "
        "See the ** Insights** module for model-driven feature importance."
    )
