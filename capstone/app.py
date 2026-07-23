import streamlit as st

st.set_page_config(
    page_title="EstateIQ | Gurgaon Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 EstateIQ")
st.subheader("Gurgaon Real Estate Intelligence Platform")

st.markdown(
    """
Welcome! This app is built on ~3,800 cleaned property listings across Gurgaon
sectors, put together through a full data-science pipeline: scraping →
cleaning → EDA → feature engineering → modeling.

**Use the sidebar to navigate between modules:**

| Module | What it does |
|---|---|
| 📊 **Analytics** | Explore prices, sectors, amenities, and trends across the market |
| 💰 **Price Predictor** | Get an estimated price for a property based on its specs |
| 🏘️ **Recommender** | Find similar properties based on facilities & location |
| 💡 **Insights** | Understand *what actually drives* price in Gurgaon |
"""
)

st.info("Start with **📊 Analytics** in the sidebar to explore the market data.")
