import streamlit as st

st.set_page_config(
    page_title="Air Quality Analytics",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Air Quality Analytics Dashboard")
st.markdown("---")

st.write("""
This application provides an interactive platform to explore, visualise,
and predict air quality metrics from any uploaded dataset.

**Use the sidebar to navigate between sections:**

| Page | Description |
|------|-------------|
| 📂 Dataset | Upload and preview your CSV dataset |
| 📊 Visualisations | Explore trends, distributions and correlations |
| 🤖 Model Outputs | Train models and view prediction results |
""")

st.info("👈 Upload your dataset on the **Dataset** page first, then explore the other sections.")