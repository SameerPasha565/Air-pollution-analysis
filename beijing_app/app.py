import streamlit as st

st.set_page_config(
    page_title="Beijing Air Quality App",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Beijing Air Quality Analytics")
st.markdown("---")

st.write("""
This application provides an interactive platform to explore, visualise, and predict
PM2.5 air pollution across four Beijing monitoring stations (2013–2017).

**Use the sidebar to navigate between sections:**

| Page | Description |
|------|-------------|
| 📂 Dataset | Upload and preview the merged Beijing dataset |
| 📊 Visualisations | Explore pollutant trends and distributions |
| 🤖 Model Outputs | View PM2.5 prediction results and feature importance |
""")

st.info("👈 Select a page from the sidebar to begin.")
