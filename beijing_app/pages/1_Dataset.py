import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dataset", page_icon="📂", layout="wide")
st.title("📂 Dataset Overview")
st.markdown("---")

DATA_PATH = "uploaded_data.csv"

uploaded_file = st.file_uploader("Upload your merged Beijing CSV dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv(DATA_PATH, index=False)
    st.success("✅ Dataset uploaded successfully!")
elif os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.info("📁 Using previously uploaded dataset.")
else:
    st.warning("⚠️ Please upload your merged Beijing dataset to begin.")
    st.stop()

# ── Dataset Shape ──────────────────────────────────────────
st.subheader("Dataset Dimensions")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", f"{df.shape[0]:,}")
col2.metric("Total Columns", df.shape[1])
col3.metric("Stations", df['station'].nunique() if 'station' in df.columns else "N/A")

st.markdown("---")

# ── Preview ────────────────────────────────────────────────
st.subheader("Data Preview")
n_rows = st.slider("Number of rows to display", 5, 50, 10)
st.dataframe(df.head(n_rows), use_container_width=True)

# ── Column Info ────────────────────────────────────────────
st.subheader("Column Information")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.values,
    "Missing Values": df.isnull().sum().values,
    "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2)
})
st.dataframe(col_info, use_container_width=True)

# ── Statistical Summary ────────────────────────────────────
st.subheader("Statistical Summary")
st.dataframe(df.describe().round(2), use_container_width=True)
