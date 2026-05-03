import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Outputs", page_icon="🤖", layout="wide")
st.title("🤖 Model Outputs — PM2.5 Prediction")
st.markdown("---")

DATA_PATH = "uploaded_data.csv"

if not os.path.exists(DATA_PATH):
    st.warning("⚠️ Please upload your dataset on the Dataset page first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ── Encode categoricals ──────────────────────────────────────
if 'station' in df.columns:
    df['station_encoded'] = LabelEncoder().fit_transform(df['station'])
if 'season' in df.columns:
    df['season_encoded'] = LabelEncoder().fit_transform(df['season'])
if 'area_type' in df.columns:
    df['area_encoded'] = LabelEncoder().fit_transform(df['area_type'])

# ── Feature setup ────────────────────────────────────────────
candidate_features = ['SO2','NO2','CO_mg','O3','month','hour','week',
                      'heating_season','station_encoded','season_encoded','area_encoded']
feature_cols = [c for c in candidate_features if c in df.columns]
df_model = df[feature_cols + ['PM2.5']].dropna()

X = df_model[feature_cols]
y = df_model['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Model info sidebar ───────────────────────────────────────
st.sidebar.header("Model Configuration")
st.sidebar.markdown(f"**Features used:** {len(feature_cols)}")
st.sidebar.markdown(f"**Training rows:** {len(X_train):,}")
st.sidebar.markdown(f"**Testing rows:** {len(X_test):,}")
st.sidebar.markdown("**Target:** PM2.5 (µg/m³)")

# ── Train models with caching ────────────────────────────────
@st.cache_resource
def train_models(X_tr, y_tr):
    lr = LinearRegression().fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               min_samples_split=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    return lr, rf

with st.spinner("Training models... (this may take 30 seconds)"):
    lr, rf = train_models(X_train_s, y_train)

lr_pred = lr.predict(X_test_s)
rf_pred = rf.predict(X_test_s)

# ── Metrics ──────────────────────────────────────────────────
st.subheader("Model Performance Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 Linear Regression (Baseline)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(y_test, lr_pred):.2f} µg/m³")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, lr_pred)):.2f} µg/m³")
    m3.metric("R²", f"{r2_score(y_test, lr_pred):.3f}")

with col2:
    st.markdown("### 🌲 Random Forest (Main Model)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(y_test, rf_pred):.2f} µg/m³",
              delta=f"{mean_absolute_error(y_test, lr_pred) - mean_absolute_error(y_test, rf_pred):.2f} better")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, rf_pred)):.2f} µg/m³")
    m3.metric("R²", f"{r2_score(y_test, rf_pred):.3f}",
              delta=f"{r2_score(y_test, rf_pred) - r2_score(y_test, lr_pred):.3f} better")

st.markdown("---")

# ── Actual vs Predicted ──────────────────────────────────────
st.subheader("Actual vs Predicted PM2.5")
n = st.slider("Number of test samples to display", 100, 1000, 500)

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test.values[:n], name="Actual PM2.5",
                          line=dict(color="#333333", width=1.5)))
fig.add_trace(go.Scatter(y=lr_pred[:n], name="Linear Regression",
                          line=dict(color="#2A9D8F", width=1, dash="dash")))
fig.add_trace(go.Scatter(y=rf_pred[:n], name="Random Forest",
                          line=dict(color="#E63946", width=1, dash="dot")))
fig.update_layout(title="Actual vs Predicted PM2.5 (Test Set)",
                  xaxis_title="Test Sample Index (chronological)",
                  yaxis_title="PM2.5 (µg/m³)",
                  template="plotly_white", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Feature Importance ───────────────────────────────────────
st.subheader("Feature Importance (Random Forest)")
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig2 = px.bar(importances.reset_index(), x=0, y='index',
              orientation='h',
              title="Feature Importance — Drivers of PM2.5",
              labels={0: "Importance Score", "index": "Feature"},
              color=0, color_continuous_scale="Oranges")
fig2.update_layout(template="plotly_white", showlegend=False, height=450)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Live Prediction Widget ───────────────────────────────────
st.subheader("🎯 Live PM2.5 Predictor")
st.write("Adjust the sliders to simulate pollution conditions and predict PM2.5:")

input_vals = {}
cols = st.columns(3)
slider_cfg = {
    'SO2':      (0.0, 200.0, 20.0),  'NO2': (0.0, 200.0, 40.0),
    'CO_mg':    (0.1, 10.0,  1.0),   'O3':  (0.0, 150.0, 30.0),
    'month':    (1, 12, 6),           'hour': (0, 23, 12),
    'week':     (1, 52, 26),          'heating_season': (0, 1, 0),
    'station_encoded': (0, 3, 0),     'season_encoded':  (0, 3, 1),
    'area_encoded':    (0, 1, 0)
}

for i, feat in enumerate(feature_cols):
    if feat in slider_cfg:
        lo, hi, default = slider_cfg[feat]
        step = 1 if feat in ['month','hour','week','heating_season','station_encoded','season_encoded','area_encoded'] else 0.1
        input_vals[feat] = cols[i % 3].slider(feat, float(lo), float(hi), float(default), step=float(step))

if st.button("🔮 Predict PM2.5", type="primary"):
    input_df = pd.DataFrame([input_vals])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pred = rf.predict(input_scaled)[0]

    if pred < 35:
        cat, colour = "Excellent / Good 🟢", "green"
    elif pred < 75:
        cat, colour = "Lightly Polluted 🟡", "orange"
    elif pred < 115:
        cat, colour = "Moderately Polluted 🟠", "orangered"
    else:
        cat, colour = "Heavily / Severely Polluted 🔴", "red"

    st.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
    st.markdown(f"**Air Quality Category:** :{colour}[{cat}]")
