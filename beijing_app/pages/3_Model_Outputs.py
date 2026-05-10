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
st.title("🤖 Model Outputs — Regression Prediction")
st.markdown("---")

DATA_PATH = "uploaded_data.csv"

if not os.path.exists(DATA_PATH):
    st.warning("⚠️ Please upload your dataset on the Dataset page first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ── Auto-encode all categorical columns ───────────────────
for col in df.select_dtypes(include='object').columns:
    df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

numeric_cols = df.select_dtypes(include='number').columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in your dataset.")
    st.stop()

# ── Sidebar — fully user-driven ───────────────────────────
st.sidebar.header("Model Configuration")

target_col = st.sidebar.selectbox(
    "🎯 Target Column (what to predict)",
    options=numeric_cols,
    placeholder="Select target..."
)

feature_cols = st.sidebar.multiselect(
    "🔧 Feature Columns",
    options=[c for c in numeric_cols if c != target_col],
    default=[],
    placeholder="Choose feature columns..."
)

if not feature_cols:
    st.info("👈 Select a target and at least one feature column from the sidebar.")
    st.stop()

st.sidebar.markdown(f"**Features:** {len(feature_cols)}")
st.sidebar.markdown(f"**Target:** {target_col}")

# ── Prepare data ──────────────────────────────────────────
df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols]
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

st.sidebar.markdown(f"**Training rows:** {len(X_train):,}")
st.sidebar.markdown(f"**Testing rows:** {len(X_test):,}")

# ── Train models ──────────────────────────────────────────
@st.cache_resource
def train_models(X_tr, y_tr):
    lr = LinearRegression().fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               min_samples_split=5, random_state=42,
                               n_jobs=-1).fit(X_tr, y_tr)
    return lr, rf

with st.spinner("Training models..."):
    lr, rf = train_models(X_train_s, y_train)

lr_pred = lr.predict(X_test_s)
rf_pred = rf.predict(X_test_s)

# ── Metrics ───────────────────────────────────────────────
st.subheader("Model Performance Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 Linear Regression (Baseline)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mean_absolute_error(y_test, lr_pred):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
    m3.metric("R²",   f"{r2_score(y_test, lr_pred):.3f}")

with col2:
    st.markdown("### 🌲 Random Forest (Main Model)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mean_absolute_error(y_test, rf_pred):.2f}",
              delta=f"{mean_absolute_error(y_test, lr_pred) - mean_absolute_error(y_test, rf_pred):.2f} better")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
    m3.metric("R²",   f"{r2_score(y_test, rf_pred):.3f}",
              delta=f"{r2_score(y_test, rf_pred) - r2_score(y_test, lr_pred):.3f} better")

st.markdown("---")

# ── Actual vs Predicted ───────────────────────────────────
st.subheader(f"Actual vs Predicted — {target_col}")
n = st.slider("Test samples to display", 100, min(1000, len(y_test)), 500)

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test.values[:n], name=f"Actual {target_col}",
                         line=dict(color="#333333", width=1.5)))
fig.add_trace(go.Scatter(y=lr_pred[:n], name="Linear Regression",
                         line=dict(color="#2A9D8F", width=1, dash="dash")))
fig.add_trace(go.Scatter(y=rf_pred[:n], name="Random Forest",
                         line=dict(color="#E63946", width=1, dash="dot")))
fig.update_layout(title=f"Actual vs Predicted {target_col} (Test Set)",
                  xaxis_title="Test Sample Index",
                  yaxis_title=target_col,
                  template="plotly_white", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Feature Importance ────────────────────────────────────
st.subheader("Feature Importance (Random Forest)")
importances = pd.Series(rf.feature_importances_,
                        index=feature_cols).sort_values(ascending=True)
fig2 = px.bar(importances.reset_index(), x=0, y='index',
              orientation='h',
              title=f"Feature Importance — Drivers of {target_col}",
              labels={0: "Importance Score", "index": "Feature"},
              color=0, color_continuous_scale="Oranges")
fig2.update_layout(template="plotly_white", showlegend=False, height=450)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Live Prediction Widget ────────────────────────────────
st.subheader(f"🎯 Live {target_col} Predictor")
st.write("Adjust the sliders to simulate conditions and get a live prediction:")

input_vals = {}
cols = st.columns(3)

for i, feat in enumerate(feature_cols):
    f_min  = float(df_model[feat].min())
    f_max  = float(df_model[feat].max())
    f_mean = float(df_model[feat].mean())
    step   = 1.0 if df_model[feat].nunique() <= 30 else round((f_max - f_min) / 100, 2)
    input_vals[feat] = cols[i % 3].slider(
        feat, f_min, f_max, f_mean, step=max(step, 0.01)
    )

if st.button(f"🔮 Predict {target_col}", type="primary"):
    input_df     = pd.DataFrame([input_vals])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pred         = rf.predict(input_scaled)[0]
    st.metric(f"Predicted {target_col}", f"{pred:.2f}")