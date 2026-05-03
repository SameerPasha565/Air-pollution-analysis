import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Visualisations", page_icon="📊", layout="wide")
st.title("📊 Visualisations")
st.markdown("---")

DATA_PATH = "uploaded_data.csv"

if not os.path.exists(DATA_PATH):
    st.warning("⚠️ Please upload your dataset on the Dataset page first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

pollutants = [c for c in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO_mg', 'O3'] if c in df.columns]
met_vars   = [c for c in ['TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN'] if c in df.columns]

# ── Sidebar filters ─────────────────────────────────────────
st.sidebar.header("Filter Options")
if 'station' in df.columns:
    stations = st.sidebar.multiselect("Select Stations", df['station'].unique(), default=list(df['station'].unique()))
    df = df[df['station'].isin(stations)]

# ── Tab layout ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📦 Distributions", "🔥 Correlations", "🏙️ Station Comparison"])

# Tab 1 — Trends
with tab1:
    st.subheader("Pollutant Trend Over Time")
    pollutant = st.selectbox("Select Pollutant", pollutants, key="trend_pol")
    if 'year' in df.columns and 'month' in df.columns:
        monthly = df.groupby(['year', 'month'])[pollutant].mean().reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        fig = px.line(monthly, x='date', y=pollutant,
                      title=f"Monthly Average {pollutant} (2013–2017)",
                      labels={pollutant: f"{pollutant} (µg/m³)", "date": "Date"},
                      color_discrete_sequence=["#E63946"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Year/month columns not found for trend chart.")

# Tab 2 — Distributions
with tab2:
    st.subheader("Pollutant Distribution")
    col1, col2 = st.columns(2)
    with col1:
        pol_dist = st.selectbox("Select Pollutant", pollutants, key="dist_pol")
        fig2 = px.histogram(df, x=pol_dist, nbins=60,
                            title=f"Distribution of {pol_dist}",
                            color_discrete_sequence=["#2A9D8F"])
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if 'AQI_Category' in df.columns:
            aqi_counts = df['AQI_Category'].value_counts().reset_index()
            aqi_counts.columns = ['AQI Category', 'Count']
            fig3 = px.pie(aqi_counts, values='Count', names='AQI Category',
                          title="AQI Category Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig3, use_container_width=True)

# Tab 3 — Correlations
with tab3:
    st.subheader("Correlation Heatmap")
    num_cols = df[pollutants].select_dtypes(include='number')
    corr = num_cols.corr().round(2)
    fig4 = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdYlGn",
                     title="Pollutant Correlation Matrix",
                     zmin=-1, zmax=1)
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Scatter Plot — Bivariate Analysis")
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X Axis", pollutants, index=0, key="scatter_x")
    with col2:
        y_var = st.selectbox("Y Axis", pollutants, index=1, key="scatter_y")

    color_by = 'station' if 'station' in df.columns else None
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig5 = px.scatter(sample, x=x_var, y=y_var, color=color_by,
                      opacity=0.4, title=f"{x_var} vs {y_var}",
                      color_discrete_sequence=px.colors.qualitative.Set1)
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

# Tab 4 — Station Comparison
with tab4:
    st.subheader("Mean Pollutant Levels by Station")
    if 'station' in df.columns and pollutants:
        station_means = df.groupby('station')[pollutants].mean().reset_index()
        pol_choice = st.selectbox("Select Pollutant", pollutants, key="station_pol")
        fig6 = px.bar(station_means, x='station', y=pol_choice,
                      color='station', title=f"Mean {pol_choice} by Station",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig6.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Station column not found.")
