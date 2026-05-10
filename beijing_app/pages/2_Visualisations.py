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

numeric_cols = df.select_dtypes(include='number').columns.tolist()
object_cols  = df.select_dtypes(include='object').columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in your dataset.")
    st.stop()

# ── Sidebar — user picks all columns freely ───────────────
st.sidebar.header("Column Configuration")

pollutants = st.sidebar.multiselect(
    "Select Pollutant Columns",
    options=numeric_cols,
    default=[],
    placeholder="Choose pollutant columns..."
)

met_vars = st.sidebar.multiselect(
    "Select Meteorological Columns",
    options=[c for c in numeric_cols if c not in pollutants],
    default=[],
    placeholder="Choose meteorological columns..."
)

if not pollutants:
    st.info("👈 Select at least one pollutant column from the sidebar to begin.")
    st.stop()

# ── Station / group filter ─────────────────────────────────
st.sidebar.header("Filter Options")
if object_cols:
    group_col = st.sidebar.selectbox("Group/Station Column", object_cols)
    selected_groups = st.sidebar.multiselect(
        f"Select {group_col} values",
        df[group_col].unique(),
        default=list(df[group_col].unique())
    )
    df = df[df[group_col].isin(selected_groups)]
else:
    group_col = None

# ── Tab layout ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📦 Distributions", "🔥 Correlations", "🏙️ Group Comparison"])

# Tab 1 — Trends
with tab1:
    st.subheader("Trend Over Time")
    pollutant = st.selectbox("Select Column", pollutants, key="trend_pol")
    date_cols = [c for c in df.columns if c in ['year', 'month', 'date', 'datetime', 'Date', 'Year', 'Month']]
    if 'year' in df.columns and 'month' in df.columns:
        monthly = df.groupby(['year', 'month'])[pollutant].mean().reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        fig = px.line(monthly, x='date', y=pollutant,
                      title=f"Monthly Average — {pollutant}",
                      color_discrete_sequence=["#E63946"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif date_cols:
        date_col = st.selectbox("Select date column", date_cols, key="trend_date")
        trend = df.groupby(date_col)[pollutant].mean().reset_index()
        fig = px.line(trend, x=date_col, y=pollutant,
                      title=f"Average {pollutant} over time",
                      color_discrete_sequence=["#E63946"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date/time columns detected for trend chart.")

# Tab 2 — Distributions
with tab2:
    st.subheader("Column Distribution")
    col1, col2 = st.columns(2)
    with col1:
        pol_dist = st.selectbox("Select Column", pollutants, key="dist_pol")
        fig2 = px.histogram(df, x=pol_dist, nbins=60,
                            title=f"Distribution of {pol_dist}",
                            color_discrete_sequence=["#2A9D8F"])
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if object_cols:
            cat_col = st.selectbox("Categorical Column for Pie Chart", object_cols, key="pie_col")
            counts = df[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, 'Count']
            fig3 = px.pie(counts, values='Count', names=cat_col,
                          title=f"{cat_col} Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No categorical columns found for pie chart.")

# Tab 3 — Correlations
with tab3:
    st.subheader("Correlation Heatmap")
    corr_cols = st.multiselect(
        "Select columns for correlation",
        options=numeric_cols,
        default=pollutants,
        key="corr_cols"
    )
    if corr_cols:
        corr = df[corr_cols].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdYlGn",
                         title="Correlation Matrix", zmin=-1, zmax=1)
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Scatter Plot — Bivariate Analysis")
    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("X Axis", numeric_cols, key="scatter_x")
    with c2:
        y_var = st.selectbox("Y Axis", numeric_cols,
                             index=1 if len(numeric_cols) > 1 else 0,
                             key="scatter_y")
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig5 = px.scatter(sample, x=x_var, y=y_var,
                      color=group_col if group_col else None,
                      opacity=0.4, title=f"{x_var} vs {y_var}",
                      color_discrete_sequence=px.colors.qualitative.Set1)
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

# Tab 4 — Group Comparison
with tab4:
    st.subheader("Mean Values by Group")
    if group_col:
        pol_choice = st.selectbox("Select Column", pollutants, key="station_pol")
        group_means = df.groupby(group_col)[pol_choice].mean().reset_index()
        fig6 = px.bar(group_means, x=group_col, y=pol_choice,
                      color=group_col,
                      title=f"Mean {pol_choice} by {group_col}",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig6.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("No categorical group column detected.")