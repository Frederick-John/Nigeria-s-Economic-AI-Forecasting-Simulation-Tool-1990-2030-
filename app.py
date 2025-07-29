import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import gradio as gr
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load and extend dataset
df = pd.read_csv("Nigeria Economy Dataset_1990-2022.csv")

# Extend dataset to 2030 if not already
future_years = pd.DataFrame({'Year': list(range(df['Year'].max() + 1, 2031))})
df = pd.concat([df, future_years], ignore_index=True)
df.fillna(method='ffill', inplace=True)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🇳🇬 AI-Powered Economic Forecasting & Policy Simulation (1990–2030)")
st.markdown("Analyze, simulate, and forecast Nigeria's GDP using machine learning and interactive visualizations.")

# Sidebar
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Select a section", ["📊 Dataset Overview", "📈 Visualization", "🔮 Forecasting", "🧠 PCA Trend Clustering", "🧪 Gradio Simulation"])

# Reusable variables
features = ['Agriculture to GDP', 'Industry to GDP', 'Services to GDP', 'Inflation rate', 'Government debt']
target = 'Real GDP'
model_dict = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "Linear Regression": LinearRegression()
}

# --- Dataset Overview ---
if section == "📊 Dataset Overview":
    st.subheader("🔎 Dataset Overview & Year Selection")

    year_range = st.slider("Select Year Range", int(df["Year"].min()), 2030, (1990, 2022))
    df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
    st.dataframe(df_filtered)

    st.subheader("📌 Summary Statistics")
    st.write(df_filtered.describe())

    st.subheader("📉 Missing Values")
    st.write(df_filtered.isnull().sum())

# --- Visualization ---
elif section == "📈 Visualization":
    st.subheader("📊 Economic Indicator Visualizations")

    viz_type = st.selectbox("Choose visualization type", ["Line Chart", "Bar Chart", "Pie Chart"])
    selected_features = st.multiselect("Select features to plot", features + [target], default=[target])
    year_range = st.slider("Year Range", int(df["Year"].min()), 2030, (1990, 2022))
    df_viz = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    if viz_type == "Line Chart":
        for feat in selected_features:
            st.line_chart(df_viz.set_index('Year')[feat])
    elif viz_type == "Bar Chart":
        st.bar_chart(df_viz.set_index('Year')[selected_features])
    elif viz_type == "Pie Chart":
        year = st.selectbox("Select year for pie chart", df_viz["Year"].unique())
        pie_data = df_viz[df_viz["Year"] == year][selected_features].iloc[0]
        st.pyplot(pie_data.plot.pie(autopct="%1.1f%%", figsize=(6, 6)).figure)

# --- Forecasting ---
elif section == "🔮 Forecasting":
    st.subheader("🔮 Forecast Nigeria's GDP Using Multiple AI Models")
    selected_models = st.multiselect("Choose models for prediction", list(model_dict.keys()), default=["Random Forest", "XGBoost"])
    results = {}

    df_model = df.dropna()
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name in selected_models:
        model = model_dict[name]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results[name] = {'R2': r2, 'RMSE': rmse, 'Prediction': pred}

    st.subheader("📈 Model Performance")
    st.write(pd.DataFrame({k: {'R2 Score': v['R2'], 'RMSE': v['RMSE']} for k, v in results.items()}))

    if len(results) > 1:
        st.subheader("📉 Comparison Plot")
        fig, ax = plt.subplots()
        for name, res in results.items():
            ax.plot(range(len(res["Prediction"])), res["Prediction"], label=name)
        ax.set_title("GDP Prediction Comparison")
        ax.legend()
        st.pyplot(fig)

# --- PCA Trend Clustering ---
elif section == "🧠 PCA Trend Clustering":
    st.subheader("🧠 PCA-Based Clustering of Economic Trends")

    df_trend = df.dropna(subset=features)
    scaler = StandardScaler().fit_transform(df_trend[features])
    pca = PCA(n_components=2).fit_transform(scaler)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca[:, 0], pca[:, 1], c=df_trend["Year"], cmap='viridis')
    plt.colorbar(scatter, label="Year")
    ax.set_title("Principal Component Analysis of Nigeria’s Economic Trends")
    st.pyplot(fig)

# --- Gradio Simulation ---
elif section == "🧪 Gradio Simulation":
    st.subheader("🧪 Manual GDP Simulation using Gradio")

    rf_model = RandomForestRegressor()
    X = df.dropna()[features]
    y = df.dropna()[target]
    rf_model.fit(X, y)

    def simulate_gdp(agri, ind, serv, infl, debt):
        df_input = pd.DataFrame([[agri, ind, serv, infl, debt]], columns=features)
        gdp = rf_model.predict(df_input)[0]
        fig, ax = plt.subplots()
        ax.bar(["Predicted GDP"], [gdp])
        ax.set_ylabel("Billion USD")
        return f"📈 Predicted GDP: {gdp:.2f} Billion USD", fig

    with st.expander("💡 Launch Gradio Tool"):
        demo = gr.Interface(
            fn=simulate_gdp,
            inputs=[
                gr.Slider(0, 100, value=30, label="Agriculture to GDP (%)"),
                gr.Slider(0, 100, value=20, label="Industry to GDP (%)"),
                gr.Slider(0, 100, value=50, label="Services to GDP (%)"),
                gr.Slider(0, 50, value=15, label="Inflation Rate (%)"),
                gr.Slider(0, 300, value=100, label="Government Debt (Billion USD)")
            ],
            outputs=["text", "plot"],
            title="🔧 Simulate Nigeria's GDP",
            description="Manually input economic variables to predict GDP using trained Random Forest model.",
            live=True
        )
        demo.launch(share=False)
