import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("combined_output.csv")
    df = df.dropna(subset=['PM2.5'])  # Drop rows with missing target
    df['wd'] = df['wd'].astype('category').cat.codes  # Encode wind direction

    # Fill missing values in features
    feature_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    return df

# Load and filter data
data = load_data()

# 🎉 Welcome message (always shows)
st.title("🌫️ Beijing Air Quality Dashboard")
st.markdown("""
Welcome to the **Beijing Air Quality Analysis and PM2.5 Prediction Tool**!

This app allows you to:
- Explore and understand air pollution data
- Visualize trends and relationships
- Predict PM2.5 concentrations using machine learning

👉 Use the sidebar to navigate through the sections.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "PM2.5 Prediction"])

# Station filter
stations = sorted(data["station"].unique())
selected_station = st.sidebar.selectbox("Select a station", stations)
filtered_data = data[data["station"] == selected_station]

# Define feature columns and target
features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
target = 'PM2.5'

# Page 1: Data Overview
if page == "Data Overview":
    st.header("📄 Dataset Overview")
    st.write(f"### First 5 rows from: {selected_station}")
    st.dataframe(filtered_data.head())

    st.write("### Summary Statistics")
    st.write(filtered_data.describe())

# Page 2: EDA
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    sample_df = filtered_data.sample(n=min(5000, len(filtered_data)), random_state=42).copy()

    st.write("### PM2.5 Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(sample_df['PM2.5'], bins=40, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.write("### PM2.5 vs Temperature")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=sample_df, x='TEMP', y='PM2.5', alpha=0.3, ax=ax2)
    st.pyplot(fig2)

    st.write("### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    corr = sample_df[features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

# Page 3: PM2.5 Prediction
elif page == "PM2.5 Prediction":
    st.header("🔮 Predict PM2.5 Levels")

    # Prepare data
    X = filtered_data[features]
    y = filtered_data[target]

    # Check again for NaNs
    if X.isnull().any().any() or y.isnull().any():
        st.error("❌ Data contains missing values. Please select a different station.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Input form
        st.write("### Input feature values below:")
        input_data = {}
        for col in features:
            val = st.number_input(col, value=float(filtered_data[col].mean()))
            input_data[col] = val

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                st.success(f"✅ Predicted PM2.5 value: **{prediction[0]:.2f} µg/m³**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Evaluation
        y_pred = model.predict(X_test)
        st.write("### 📈 Model Performance")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
