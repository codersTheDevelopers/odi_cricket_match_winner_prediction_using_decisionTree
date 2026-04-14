import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Load model
model = pickle.load(open("cricket_model.pkl", "rb"))

st.title("🏏 Cricket Match Winner Predictor")

# List of teams (adjust if needed)
teams = [
    "India", "Pakistan", "Australia", "England",
    "South Africa", "New Zealand", "Sri Lanka", "Bangladesh", "West Indies"
]

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# Simple encoding (temporary workaround)
team_mapping = {team: i for i, team in enumerate(teams)}

if st.button("Predict Winner"):

    if team1 == team2:
        st.error("Please select two different teams")
    else:
        # Encode teams
        t1 = team_mapping[team1]
        t2 = team_mapping[team2]

        # Model input (reshape as needed)
        input_data = np.array([[t1, t2]])

        prediction = model.predict(input_data)

        # Interpret result
        if prediction[0] == 0:
            winner = team1
        else:
            winner = team2

        st.success(f"🏆 Predicted Winner: {winner}")



# Load dataset
df = pd.read_csv("ODI_Match_info.csv")
df.columns = df.columns.str.strip()  # clean column names

st.title("ODI Cricket Dataset - EDA Dashboard")

# Tabs for organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Missing Values", "Distributions", "Correlations", "Categorical Analysis"])

# Tab 1: Overview
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    
    st.write("Summary Statistics")
    st.write(df.describe(include="all"))

# Tab 2: Missing Values
with tab2:
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    st.pyplot(fig)

# Tab 3: Distributions
with tab3:
    st.subheader("Numeric Distributions")
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Tab 4: Correlations
with tab4:
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Tab 5: Categorical Analysis
with tab5:
    st.subheader("Categorical Breakdown")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        st.write(f"Counts for {col}")
        st.bar_chart(df[col].value_counts())