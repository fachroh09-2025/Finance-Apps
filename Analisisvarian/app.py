import streamlit as st
import pandas as pd
import plotly.express as px
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.set_page_config(page_title="Budget vs. Actuals AI", page_icon="📊", layout="wide")
st.title("📊 Budget vs. Actuals AI – Variance Analysis & Commentary")
st.write("Upload your Budget vs. Actuals file and get AI-driven financial insights!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Check for required columns
    required_columns = ["Category", "Budget", "Actual"]
    if not all(col in df.columns for col in required_columns):
        st.error("⚠️ The uploaded file must contain 'Category', 'Budget', and 'Actual' columns!")
        st.stop()

    # Calculate Variance and Variance Percentage
    df["Variance"] = df["Actual"] - df["Budget"]
    df["Variance %"] = (df["Variance"] / df["Budget"]) * 100

    # Display data preview
    st.subheader("📊 Data Preview with Variance Calculation")
    st.dataframe(df)

    # Plot Variance Analysis
    st.subheader("📈 Budget vs. Actual Variance Analysis")
    
    fig_bar = px.bar(
        df,
        x="Category",
        y="Variance",
        color="Variance",
        title="📊 Variance by Category",
        text_auto=".2s",
        color_continuous_scale=["red", "yellow", "green"],
    )
    st.plotly_chart(fig_bar)

    fig_line = px.line(
        df,
        x="Category",
        y=["Budget", "Actual"],
        markers=True,
        title="📉 Budget vs. Actual Performance",
    )
    st.plotly_chart(fig_line)

    # AI Section
    st.subheader("🤖 AI-Powered Variance Analysis")

    # AI Summary of Variance Data
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI financial analyst providing variance analysis insights on budget vs. actuals."},
            {"role": "user", "content": f"Here is the budget vs. actual variance summary:\n{df.to_string()}\nWhat are the key insights and recommendations?"}
        ],
        model="llama-3.1-8b-instant",
    )

    st.write(response.choices[0].message.content)

    # AI Chat - Users Can Ask Questions
    st.subheader("🗣️ Chat with AI About Variance Analysis")

    user_query = st.text_input("🔍 Ask the AI about your variance data:")
    if user_query:
        chat_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI financial analyst helping users understand their budget vs. actual variance analysis."},
                {"role": "user", "content": f"Variance Data:\n{df.to_string()}\n{user_query}"}
            ],
            model="llama-3.1-8b-instant",
        )
        st.write(chat_response.choices[0].message.content)
