import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# load environment variables securely
from dotenv import load_dotenv
load_dotenv()  # will look for a .env in current working dir

# Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    # If groq is not installed, we won't crash the whole app.
    GROQ_AVAILABLE = False

# --- Helper functions ---


def validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    """
    Ensure the uploaded Excel contains required columns.
    Returns None if OK, else error message.
    """
    required = {"Category", "Budget", "Actual"}
    existing = set(df.columns.str.strip())
    missing = required - existing
    if missing:
        return f"File missing required columns: {', '.join(missing)}"
    return None


def compute_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance and Variance %.
    Handles Budget == 0 safely (Variance % = NaN or inf-handled).
    """
    df = df.copy()
    # normalize column names (strip)
    df.columns = df.columns.str.strip()
    # ensure numeric
    df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce").fillna(0.0)
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce").fillna(0.0)

    df["Variance"] = df["Actual"] - df["Budget"]

    # Avoid division by zero: when Budget == 0, set Variance % to np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Variance %"] = np.where(
            df["Budget"] == 0,
            np.nan,
            (df["Variance"] / df["Budget"]) * 100,
        )
    return df


def variance_color(value: float) -> str:
    """
    Helper used if mapping colors manually.
    But we'll prefer a Plotly continuous color scale (custom).
    """
    if pd.isna(value):
        return "gray"
    if value < 0:
        return "red"
    if value == 0:
        return "yellow"
    return "green"


def create_variance_bar(df: pd.DataFrame):
    """
    Create an interactive bar chart (Variance by Category) with red->yellow->green scheme.
    We'll use a custom continuous color scale mapping negative -> red, near zero -> yellow, positive -> green.
    """
    # sort for consistent display
    chart_df = df.sort_values("Variance", ascending=True)

    # Map colors using a diverging palette. We'll create a numeric color scale
    # We normalize by setting center at 0. Use sign and magnitude to color.
    # For px.bar continuous_color, we supply color=Variance and define color_continuous_scale.
    # We'll set range symmetric around max absolute for better color mapping.
    max_abs = max(chart_df["Variance"].abs().max(), 1.0)  # avoid zero
    # Custom three-stop scale: negative (red) -> zero (yellow) -> positive (green)
    color_scale = [
        [0.0, "rgb(215,25,28)"],   # strong red (negative)
        [0.5, "rgb(255,255,179)"], # yellow (around zero)
        [1.0, "rgb(26,150,65)"],   # green (positive)
    ]

    fig = px.bar(
        chart_df,
        x="Category",
        y="Variance",
        color="Variance",
        color_continuous_scale=color_scale,
        range_color=[-max_abs, max_abs],
        labels={"Variance": "Variance (Actual - Budget)"},
        title="Variance by Category",
        hover_data=["Budget", "Actual", "Variance %"],
    )

    fig.update_layout(xaxis_tickangle=-45, coloraxis_colorbar=dict(title="Variance"))
    return fig


def create_budget_vs_actual_line(df: pd.DataFrame):
    """
    Creates a line chart (Budget vs Actual) per Category.
    We will melt the dataframe to long form for px.line.
    """
    # If categories are many, line will plot categories as separate traces
    long = df.melt(id_vars=["Category"], value_vars=["Budget", "Actual"], var_name="Type", value_name="Amount")

    fig = px.line(
        long,
        x="Category",
        y="Amount",
        color="Type",
        markers=True,
        labels={"Amount": "Amount", "Category": "Category", "Type": "Series"},
        title="Budget vs Actual (per Category)",
        hover_data=["Amount"],
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def build_summary_text(df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Build a concise textual summary of the variance dataset to provide as context to the AI model.
    We'll include totals and top positive/negative variances.
    """
    total_budget = df["Budget"].sum()
    total_actual = df["Actual"].sum()
    total_variance = df["Variance"].sum()

    # Top positive (overspend → positive variance if Actual > Budget? Actually Positive variance = over-performance if revenue)
    # Depending on semantics user wants: we'll report largest positive and negative variances by absolute value and sign.
    top_pos = df.sort_values("Variance", ascending=False).head(top_n)[["Category", "Budget", "Actual", "Variance", "Variance %"]]
    top_neg = df.sort_values("Variance", ascending=True).head(top_n)[["Category", "Budget", "Actual", "Variance", "Variance %"]]

    summary_lines = [
        f"Total Budget: {total_budget:,.2f}",
        f"Total Actual: {total_actual:,.2f}",
        f"Total Variance (Actual - Budget): {total_variance:,.2f}",
        "",
        f"Top {top_n} largest positive variances (Actual > Budget):",
    ]
    for _, r in top_pos.iterrows():
        summary_lines.append(f" - {r['Category']}: Variance={r['Variance']:,.2f}, Variance%={'{:.2f}%'.format(r['Variance %']) if not pd.isna(r['Variance %']) else 'N/A'}")

    summary_lines.append("")
    summary_lines.append(f"Top {top_n} largest negative variances (Actual < Budget):")
    for _, r in top_neg.iterrows():
        summary_lines.append(f" - {r['Category']}: Variance={r['Variance']:,.2f}, Variance%={'{:.2f}%'.format(r['Variance %']) if not pd.isna(r['Variance %']) else 'N/A'}")

    return "\n".join(summary_lines)


def call_groq_summary(client: Groq, context_text: str, model: str = "llama-3.1-8b-instant"):
    """
    Call Groq's chat completion to get summary insights & recommendations.
    This function expects 'client' to be an initialized Groq instance with valid API key.
    Implementation based on Groq python SDK patterns.
    """
    # Construct a simple conversation: system instructions + user content
    system_msg = "You are a helpful financial analyst. Provide a concise summary of the data and 3 recommendations to management for addressing material variances."
    user_msg = f"Here is the dataset summary (Budget vs Actual):\n\n{context_text}\n\nPlease provide a short summary and 3 actionable recommendations."

    # The SDK supports chat completions. Example pattern:
    # response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"system","content":...}, {"role":"user","content":...}])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        # response structure may vary; this tries to extract content
        # Many Groq SDK responses provide .choices[0].message["content"]
        if hasattr(response, "choices"):
            content = response.choices[0].message.get("content") or response.choices[0].get("text")
        else:
            # fallback
            content = str(response)
        return content
    except Exception as e:
        return f"[Groq API call failed] {e}"


def call_groq_qa(client: Groq, context_text: str, question: str, model: str = "llama-3.1-8b-instant"):
    """
    Ask a user question to the model, with the dataset summary as context.
    """
    system_msg = "You are a helpful financial analyst. Use the provided dataset summary to answer the user's question concisely and refer to categories/values if relevant."
    user_msg = f"Dataset summary:\n{context_text}\n\nQuestion: {question}\nPlease answer concisely and, if numerical calculation is needed, show the numbers used."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        if hasattr(response, "choices"):
            content = response.choices[0].message.get("content") or response.choices[0].get("text")
        else:
            content = str(response)
        return content
    except Exception as e:
        return f"[Groq API call failed] {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="Budget vs. Actuals AI – Variance Analysis & Commentary", layout="wide")

st.title("Budget vs. Actuals AI – Variance Analysis & Commentary")

st.markdown(
    """
    Upload an Excel (`.xlsx`) file with columns: **Category**, **Budget**, **Actual**.
    The app will calculate Variance and Variance % and generate interactive charts + AI commentary (via Groq).
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload Excel (.xlsx) file", type=["xlsx"], accept_multiple_files=False)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        st.stop()

    # Validate
    validation_error = validate_dataframe(df)
    if validation_error:
        st.error(validation_error)
        st.stop()

    # Compute variance
    df_calc = compute_variance(df)

    st.subheader("Preview — Calculated Data")
    st.dataframe(df_calc.style.format({"Budget": "{:,.2f}", "Actual": "{:,.2f}", "Variance": "{:,.2f}", "Variance %": "{:,.2f}%"}), height=300)

    # Charts
    st.subheader("Charts")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_variance_bar(df_calc), use_container_width=True)

    with col2:
        st.plotly_chart(create_budget_vs_actual_line(df_calc), use_container_width=True)

    # Prepare summary text for AI context
    summary_text = build_summary_text(df_calc, top_n=5)

    st.subheader("AI Insights & Recommendations (Groq)")

    groq_api_key = os.getenv("GROQ_API_KEY", None)

    if not GROQ_AVAILABLE:
        st.warning("Groq SDK is not installed in the environment. AI features disabled. To enable them, install package 'groq'.")
        st.info("Example: pip install groq")
    elif not groq_api_key:
        st.warning("GROQ_API_KEY not found in environment. Put your Groq API key in a .env file as GROQ_API_KEY=your_key")
    else:
        # initialize client
        try:
            client = Groq(api_key=groq_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            client = None

        if client:
            # Part 1: auto-generated summary & recommendations
            with st.spinner("Generating summary & recommendations from the AI..."):
                ai_summary = call_groq_summary(client, summary_text, model="llama-3.1-8b-instant")
            st.markdown("**AI Summary & Top Recommendations:**")
            st.write(ai_summary)

            # Part 2: interactive Q&A
            st.markdown("---")
            st.markdown("**Ask a question about the variance data**")
            user_question = st.text_input("Type your question (e.g., 'Which categories have the largest overspend?')", key="qa_input")
            if user_question:
                with st.spinner("AI answering..."):
                    ai_answer = call_groq_qa(client, summary_text, user_question, model="llama-3.1-8b-instant")
                st.markdown("**AI Answer:**")
                st.write(ai_answer)

# If no file uploaded, show an example and instructions
else:
    st.info("Upload a .xlsx file to start. Example file structure:")
    st.markdown(
        """
        | Category        | Budget   | Actual   |
        |----------------|---------:|---------:|
        | Revenue A      | 100000   | 120000   |
        | Revenue B      | 50000    | 45000    |
        | Expense C      | 20000    | 25000    |
        """
    )

    st.markdown("**Notes / Tips**")
    st.markdown(
        """
        - Ensure column names are exactly: **Category**, **Budget**, **Actual** (case-insensitive but recommended to match).
        - For Budget values equal to zero, Variance% will be shown as `N/A`.
        - To enable AI features:
          1. Create a `.env` file with `GROQ_API_KEY=your_api_key_here`.
          2. Install the Groq SDK: `pip install groq`.
          3. Restart the Streamlit app.
        - Recommended dependencies for requirements.txt:
          ```
          streamlit
          pandas
          numpy
          plotly
          python-dotenv
          openpyxl
          groq
          ```
        """
    )
