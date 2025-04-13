import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EDA Dashboard - Model Training Logs",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["📊 Dashboard", "📁 Upload File", "📈 Visualizations", "📥 Download"])

# --- FILE LOADING ---
def load_data(file_path="lstm_warm_start_log.txt"):
    try:
        df = pd.read_csv(file_path, sep="\t")
        df['Rolling_Loss'] = df['Loss'].rolling(window=5).mean()
        df['Loss_Diff'] = df['Loss'].diff()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- FILE UPLOAD SECTION ---
if section == "📁 Upload File":
    st.header("📁 Upload Training Log File")
    uploaded_file = st.file_uploader("Choose a .txt log file", type="txt")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep="\t")
        df.to_csv("uploaded_log.csv", index=False)
        st.success("File uploaded successfully. Go to '📊 Dashboard' or '📈 Visualizations' to explore.")

# --- DASHBOARD SECTION ---
if section == "📊 Dashboard":
    st.title("BALA VISWANATH GUDIMETLA'S 📊 Model Training Dashboard")
    df = load_data()
    if df is not None:
        min_loss_row = df.loc[df['Loss'].idxmin()]
        last_epoch = int(df['Epoch'].max())

        col1, col2, col3 = st.columns(3)
        col1.metric("🔻 Min Loss", f"{min_loss_row['Loss']:.4f}", f"Epoch {int(min_loss_row['Epoch'])}")
        col2.metric("📈 Max Loss", f"{df['Loss'].max():.4f}")
        col3.metric("📅 Total Epochs", f"{last_epoch}")

        st.subheader("📍 Loss Over Epochs")
        fig = px.line(df, x='Epoch', y='Loss', title="Loss per Epoch", markers=True)
        fig.add_scatter(x=df['Epoch'], y=df['Rolling_Loss'], mode='lines', name='Rolling Avg (5)')
        st.plotly_chart(fig, use_container_width=True)

# --- VISUALIZATION SECTION ---
if section == "📈 Visualizations":
    st.title("📈 Visual Exploration")
    df = load_data()
    if df is not None:
        with st.expander("📉 Loss Difference (Δ Loss) Between Epochs"):
            fig2 = px.bar(df[1:], x='Epoch', y='Loss_Diff', title="Δ Loss per Epoch", color='Loss_Diff')
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📊 Loss Distribution Histogram"):
            fig3 = px.histogram(df, x='Loss', nbins=15, title="Loss Value Distribution")
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("🔍 Zoomed-In View of Last 10 Epochs"):
            df_tail = df.tail(10)
            fig4 = px.line(df_tail, x='Epoch', y='Loss', title="Last 10 Epochs Loss", markers=True)
            st.plotly_chart(fig4, use_container_width=True)

# --- DOWNLOAD SECTION ---
if section == "📥 Download":
    st.title("📥 Download Reports")
    df = load_data()
    if df is not None:
        st.subheader("Download Data with Rolling Loss & Deltas")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='processed_log.csv',
            mime='text/csv'
        )

        st.subheader("Download as Excel")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='EDA')
            writer.save()
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name="eda_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )