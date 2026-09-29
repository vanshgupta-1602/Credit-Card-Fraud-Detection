import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b1120;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .card-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 25px;
        font-weight: 700;
    }

    /* Section heading */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .fraud-card {
        background: #3b1f24;
        border: 1px solid #7f1d1d;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
    }

    .safe-card {
        background: #123524;
        border: 1px solid #166534;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
    }

    .result-title {
        font-size: 26px;
        font-weight: 700;
    }

    .result-subtitle {
        color: #cbd5e1;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
model = joblib.load("notebooks/fraud_model.pkl")
scaler = joblib.load("notebooks/scaler.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/creditcard_clean.csv")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI/ML powered transaction risk analysis system</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- OVERVIEW CARDS ----------------
total_transactions = len(df)
fraud_transactions = int((df["Class"] == 1).sum())
legitimate_transactions = int((df["Class"] == 0).sum())

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="info-card">
        <div class="card-title">TOTAL TRANSACTIONS</div>
        <div class="card-value">{total_transactions:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="info-card">
        <div class="card-title">LEGITIMATE</div>
        <div class="card-value">🟢 {legitimate_transactions:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="info-card">
        <div class="card-title">FRAUDULENT</div>
        <div class="card-value">🔴 {fraud_transactions:,}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TRANSACTION ANALYSIS ----------------
st.markdown(
    '<div class="section-title">🔍 Transaction Analysis</div>',
    unsafe_allow_html=True
)

option = st.selectbox(
    "Select a transaction for demonstration",
    [
        "Demo Legitimate Transaction",
        "Demo Fraudulent Transaction"
    ]
)

if option == "Demo Legitimate Transaction":

    sample = (
        df[df["Class"] == 0]
        .drop("Class", axis=1)
        .iloc[[0]]
    )

    actual = 0

else:

    sample = (
        df[df["Class"] == 1]
        .drop("Class", axis=1)
        .iloc[[0]]
    )

    actual = 1


# ---------------- PREDICTION ----------------
if st.button("🔎 Analyze Transaction", use_container_width=True):

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    probability = model.predict_proba(sample_scaled)[0][1]

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )

    # -------- RESULT --------
    if prediction == 1:

        st.markdown(f"""
        <div class="fraud-card">
            <div class="result-title">🚨 Fraudulent Transaction</div>
            <div class="result-subtitle">
                The model has classified this transaction as potentially fraudulent.
            </div>
        </div>
        """, unsafe_allow_html=True)

        risk = "HIGH RISK"

    else:

        st.markdown(f"""
        <div class="safe-card">
            <div class="result-title">✅ Legitimate Transaction</div>
            <div class="result-subtitle">
                The model has classified this transaction as legitimate.
            </div>
        </div>
        """, unsafe_allow_html=True)

        risk = "LOW RISK"


    st.write("")

    # -------- METRICS --------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Actual Class",
            "Fraud" if actual == 1 else "Legitimate"
        )

    with col2:
        st.metric(
            "Predicted Class",
            "Fraud" if prediction == 1 else "Legitimate"
        )

    with col3:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )


    # -------- RISK --------
    st.write("")

    st.markdown("### ⚠️ Risk Assessment")

    if probability >= 0.5:
        st.error(f"🔴 {risk}")
    elif probability >= 0.2:
        st.warning("🟠 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")


# ---------------- MODEL INFORMATION ----------------
st.divider()

st.markdown(
    '<div class="section-title">🤖 Machine Learning Model</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">MODEL</div>
        <div class="card-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">PREPROCESSING</div>
        <div class="card-value">StandardScaler</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">TARGET</div>
        <div class="card-value">Class 0 / 1</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Credit Card Fraud Detection • Machine Learning Project<br>
    Built with Python • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)




