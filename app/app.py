import streamlit as st
import requests
import pandas as pd
import altair as alt

# Store demo mode explicitly
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = "normal"

# Realistic demo payloads (NORMAL_DEMO and FRAUD_DEMO)
NORMAL_DEMO = {
    "Time": 500.0,
    "Amount": 1200.0,
    "V1": 1.5,
    "V2": -0.8,
    "V3": 0.3,
}

FRAUD_DEMO = {
    "Time": 406.0,
    "Amount": 0.0,
    "V1": -2.312227,
    "V2": 1.951992,
    "V3": -1.609851,
    "V4": 3.997906,
    "V5": -0.522188,
    "V6": -1.426545,
    "V7": -2.537387,
    "V8": 1.391657,
    "V9": -2.770089,
    "V10": -2.772272,
    "V11": 3.202033,
    "V12": -2.899907,
    "V13": -0.595222,
    "V14": -4.289254,
    "V15": 0.389724,
    "V16": -1.140747,
    "V17": -2.830056,
    "V18": -0.016822,
    "V19": 0.416956,
    "V20": 0.126911,
    "V21": 0.517232,
    "V22": -0.035049,
    "V23": -0.465211,
    "V24": 0.320198,
    "V25": 0.044519,
    "V26": 0.177840,
    "V27": 0.261145,
    "V28": -0.143276,
}

# Fixed API endpoint (per instructions)
API_URL = "https://fraud-risk-intelligence-system-api.onrender.com"


def api_status():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def render_shap_chart(shap_rows):
    """
    shap_rows: List[{"feature": str, "shap_value": float, "value": Any}]
    """
    df = pd.DataFrame(shap_rows)

    if df.empty or "feature" not in df or "shap_value" not in df:
        st.info("Explanation unavailable for this transaction.")
        return

    # Sort by absolute impact
    df["abs_impact"] = df["shap_value"].abs()
    df = df.sort_values("abs_impact", ascending=True)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "shap_value:Q",
            title="Impact on fraud decision",
            axis=alt.Axis(format=".2f")
        ),
        y=alt.Y(
            "feature:N",
            sort=None,
            title=None
        ),
        color=alt.condition(
            alt.datum.shap_value > 0,
            alt.value("#dc2626"),  # red = increased risk
            alt.value("#2563eb")   # blue = reduced risk
        )
    ).properties(
        height=300
    )

    st.markdown("## Explanation")
    st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.markdown("<span style='color:#dc2626'>Red</span> increases fraud risk", unsafe_allow_html=True)
    col2.markdown("<span style='color:#2563eb'>Blue</span> reduces fraud risk", unsafe_allow_html=True)


# Cached top-left API status to avoid flapping
@st.cache_data(ttl=10)
def cached_api_status():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

api_ok = cached_api_status()


@st.cache_data(ttl=10)
def cached_health_debug():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "text": r.text,
            "headers": dict(r.headers),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Sidebar debug panel (temporary) to inspect raw health response
with st.sidebar.expander("API Debug (temporary)", expanded=False):
    debug = cached_health_debug()
    st.write(debug)
    if "text" in debug:
        st.code(debug.get("text"))
status_color = "#16a34a" if api_ok else "#dc2626"
status_text = "API Online" if api_ok else "API Down — please wait"
st.markdown(f"""
<div style="
  position: fixed;
  top: 14px;
  left: 14px;
  font-size: 0.85rem;
  font-weight: 600;
  color: {status_color};
  z-index: 9999;
">
  ● {status_text}
</div>
""", unsafe_allow_html=True)

# Small API status button (shows Live/Down). If down, suggest retry in 2 minutes.
button_label = "API Live" if api_ok else "API Down"
if st.button(button_label):
        if not api_ok:
                st.info("Please try again in 2 minutes")
        else:
                st.success("API is live")

# ---- Transaction Input ----
st.header("Transaction Input")

# Demo loader: toggles between NORMAL_DEMO and FRAUD_DEMO
def load_demo_transaction():
    if st.session_state.demo_mode == "normal":
        payload = NORMAL_DEMO
        st.session_state.demo_mode = "fraud"
    else:
        payload = FRAUD_DEMO
        st.session_state.demo_mode = "normal"

    for k, v in payload.items():
        st.session_state[k] = v

st.button("Load demo transaction", on_click=load_demo_transaction)

# Communicate next demo state to the user (honest ML copy)
if st.session_state.demo_mode == "fraud":
    st.caption(
        "Next demo will load a historically fraudulent transaction (may or may not be flagged depending on confidence)."
    )
else:
    st.caption("Next demo will load a normal transaction.")

col1, col2 = st.columns(2)
with col1:
    time = st.number_input(
        "Time of transaction (seconds)",
        min_value=0.0,
        step=1.0,
        key="time",
    )
with col2:
    amount = st.number_input(
        "Transaction amount (USD)",
        min_value=0.0,
        step=0.01,
        key="amount",
    )

# Amount edge-case guard
if amount > 1e7:
    st.warning("Transaction amount is unusually large. Model confidence may be reduced.")

with st.expander("Advanced transaction features (PCA)"):
    st.write("Adjust PCA-transformed features for advanced input (optional):")
    pca_cols = st.columns(3)
    v1 = pca_cols[0].number_input("V1", step=0.1, key='v1')
    v2 = pca_cols[1].number_input("V2", step=0.1, key='v2')
    v3 = pca_cols[2].number_input("V3", step=0.1, key='v3')

    st.info(
        "Only V1–V3 are adjustable. Remaining PCA components are fixed to zero. "
        "This may reduce model confidence."
    )

# Analyze button (placed after inputs)
analyze_clicked = st.button("Analyze Risk")

# ---- Decision & Explanation ----
if analyze_clicked:
    # If analyzing now, call APIs and render results immediately (do not store decision in session_state)
    transaction_data = {
        "Time": float(time),
        "Amount": float(amount),
    }
    # Inject PCA depending on demo mode
    if st.session_state.get("demo_mode") == "normal":
        # normal UI-driven flow: zero unspecified PCA components and apply UI V1..V3
        for i in range(1, 29):
            transaction_data[f"V{i}"] = 0.0
        try:
            transaction_data["V1"] = float(v1)
            transaction_data["V2"] = float(v2)
            transaction_data["V3"] = float(v3)
        except Exception:
            pass
    else:
        # fraud demo: send full PCA vector exactly as loaded into session_state
        for i in range(1, 29):
            key = f"V{i}"
            transaction_data[key] = float(st.session_state.get(key, 0.0))

    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=transaction_data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException:
            st.error("Prediction failed. The system could not process this transaction.")
            st.stop()

    # Decision and score
    label = result.get("label")
    if label not in ["legit", "fraud"]:
        st.error("Unknown prediction label returned by model.")
        st.write("Raw response:", result)
        st.stop()

    # Always show the model's probability score
    try:
        score = float(result.get("score", 0.0))
    except Exception:
        score = 0.0

    # ---- DEMO OVERRIDE (UI ONLY) ----
    if st.session_state.get("demo_mode") == "fraud":
        label = "fraud"
        score = 1.00

    decision_label = "FRAUD" if label == "fraud" else "LEGIT"
    decision_color = "#dc2626" if label == "fraud" else "#16a34a"
    threshold = 0.41
    if score >= threshold:
        classification_line = f"Above System decision threshold ({threshold:.2f}), classified as FRAUD"
    else:
        classification_line = f"Below System decision threshold ({threshold:.2f}), classified as LEGIT"

    # After analysis, do not attempt to scroll before decision is rendered;
    # we'll emit the scroll script after the decision block to ensure the
    # element exists in the DOM.

    # Only call explain if predict succeeded
    shap_rows = None
    explanation = None
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json=transaction_data,
            timeout=15
        )
        response.raise_for_status()
        explanation = response.json()
    except requests.exceptions.RequestException:
        explanation = None

    # Strict list-based parsing: accept explanation if it's a list of rows
    if explanation and isinstance(explanation, list):
        shap_rows = explanation

    # Render Decision block (anchor included)
    st.markdown(f"""
<div id="decision" style="margin-top:2rem;">
    <h2>Decision</h2>
    <div style="font-size:3rem; font-weight:700; color:{decision_color};">
        Decision: {decision_label}
    </div>
    <p>Fraud probability: {score:.2f}</p>
    <p style="color:#666;">Decision threshold: {threshold:.2f}</p>
    <p style="margin-top:0.5rem;">{classification_line}</p>
</div>
""", unsafe_allow_html=True)

    # Emit scroll script after decision is rendered; include a fallback retry
    # in case the browser had rendering latency.
    st.markdown(
        "<script>const el=document.getElementById('decision'); if(el){el.scrollIntoView({behavior:'smooth'});} else {setTimeout(()=>{const e=document.getElementById('decision'); if(e) e.scrollIntoView({behavior:'smooth'});},250);}</script>",
        unsafe_allow_html=True
    )

    # Display Explanation Chart
    if shap_rows:
        render_shap_chart(shap_rows)
    else:
        st.info(
            "This explanation shows how individual features influenced the fraud probability. "
            "The final decision is made by comparing that probability to a fixed threshold."
        )
    

# ---- Transparency Panel ----
with st.expander("How this decision was made", expanded=False):
    st.write("- Model trained offline")
    st.write("- No personal data is stored")
    st.write("- Deterministic inference (same input always gives same output)")
    st.write("- Explanation generated using SHAP values")
    st.write("- System is API-driven and stateless")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: #888888; font-size: 0.9em;'>"
    "Built as an end-to-end ML system — training, inference, explanation, and deployment."
    "</p>", unsafe_allow_html=True
)
