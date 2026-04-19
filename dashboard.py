import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
matplotlib.use("Agg")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CardioRisk AI — Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ANTHEM BCBS COLOR PALETTE + GLOBAL CSS
# Primary blue:    #003087  (Anthem deep navy)
# Secondary blue:  #0070C0  (Anthem bright blue)
# Light blue:      #E8F4FD  (Background tint)
# Accent blue:     #00A3E0  (Anthem sky blue)
# White:           #FFFFFF
# Text dark:       #1A1A2E
# Risk high:       #C8102E  (Clinical red)
# Risk moderate:   #E8A020  (Amber)
# Risk low:        #00843D  (Clinical green)
# ============================================================

ANTHEM_CSS = """
<style>
    /* ── Global background ── */
    .stApp {
        background-color: #F0F6FC;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #003087 0%, #0070C0 60%, #00A3E0 100%);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0,48,135,0.18);
    }
    .header-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.88;
        margin: 4px 0 0 0;
    }
    .header-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.8rem;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ── Disclaimer banner ── */
    .disclaimer-box {
        background: #E8F4FD;
        border-left: 4px solid #0070C0;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #003087;
    }

    /* ── Chat container ── */
    .chat-container {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,48,135,0.08);
        border: 1px solid #D0E8F8;
        margin-bottom: 20px;
    }

    /* ── Chat message styling ── */
    .stChatMessage {
        background: transparent !important;
    }

    /* Assistant messages */
    [data-testid="stChatMessageContent"] {
        background: #E8F4FD !important;
        border-radius: 12px !important;
        border: 1px solid #C5DFF5 !important;
        padding: 12px 16px !important;
        color: #1A1A2E !important;
    }

    /* ── Chat input — prominent glowing style ── */
    .stChatInput {
        background: white !important;
        border-radius: 32px !important;
        border: 2.5px solid #0070C0 !important;
        box-shadow: 0 0 0 4px rgba(0,112,192,0.12),
                    0 4px 16px rgba(0,48,135,0.12) !important;
        padding: 4px 8px !important;
        margin-top: 8px !important;
    }
    .stChatInput textarea {
        border-radius: 28px !important;
        border: none !important;
        background: white !important;
        color: #1A1A2E !important;
        font-size: 0.95rem !important;
        padding: 14px 20px !important;
        caret-color: #0070C0 !important;
    }
    .stChatInput textarea::placeholder {
        color: #7BAFD4 !important;
        font-style: italic !important;
    }
    .stChatInput textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .stChatInput button {
        background: linear-gradient(135deg, #003087, #0070C0) !important;
        border-radius: 50% !important;
        color: white !important;
        border: none !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: 0 2px 8px rgba(0,48,135,0.25) !important;
    }
    .chat-input-label {
        background: linear-gradient(135deg, #003087, #0070C0);
        color: white;
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(0,48,135,0.2);
    }

    /* ── Progress bar ── */
    .progress-container {
        background: #D0E8F8;
        border-radius: 8px;
        height: 8px;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #003087, #00A3E0);
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    .progress-label {
        font-size: 0.8rem;
        color: #0070C0;
        font-weight: 500;
        margin-bottom: 6px;
    }

    /* ── Section cards ── */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,48,135,0.07);
        border: 1px solid #D0E8F8;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #003087;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E8F4FD;
    }

    /* ── Risk badge ── */
    .risk-badge-high {
        background: #FFF0F0;
        border: 2px solid #C8102E;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
    }
    .risk-badge-moderate {
        background: #FFF8EC;
        border: 2px solid #E8A020;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
    }
    .risk-badge-low {
        background: #F0FFF6;
        border: 2px solid #00843D;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #D0E8F8;
        box-shadow: 0 1px 4px rgba(0,48,135,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #0070C0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        color: #003087 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        color: #003087 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #D0E8F8;
    }

    /* ── Reset button ── */
    .stButton > button {
        background: linear-gradient(135deg, #003087, #0070C0) !important;
        color: white !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 3px 10px rgba(0,48,135,0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 16px rgba(0,48,135,0.3) !important;
    }

    /* ── Caption ── */
    .footer-caption {
        text-align: center;
        font-size: 0.78rem;
        color: #6B8CAE;
        padding: 16px 0;
        border-top: 1px solid #D0E8F8;
        margin-top: 8px;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #0070C0 !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Elegant section divider (replaces white box spacers) ── */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 20px 0;
        gap: 12px;
    }
    .section-divider::before,
    .section-divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg,
            transparent, #C5DFF5 30%, #C5DFF5 70%, transparent);
    }
    .section-divider-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #0070C0;
        opacity: 0.5;
    }

    /* Remove Streamlit default top padding that causes white space */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stAppViewContainer"] > section:first-child {
        padding-top: 0 !important;
    }
</style>
"""

# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "best_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

saved = load_model()

# ============================================================
# FEATURE DESCRIPTIONS
# ============================================================

FEATURE_DESCRIPTIONS = {
    "age":               "Age (years)",
    "sex":               "Sex (1=male, 0=female)",
    "cp":                "Chest pain type (1-4)",
    "trestbps":          "Resting blood pressure (mm Hg)",
    "chol":              "Serum cholesterol (mg/dl)",
    "fbs":               "Fasting blood sugar >120 mg/dl",
    "restecg":           "Resting ECG result (0-2)",
    "thalach":           "Max heart rate achieved",
    "exang":             "Exercise-induced angina (1=yes)",
    "oldpeak":           "ST depression (exercise vs rest)",
    "slope":             "Slope of peak ST segment (1-3)",
    "ca":                "Major vessels colored (0-3)",
    "thal":              "Thalassemia (3=normal, 6=fixed, 7=reversible)",
    "ca_was_missing":    "CA value was missing",
    "thal_was_missing":  "Thal value was missing",
    "slope_was_missing": "Slope value was missing",
    "chol_was_missing":  "Chol value was missing",
}

# ============================================================
# CHATBOT QUESTIONS
# ============================================================

QUESTIONS = [
    {
        "field": "age",
        "q": "What is the patient's age? (29–77)",
        "type": "int",
        "range": (29, 77),
    },
    {
        "field": "sex",
        "q": "Is the patient male or female? (type 'male' or 'female')",
        "type": "choice",
        "choices": {"male": 1, "female": 0},
    },
    {
        "field": "cp",
        "q": "Chest pain type?\n1 = Typical angina\n2 = Atypical angina\n3 = Non-anginal pain\n4 = Asymptomatic\n(type 1, 2, 3, or 4)",
        "type": "int",
        "range": (1, 4),
    },
    {
        "field": "trestbps",
        "q": "Resting blood pressure in mm Hg? (94–200)",
        "type": "int",
        "range": (94, 200),
    },
    {
        "field": "chol",
        "q": "Serum cholesterol in mg/dl? (126–564)",
        "type": "int",
        "range": (126, 564),
    },
    {
        "field": "fbs",
        "q": "Is fasting blood sugar > 120 mg/dl? (yes / no)",
        "type": "yesno",
    },
    {
        "field": "restecg",
        "q": "Resting ECG result?\n0 = Normal\n1 = ST-T wave abnormality\n2 = Left ventricular hypertrophy\n(type 0, 1, or 2)",
        "type": "int",
        "range": (0, 2),
    },
    {
        "field": "thalach",
        "q": "Maximum heart rate achieved? (71–202)",
        "type": "int",
        "range": (71, 202),
    },
    {
        "field": "exang",
        "q": "Exercise-induced angina? (yes / no)",
        "type": "yesno",
    },
    {
        "field": "oldpeak",
        "q": "ST depression induced by exercise vs rest? (0.0–6.2)",
        "type": "float",
        "range": (0.0, 6.2),
    },
    {
        "field": "slope",
        "q": "Slope of peak ST segment?\n1 = Upsloping\n2 = Flat\n3 = Downsloping\n(type 1, 2, or 3)",
        "type": "int",
        "range": (1, 3),
    },
    {
        "field": "ca",
        "q": "Number of major vessels colored by fluoroscopy? (0, 1, 2, or 3)",
        "type": "int",
        "range": (0, 3),
    },
    {
        "field": "thal",
        "q": "Thalassemia result?\n3 = Normal\n6 = Fixed defect\n7 = Reversible defect\n(type 3, 6, or 7)",
        "type": "choice_int",
        "choices": {3: 3, 6: 6, 7: 7},
    },
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_answer(user_input, q):
    text = user_input.strip().lower()
    if not text:
        raise ValueError("Please enter a value — field cannot be blank")
    if len(text) > 50:
        raise ValueError("Input too long — please enter a simple value")

    if q["type"] == "choice_int":
        if "." in text:
            raise ValueError(
                f"Please type a whole number only. "
                f"Valid options: {list(q['choices'].keys())}. "
                f"You typed: '{user_input}'")
        try:
            val = int(text.strip())
        except ValueError:
            raise ValueError(
                f"Please type a whole number only. "
                f"Valid options: {list(q['choices'].keys())}. "
                f"You typed: '{user_input}'")
        if val not in q["choices"]:
            raise ValueError(
                f"{val} is not valid. "
                f"Only {list(q['choices'].keys())} are accepted. "
                f"Please type 3, 6, or 7.")
        return float(q["choices"][val])

    if q["type"] == "choice":
        words   = text.replace(",", " ").replace("/", " ").split()
        matched = None
        for word in words:
            for key in q["choices"]:
                if word == key:
                    if matched is not None and matched != key:
                        raise ValueError(
                            f"Ambiguous input '{user_input}'. "
                            f"Please type exactly: {list(q['choices'].keys())}")
                    matched = key
        if matched is not None:
            return float(q["choices"][matched])
        matches = [k for k in q["choices"] if k in text]
        if len(matches) == 1:
            return float(q["choices"][matches[0]])
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous input. Please type exactly: {list(q['choices'].keys())}")
        raise ValueError(
            f"Please type one of: {list(q['choices'].keys())}. "
            f"You typed: '{user_input}'")

    if q["type"] == "yesno":
        yes_words = ["yes", "y", "1", "true", "yeah", "yep"]
        no_words  = ["no",  "n", "0", "false", "nope", "nah"]
        if text in yes_words:
            return 1.0
        if text in no_words:
            return 0.0
        words = text.split()
        if any(w in yes_words for w in words):
            return 1.0
        if any(w in no_words for w in words):
            return 0.0
        raise ValueError(f"Please type 'yes' or 'no'. You typed: '{user_input}'")

    if q["type"] == "int":
        cleaned = text.lstrip("0") or "0"
        if "." in cleaned:
            try:
                as_float = float(cleaned)
                if as_float != int(as_float):
                    raise ValueError(
                        f"Please enter a whole number. "
                        f"You typed '{user_input}'. "
                        f"Did you mean {int(as_float)}?")
                cleaned = str(int(as_float))
            except ValueError as e:
                if "whole number" in str(e):
                    raise
                raise ValueError(f"Please enter a whole number. You typed: '{user_input}'")
        try:
            val = int(cleaned)
        except ValueError:
            import re
            nums = re.findall(r"\d+", text)
            if nums:
                val = int(nums[0])
            else:
                raise ValueError(f"Please enter a number. You typed: '{user_input}'")
        if "range" in q:
            lo, hi = q["range"]
            if not (lo <= val <= hi):
                raise ValueError(
                    f"Value {val} is out of range. "
                    f"Please enter a value between {lo} and {hi}")
        return float(val)

    if q["type"] == "float":
        try:
            val = float(text)
        except ValueError:
            import re
            nums = re.findall(r"[\d.]+", text)
            if nums:
                val = float(nums[0])
            else:
                raise ValueError(f"Please enter a decimal number. You typed: '{user_input}'")
        if "range" in q:
            lo, hi = q["range"]
            if not (lo <= val <= hi):
                raise ValueError(
                    f"Value {val} is out of range. "
                    f"Please enter a value between {lo} and {hi}")
        return val

    raise ValueError("Unknown question type — please contact support")


def get_risk_level(prob):
    if prob >= 0.70:
        return "HIGH", "#C8102E", "🔴"
    elif prob >= 0.40:
        return "MODERATE", "#E8A020", "🟡"
    else:
        return "LOW", "#00843D", "🟢"


def get_recommendation(prob):
    if prob >= 0.70:
        return ("Recommend cardiology referral and further "
                "diagnostic workup. Immediate clinical evaluation advised.")
    elif prob >= 0.40:
        return ("Recommend lifestyle intervention discussion and "
                "follow-up appointment within 3 months.")
    else:
        return ("Recommend routine monitoring and preventive care. "
                "Continue regular check-ups.")


def build_patient_vector(inputs, feature_names):
    row = {}
    for feat in feature_names:
        if feat.endswith("_was_missing"):
            original = feat.replace("_was_missing", "")
            row[feat] = 1.0 if inputs.get(original) is None else 0.0
        else:
            row[feat] = inputs.get(feat, np.nan)
    return pd.DataFrame([row])[feature_names]


def predict_patient(saved, inputs):
    feature_names = saved["feature_names"]
    preprocessor  = saved["preprocessor"]
    model         = saved["model"]
    model_raw     = saved["model_raw"]

    X_input     = build_patient_vector(inputs, feature_names)
    X_processed = preprocessor.transform(X_input)
    X_df        = pd.DataFrame(X_processed, columns=feature_names)
    y_prob      = float(model.predict_proba(X_df)[0, 1])

    model_name = saved["model_name"]
    if model_name in ["Random Forest", "XGBoost"]:
        explainer   = shap.TreeExplainer(model_raw)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]
    else:
        background  = (saved["background"].values
                       if "background" in saved else X_df.values)
        explainer   = shap.LinearExplainer(
            model_raw, background,
            feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        p            = y_prob
        shap_values  = shap_values * p * (1 - p)
        base_val     = float(explainer.expected_value)
        try:
            base_val = 1 / (1 + math.exp(-base_val))
        except (OverflowError, ValueError):
            base_val = y_prob

    shap_vals = np.array(shap_values).ravel()
    shap_df   = pd.DataFrame({
        "feature":      feature_names,
        "shap_value":   shap_vals,
        "actual_value": X_df.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    return y_prob, shap_df, X_df, shap_vals, float(base_val)


def plot_shap_waterfall(shap_vals, base_value, X_df,
                         feature_names, y_prob):
    readable = [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
    shap_exp  = shap.Explanation(
        values        = shap_vals,
        base_values   = float(base_value),
        data          = X_df.values[0],
        feature_names = readable,
    )
    # Anthem-themed SHAP colors
    plt.rcParams["axes.facecolor"]  = "#F8FBFF"
    plt.rcParams["figure.facecolor"] = "white"
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, show=False)
    plt.title(f"Prediction reasoning — risk score {y_prob:.1%}",
              fontsize=11, pad=12, color="#003087", fontweight="bold")
    plt.tight_layout()
    return fig


def make_health_gauge(title, value, max_value, green_max, yellow_max):
    # Determine needle color based on which zone value falls in
    if value <= green_max:
        needle_color = "#00843D"
    elif value <= yellow_max:
        needle_color = "#E8A020"
    else:
        needle_color = "#C8102E"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 13, "color": "#003087",
                                          "family": "Segoe UI"}},
        number = {"font": {"size": 30, "color": needle_color,
                            "family": "Segoe UI", "weight": "bold"}},
        gauge = {
            "axis": {
                "range": [0, max_value],
                "tickcolor": "#6B8CAE",
                "tickfont":  {"color": "#6B8CAE", "size": 10},
                "tickwidth": 1,
            },
            # Needle bar uses zone color so it pops visually
            "bar": {"color": needle_color, "thickness": 0.28},
            "bgcolor":    "#F8FBFF",
            "borderwidth": 2,
            "bordercolor": "#D0E8F8",
            "steps": [
                # Fully saturated zone colors — green/amber/red
                {"range": [0, green_max],          "color": "#C6EFD4"},
                {"range": [green_max, yellow_max], "color": "#FFEAA0"},
                {"range": [yellow_max, max_value], "color": "#FFBABA"},
            ],
            "threshold": {
                "line":      {"color": needle_color, "width": 4},
                "thickness": 0.85,
                "value":     value,
            },
        }
    ))
    fig.update_layout(
        height        = 240,
        margin        = dict(t=60, b=10, l=20, r=20),
        paper_bgcolor = "white",
        plot_bgcolor  = "white",
    )
    return fig


DIVIDER = """
<div class="section-divider">
    <span class="section-divider-dot"></span>
    <span class="section-divider-dot"></span>
    <span class="section-divider-dot"></span>
</div>
"""

def show_results(saved, inputs, y_prob,
                 shap_df, X_df, shap_vals, base_value):
    risk_level, risk_color, emoji = get_risk_level(y_prob)
    recommendation = get_recommendation(y_prob)

    # ── Risk result card ──────────────────────────────────
    badge_class = {
        "HIGH": "risk-badge-high",
        "MODERATE": "risk-badge-moderate",
        "LOW": "risk-badge-low",
    }[risk_level]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Risk Assessment Result</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Risk Score", f"{y_prob:.1%}")
    with col2:
        st.markdown(f"""
        <div class="{badge_class}">
            <p style="margin:0;font-size:11px;color:{risk_color};
                      font-weight:600;text-transform:uppercase;
                      letter-spacing:1px;">Risk Level</p>
            <p style="margin:4px 0 0 0;font-size:22px;
                      font-weight:700;color:{risk_color};">
                {emoji} {risk_level}
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:#E8F4FD;border-left:4px solid #0070C0;
                    border-radius:0 8px 8px 0;padding:14px 18px;height:100%;">
            <p style="margin:0;font-size:11px;color:#0070C0;
                      font-weight:600;text-transform:uppercase;
                      letter-spacing:0.5px;">Clinical Recommendation</p>
            <p style="margin:6px 0 0 0;font-size:13px;color:#1A1A2E;
                      line-height:1.5;">
                {recommendation}
            </p>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Risk gauge — sharp Plotly version ────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📈 Risk Gauge</p>',
                unsafe_allow_html=True)

    # Zone widths: LOW 0-40%, MODERATE 40-70%, HIGH 70-100%
    gauge_fig = go.Figure()

    # Zone backgrounds
    for x0, x1, color, label, lx in [
        (0.0, 0.40, "#C6EFD4", "LOW",      0.20),
        (0.40, 0.70, "#FFEAA0", "MODERATE", 0.55),
        (0.70, 1.00, "#FFBABA", "HIGH",     0.85),
    ]:
        gauge_fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1, y0=0, y1=0.6,
            fillcolor=color, line_width=0, layer="below")
        gauge_fig.add_annotation(
            x=lx, y=0.68, text=f"<b>{label}</b>",
            showarrow=False, font=dict(
                size=11,
                color={"LOW": "#00843D",
                       "MODERATE": "#C87000",
                       "HIGH": "#C8102E"}[label],
                family="Segoe UI"),
            xref="x", yref="y")

    # Score bar
    gauge_fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=0, x1=y_prob, y0=0.05, y1=0.55,
        fillcolor=risk_color, opacity=0.9,
        line_width=0)

    # Score marker line
    gauge_fig.add_shape(
        type="line", xref="x", yref="y",
        x0=y_prob, x1=y_prob, y0=-0.05, y1=0.65,
        line=dict(color=risk_color, width=3))

    # Score label on bar
    gauge_fig.add_annotation(
        x=y_prob, y=0.3,
        text=f"<b>{y_prob:.1%}</b>",
        showarrow=False,
        font=dict(size=13, color="white", family="Segoe UI"),
        xref="x", yref="y",
        bgcolor=risk_color,
        borderpad=4)

    gauge_fig.update_xaxes(
        range=[0, 1],
        tickvals=[0, 0.20, 0.40, 0.60, 0.70, 0.80, 1.0],
        ticktext=["0%", "20%", "40%", "60%", "70%", "80%", "100%"],
        tickfont=dict(size=10, color="#6B8CAE"),
        showgrid=False, zeroline=False)
    gauge_fig.update_yaxes(
        visible=False, range=[-0.1, 0.9])
    gauge_fig.update_layout(
        height=140,
        margin=dict(t=30, b=20, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=f"<b>Risk score: {y_prob:.1%}  ·  {risk_level} risk</b>",
            font=dict(size=13, color="#003087", family="Segoe UI"),
            x=0.5, xanchor="center"),
        showlegend=False)

    st.plotly_chart(gauge_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Health indicator gauges ───────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🩺 Patient Health Indicators</p>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_health_gauge(
            "Blood Pressure (mm Hg)",
            inputs.get("trestbps", 130), 200, 120, 140),
            use_container_width=True)
    with g2:
        st.plotly_chart(make_health_gauge(
            "Cholesterol (mg/dl)",
            inputs.get("chol", 200), 400, 200, 240),
            use_container_width=True)
    with g3:
        st.plotly_chart(make_health_gauge(
            "Max Heart Rate (bpm)",
            inputs.get("thalach", 150), 202, 100, 160),
            use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── SHAP explanation ──────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔍 Prediction Reasoning (SHAP)</p>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.88rem;color:#6B8CAE;margin-bottom:16px;'>"
        "The chart shows which clinical features pushed this prediction "
        "higher or lower than the average patient baseline.</p>",
        unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        try:
            fig_wf = plot_shap_waterfall(
                shap_vals, base_value,
                X_df, saved["feature_names"], y_prob)
            st.pyplot(fig_wf)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP chart unavailable: {e}")

    with col_s2:
        st.markdown("""
        <div style="background:#FFF0F0;border-radius:8px;
                    padding:14px 16px;margin-bottom:12px;">
            <p style="margin:0 0 8px 0;font-size:12px;font-weight:700;
                      color:#C8102E;text-transform:uppercase;
                      letter-spacing:0.5px;">⬆ Factors Increasing Risk</p>
        """, unsafe_allow_html=True)
        for _, row in shap_df[shap_df["shap_value"] > 0].head(4).iterrows():
            name = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
            st.markdown(
                f"<p style='margin:4px 0;font-size:12px;color:#1A1A2E;'>"
                f"🔴 <strong>{name}</strong><br>"
                f"<span style='color:#6B8CAE;font-size:11px;'>"
                f"value={row['actual_value']:.2f} · "
                f"impact=+{row['shap_value']:.3f}</span></p>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F0FFF6;border-radius:8px;
                    padding:14px 16px;margin-bottom:12px;">
            <p style="margin:0 0 8px 0;font-size:12px;font-weight:700;
                      color:#00843D;text-transform:uppercase;
                      letter-spacing:0.5px;">⬇ Factors Decreasing Risk</p>
        """, unsafe_allow_html=True)
        for _, row in shap_df[shap_df["shap_value"] < 0].head(3).iterrows():
            name = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
            st.markdown(
                f"<p style='margin:4px 0;font-size:12px;color:#1A1A2E;'>"
                f"🟢 <strong>{name}</strong><br>"
                f"<span style='color:#6B8CAE;font-size:11px;'>"
                f"value={row['actual_value']:.2f} · "
                f"impact={row['shap_value']:.3f}</span></p>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#E8F4FD;border-radius:8px;padding:12px 14px;">
            <p style="margin:0;font-size:11px;color:#0070C0;font-weight:600;">
                HOW TO READ IMPACT SCORES</p>
            <p style="margin:6px 0 0 0;font-size:11px;color:#1A1A2E;
                      line-height:1.6;">
                Each number shows how much a feature shifted the prediction
                from the average patient baseline.<br>
                <strong>Positive</strong> = pushed risk higher<br>
                <strong>Negative</strong> = pushed risk lower
            </p>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Patient summary table — styled HTML ──────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Patient Input Summary</p>',
                unsafe_allow_html=True)

    display_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak",
                    "slope", "ca", "thal"]

    # Human-readable value display
    SEX_MAP     = {1.0: "Male", 0.0: "Female"}
    YESNO_MAP   = {1.0: "Yes",  0.0: "No"}
    YESNO_COLS  = {"fbs", "exang"}

    rows_html = ""
    for i, f in enumerate(display_cols):
        if f not in inputs:
            continue
        raw   = inputs[f]
        label = FEATURE_DESCRIPTIONS.get(f, f)
        if f == "sex":
            val_display = SEX_MAP.get(raw, str(raw))
        elif f in YESNO_COLS:
            val_display = YESNO_MAP.get(raw, str(raw))
        elif raw == int(raw):
            val_display = str(int(raw))
        else:
            val_display = str(raw)

        row_bg = "#F8FBFF" if i % 2 == 0 else "white"
        rows_html += f"""
        <tr style="background:{row_bg};">
            <td style="padding:10px 16px;font-size:0.88rem;
                       color:#1A1A2E;font-weight:500;
                       border-bottom:1px solid #E8F0F8;
                       width:60%;">{label}</td>
            <td style="padding:10px 16px;font-size:0.88rem;
                       color:#003087;font-weight:700;
                       border-bottom:1px solid #E8F0F8;
                       text-align:right;">{val_display}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;
                  border-radius:10px;overflow:hidden;
                  border:1.5px solid #D0E8F8;">
        <thead>
            <tr style="background:linear-gradient(135deg,#003087,#0070C0);">
                <th style="padding:10px 16px;text-align:left;
                           font-size:0.8rem;color:white;
                           font-weight:600;letter-spacing:0.5px;
                           text-transform:uppercase;">Clinical Feature</th>
                <th style="padding:10px 16px;text-align:right;
                           font-size:0.8rem;color:white;
                           font-weight:600;letter-spacing:0.5px;
                           text-transform:uppercase;">Value</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────
    st.markdown(
        f"<div class='footer-caption'>"
        f"Model: {saved['model_name']} with Platt scaling calibration &nbsp;|&nbsp; "
        f"Trained on UCI Heart Disease dataset (920 patients) &nbsp;|&nbsp; "
        f"This tool does not constitute a medical diagnosis."
        f"</div>",
        unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "step" not in st.session_state:
    st.session_state.step     = 0
    st.session_state.answers  = {}
    st.session_state.done     = False
    st.session_state.results  = None
    st.session_state.error    = ""

# ============================================================
# RENDER CSS  +  STICKY HEADER CSS
# ============================================================

st.markdown(ANTHEM_CSS, unsafe_allow_html=True)

# Sticky header CSS — pins banner + disclaimer + progress to top
st.markdown("""
<style>
    /* Sticky header wrapper */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #F0F6FC;
        padding-bottom: 8px;
    }

    /* Question card — no min-height so it doesn't create white box */
    .question-card {
        background: white;
        border-radius: 16px;
        padding: 32px 36px;
        box-shadow: 0 2px 16px rgba(0,48,135,0.09);
        border: 1px solid #D0E8F8;
        margin: 24px 0 16px 0;
    }
    .question-number {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0070C0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }
    .question-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1A1A2E;
        line-height: 1.55;
        margin-bottom: 20px;
        white-space: pre-line;
    }
    .answer-display {
        background: #E8F4FD;
        border: 1.5px solid #0070C0;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 1.0rem;
        color: #003087;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 4px;
    }
    .answer-label {
        font-size: 0.75rem;
        color: #6B8CAE;
        margin-bottom: 4px;
        font-weight: 500;
    }

    /* Navigation buttons row */
    .nav-row {
        display: flex;
        gap: 12px;
        margin-top: 8px;
        align-items: center;
    }

    /* Answer input field */
    .stTextInput input {
        border-radius: 10px !important;
        border: 2px solid #0070C0 !important;
        font-size: 1.05rem !important;
        padding: 12px 16px !important;
        color: #1A1A2E !important;
        background: white !important;
        box-shadow: 0 0 0 3px rgba(0,112,192,0.08) !important;
    }
    .stTextInput input:focus {
        border-color: #003087 !important;
        box-shadow: 0 0 0 4px rgba(0,48,135,0.14) !important;
    }

    /* Answer history dots */
    .dot-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #D0E8F8;
        display: inline-block;
    }
    .dot-done {
        background: #0070C0;
    }
    .dot-current {
        background: #003087;
        width: 12px;
        height: 12px;
        box-shadow: 0 0 0 3px rgba(0,112,192,0.25);
    }

    /* ── Form submit button (Next/Submit) — force Anthem blue ── */
    /* Overrides Streamlit's red primary theme color */
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button,
    [data-testid="stForm"] button[kind="primaryFormSubmit"],
    [data-testid="stForm"] button {
        background: linear-gradient(135deg, #003087, #0070C0) !important;
        color: white !important;
        border: none !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 3px 10px rgba(0,48,135,0.25) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #002070, #005BA0) !important;
        box-shadow: 0 5px 16px rgba(0,48,135,0.35) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STICKY HEADER — always visible
# ============================================================

st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
    <div>
        <p class="header-title">🫀 CardioRisk AI</p>
        <p class="header-subtitle">
            Heart Disease Risk Assessment &nbsp;·&nbsp;
            Powered by Machine Learning
        </p>
    </div>
    <div class="header-badge">
        UCI Heart Disease Dataset &nbsp;·&nbsp; 920 Patients
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Disclaimer:</strong> This tool is for research and educational
    purposes only. It does not constitute a medical diagnosis.
    All predictions must be reviewed by a qualified clinician.
</div>
""", unsafe_allow_html=True)

if saved is None:
    st.error("Model file not found. Please run model.py first.")
    st.stop()

# Progress bar — always visible in sticky header
if not st.session_state.done:
    total        = len(QUESTIONS)
    current_step = st.session_state.step
    progress_pct = int((current_step / total) * 100)

    # Dot indicators
    dots_html = "<div class='dot-row'>"
    for i in range(total):
        if i < current_step:
            dots_html += "<span class='dot dot-done'></span>"
        elif i == current_step:
            dots_html += "<span class='dot dot-current'></span>"
        else:
            dots_html += "<span class='dot'></span>"
    dots_html += "</div>"

    st.markdown(
        f"<p class='progress-label'>"
        f"Question {min(current_step + 1, total)} of {total} "
        f"— {progress_pct}% complete</p>"
        f"<div class='progress-container'>"
        f"<div class='progress-fill' style='width:{progress_pct}%;'>"
        f"</div></div>"
        f"{dots_html}",
        unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end sticky-header

# ============================================================
# RESULTS PANEL
# ============================================================

if st.session_state.done and st.session_state.results:
    y_prob, shap_df, X_df, shap_vals, base_value = \
        st.session_state.results
    show_results(
        saved, st.session_state.answers,
        y_prob, shap_df, X_df, shap_vals, base_value)

    st.markdown("""
    <div class="section-divider">
        <span class="section-divider-dot"></span>
        <span class="section-divider-dot"></span>
        <span class="section-divider-dot"></span>
    </div>
    """, unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Start New Assessment", type="primary"):
            for key in ["step", "answers", "done",
                        "results", "error"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ============================================================
# PAGED QUESTION FORM
# One question per page, back/next navigation
# Enter key submits — st.form handles this natively
# ============================================================

elif not st.session_state.done:
    current_q = QUESTIONS[st.session_state.step]
    field     = current_q["field"]
    q_num     = st.session_state.step + 1
    total     = len(QUESTIONS)

    # ── Question card ──────────────────────────────────────
    st.markdown('<div class="question-card">', unsafe_allow_html=True)

    st.markdown(
        f"<p class='question-number'>Question {q_num} of {total}</p>"
        f"<p class='question-text'>{current_q['q']}</p>",
        unsafe_allow_html=True)

    # Show current saved answer if navigating back
    if field in st.session_state.answers:
        current_val = st.session_state.answers[field]
        if current_q["type"] == "choice":
            display_val = next(
                (k for k, v in current_q["choices"].items()
                 if v == current_val), str(current_val))
        elif current_q["type"] == "yesno":
            display_val = "Yes" if current_val == 1.0 else "No"
        else:
            display_val = (str(int(current_val))
                           if current_val == int(current_val)
                           else str(current_val))
        st.markdown(
            f"<p class='answer-label'>Current answer:</p>"
            f"<span class='answer-display'>✓ {display_val}</span>",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Show error if any
    if st.session_state.error:
        st.error(f"⚠️ {st.session_state.error}")
        st.session_state.error = ""

    # ── Answer form — Enter key triggers submission ────────
    # st.form captures Enter key press as a submit event
    next_label = (
        "Submit ✓" if st.session_state.step == total - 1
        else "Next →"
    )

    with st.form(key=f"q_form_{st.session_state.step}",
                 clear_on_submit=True):
        user_input = st.text_input(
            "Your answer:",
            label_visibility = "collapsed",
            placeholder      = "Type your answer and press Enter...",
        )
        submitted = st.form_submit_button(
            next_label,
            use_container_width=True,
            type="primary",
        )

    # Autofocus using streamlit components — runs in correct context
    # st.markdown scripts are sandboxed and can't access parent DOM
    # components.html() has access to window.parent.document
    import streamlit.components.v1 as components
    components.html("""
    <script>
        function focusInput() {
            const doc = window.parent.document;
            const inputs = doc.querySelectorAll(
                'input[type="text"]');
            if (inputs.length > 0) {
                inputs[0].focus();
                inputs[0].click();
            }
        }
        // 150ms lets Streamlit finish mounting the new question
        setTimeout(focusInput, 150);
    </script>
    """, height=0)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Back button — separate from form, styled grey ──────
    st.markdown("""
    <style>
        /* Back button gets a grey style to distinguish from Next */
        div[data-testid="stButton"]:first-of-type button {
            background: #6B8CAE !important;
            box-shadow: none !important;
        }
        div[data-testid="stButton"]:first-of-type button:hover {
            background: #4A6B8A !important;
        }
    </style>
    """, unsafe_allow_html=True)

    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        if st.button(
            "← Back",
            disabled            = (st.session_state.step == 0),
            use_container_width = True,
        ):
            st.session_state.step -= 1
            st.session_state.error = ""
            st.rerun()

    # ── Handle form submission ─────────────────────────────
    if submitted:
        if not user_input.strip():
            # Block progress — answer is required
            st.session_state.error = (
                "An answer is required before continuing. "
                "Please type your answer above.")
            st.rerun()
        else:
            try:
                val = parse_answer(user_input, current_q)
                st.session_state.answers[field] = val
                st.session_state.error = ""

                if st.session_state.step < total - 1:
                    # Move to next question
                    st.session_state.step += 1
                else:
                    # All questions answered — run prediction
                    with st.spinner("Analyzing clinical data..."):
                        try:
                            results = predict_patient(
                                saved,
                                st.session_state.answers)
                            st.session_state.results = results
                            st.session_state.done    = True
                        except Exception as e:
                            st.session_state.error = (
                                f"Prediction error: {e}")
                st.rerun()
            except ValueError as e:
                # Invalid answer — block progress, show error
                st.session_state.error = str(e)
                st.rerun()

    # ── Answered questions summary ─────────────────────────
    if st.session_state.answers:
        with st.expander(
            f"📋 View answered questions "
            f"({len(st.session_state.answers)}/{total})",
            expanded=False,
        ):
            for i, q in enumerate(QUESTIONS):
                f = q["field"]
                if f in st.session_state.answers:
                    val = st.session_state.answers[f]
                    if q["type"] == "choice":
                        disp = next(
                            (k for k, v in q["choices"].items()
                             if v == val), str(val))
                    elif q["type"] == "yesno":
                        disp = "Yes" if val == 1.0 else "No"
                    else:
                        disp = (str(int(val))
                                if val == int(val) else str(val))
                    col_q, col_v, col_edit = st.columns([3, 2, 1])
                    with col_q:
                        st.markdown(
                            f"<span style='font-size:0.85rem;"
                            f"color:#6B8CAE;'>Q{i+1}. "
                            f"{FEATURE_DESCRIPTIONS.get(f, f)}"
                            f"</span>",
                            unsafe_allow_html=True)
                    with col_v:
                        st.markdown(
                            f"<span style='font-size:0.85rem;"
                            f"font-weight:600;color:#003087;'>"
                            f"{disp}</span>",
                            unsafe_allow_html=True)
                    with col_edit:
                        if st.button("✏️", key=f"edit_{i}",
                                     help=f"Edit Q{i+1}"):
                            st.session_state.step  = i
                            st.session_state.error = ""
                            st.rerun()