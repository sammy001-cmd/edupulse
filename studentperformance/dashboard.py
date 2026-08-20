import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# FINAL MODEL INTEGRITY
# - Runtime artifacts: outputs_nigeria_tuned/
# - Probability field: low_performance_probability
# - Target: bottom-quartile low academic performance outcome
# - Inputs: Nigerian programme and early-level GPA records
# - Legacy international benchmark fields and obsolete risk fields are not used.

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="EduPulse | Academic Early Warning System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = "outputs_nigeria_tuned"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "model_results.json")
SCREENING_PATH = os.path.join(OUTPUT_DIR, "student_risk_screening.csv")

OPERATIONAL_DIR = "operational_data"
LIVE_CASES_PATH = os.path.join(OPERATIONAL_DIR, "live_cases.csv")
INTERVENTIONS_PATH = os.path.join(OPERATIONAL_DIR, "interventions.csv")
FOLLOWUPS_PATH = os.path.join(OPERATIONAL_DIR, "followups.csv")

os.makedirs(OPERATIONAL_DIR, exist_ok=True)
RISK_MODEL_PATH = os.path.join(OUTPUT_DIR, "risk_model.pkl")
CGPA_MODEL_PATH = os.path.join(OUTPUT_DIR, "cgpa_model.pkl")


# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #111827;
        --panel2: #172033;
        --border: #263244;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --good: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid var(--border);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 18px;
        border-radius: 14px;
        box-shadow: none;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: white;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.04em;
        color: white;
    }

    .brand-wordmark {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand-wordmark span {
        color: var(--accent);
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.35;
        margin-bottom: 18px;
    }

    .hero p {
        margin-top: 8px;
        color: var(--muted);
        font-size: 1rem;
    }

    .section-title {
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
    }

    .risk-card {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
    }

    .risk-low { border-left: 5px solid #22c55e; }
    .risk-moderate { border-left: 5px solid #f59e0b; }
    .risk-high { border-left: 5px solid #f97316; }
    .risk-critical { border-left: 5px solid #ef4444; }

    .small-note {
        color: var(--muted);
        font-size: 0.86rem;
    }

    .tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        background: #1e293b;
        border: 1px solid var(--border);
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #172033;
        color: white;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer-note {
        margin-top: 30px;
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #0f172a;
        color: var(--muted);
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    risk_model = joblib.load(RISK_MODEL_PATH)
    cgpa_model = joblib.load(CGPA_MODEL_PATH)
    return risk_model, cgpa_model


@st.cache_data
def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_screening():
    df = pd.read_csv(SCREENING_PATH)

    # Ensure saved screening records contain all dashboard-level fields.
    if "intervention_priority" not in df.columns and "risk_level" in df.columns:
        priority_map = {
            "LOW": "ROUTINE",
            "MODERATE": "MONITOR",
            "HIGH": "PRIORITY",
            "CRITICAL": "URGENT",
        }
        df["intervention_priority"] = df["risk_level"].map(priority_map).fillna("MONITOR")

    return df


def ensure_artifacts():
    required = {
        "Model validation metadata": RESULTS_PATH,
        "Student screening output": SCREENING_PATH,
        "Low-performance classifier": RISK_MODEL_PATH,
        "CGPA regression model": CGPA_MODEL_PATH,
    }
    missing = {label: path for label, path in required.items() if not os.path.exists(path)}

    if missing:
        st.error(
            "EduPulse cannot start because one or more trained Nigerian model artifacts are missing."
        )
        st.write(
            "Run `python nigeria_student_performance_system.py` locally to regenerate the tuned "
            "artifacts, or copy the required files into `outputs_nigeria_tuned/` before deployment."
        )
        for label, path in missing.items():
            st.write(f"- {label}: `{path}`")
        st.stop()


ensure_artifacts()
results = load_results()
screening = load_screening()
risk_model, cgpa_model = load_models()

FEATURES = results.get("features", [])
LOW_PERFORMANCE_CGPA = float(results.get("low_performance_cgpa_threshold", 0.0))
MEDIAN_CGPA = float(results.get("median_cgpa", 0.0))
UPPER_QUARTILE_CGPA = float(results.get("upper_quartile_cgpa", 0.0))
MODEL_PROB_THRESHOLD = float(results.get("selected_classifier_probability_threshold", 0.50))


# ------------------------------------------------------------
# LIVE INSTITUTIONAL WORKFLOW STATE
# ------------------------------------------------------------
# Historical training data stays in outputs_nigeria_tuned/.
# Current institutional cases are stored separately so they survive Streamlit reruns/navigation.

def _read_operational_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _write_operational_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8")


def load_live_cases():
    return _read_operational_csv(LIVE_CASES_PATH)


def save_live_cases(new_cases):
    """Persist newly assessed/uploaded students for watchlist/intervention use."""
    if new_cases is None or len(new_cases) == 0:
        return

    cases = new_cases.copy()

    if "student_id" not in cases.columns:
        cases.insert(
            0,
            "student_id",
            [f"CASE-{i:04d}" for i in range(1, len(cases) + 1)]
        )

    if "student_name" not in cases.columns:
        cases.insert(1, "student_name", "Not provided")

    existing = load_live_cases()
    combined = pd.concat([existing, cases], ignore_index=True)

    if "student_id" in combined.columns:
        combined["student_id"] = combined["student_id"].astype(str)
        combined = combined.drop_duplicates(subset=["student_id"], keep="last")

    _write_operational_csv(combined.reset_index(drop=True), LIVE_CASES_PATH)


def load_interventions():
    return _read_operational_csv(INTERVENTIONS_PATH)


def save_intervention_record(record):
    existing = load_interventions()
    new_df = pd.DataFrame([record])

    if not existing.empty and "student_id" in existing.columns:
        existing = existing[
            existing["student_id"].astype(str) != str(record.get("student_id", ""))
        ]

    combined = pd.concat([existing, new_df], ignore_index=True)
    _write_operational_csv(combined, INTERVENTIONS_PATH)


def load_followups():
    return _read_operational_csv(FOLLOWUPS_PATH)


def append_followup_record(record_df):
    existing = load_followups()
    combined = pd.concat([existing, record_df], ignore_index=True)
    _write_operational_csv(combined, FOLLOWUPS_PATH)


if "pending_assessment" not in st.session_state:
    st.session_state.pending_assessment = None

if "pending_bulk_screening" not in st.session_state:
    st.session_state.pending_bulk_screening = None


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def risk_level(prob):
    # Probability-only fallback. Main warnings combine predicted CGPA,
    # low-performance probability, and academic trajectory.
    if prob < 0.35:
        return "LOW"
    if prob < 0.60:
        return "MODERATE"
    if prob < 0.80:
        return "HIGH"
    return "CRITICAL"


def intervention_priority(level):
    return {
        "LOW": "ROUTINE",
        "MODERATE": "MONITOR",
        "HIGH": "PRIORITY",
        "CRITICAL": "URGENT",
    }.get(level, "MONITOR")


def student_factors(row):
    factors = []
    gpas = []

    for col, label in [
        ("gpa_100", "100-level GPA"),
        ("gpa_200", "200-level GPA"),
        ("gpa_300", "300-level GPA"),
    ]:
        if col in row and pd.notna(row[col]):
            value = float(row[col])
            gpas.append((label, value))
            if value < LOW_PERFORMANCE_CGPA:
                factors.append(f"{label} is below the Nigerian dataset low-performance boundary ({value:.2f}).")

    # Trajectory
    available = [v for _, v in gpas]
    if len(available) >= 2:
        if available[-1] < available[0] - 0.25:
            factors.append("Academic performance shows a declining GPA trajectory.")
        elif available[-1] > available[0] + 0.25:
            factors.append("Academic performance shows an improving GPA trajectory.")

    if not factors:
        factors.append("No major negative academic trajectory factor was detected from the available early-year GPA records.")

    return factors


def recommendations(row, level):
    recs = []

    gpa_values = []
    for c in ["gpa_100", "gpa_200", "gpa_300"]:
        if c in row and pd.notna(row[c]):
            gpa_values.append(float(row[c]))

    if level == "CRITICAL":
        recs.append("Schedule immediate academic adviser review.")
        recs.append("Place the student on a structured academic support plan.")
    elif level == "HIGH":
        recs.append("Arrange priority academic adviser consultation.")
        recs.append("Enroll the student in targeted tutorial or remedial support.")
    elif level == "MODERATE":
        recs.append("Monitor academic progress after the next assessment period.")
        recs.append("Recommend academic support where course performance is weak.")
    else:
        recs.append("Continue routine academic monitoring.")

    if len(gpa_values) >= 2 and gpa_values[-1] < gpa_values[0] - 0.25:
        recs.append("Review the causes of the declining academic trajectory with the student.")
    if any(g < LOW_PERFORMANCE_CGPA for g in gpa_values):
        recs.append("Review courses contributing to the low GPA and assign focused academic support.")

    return recs


def predict_record(df):
    prob = risk_model.predict_proba(df[FEATURES])[:, 1]
    cgpa = cgpa_model.predict(df[FEATURES])

    out = df.copy()
    out["predicted_final_cgpa"] = cgpa
    out["low_performance_probability"] = prob
    out["risk_level"] = [risk_level(x) for x in prob]
    out["intervention_priority"] = [intervention_priority(x) for x in out["risk_level"]]
    return out


def show_risk_summary(row):
    level = row["risk_level"]
    priority = row.get("intervention_priority", intervention_priority(level))
    css = f"risk-{level.lower()}"
    st.markdown(
        f"""
        <div class="risk-card {css}">
            <div style="font-size:0.84rem;color:#94a3b8">ACADEMIC RISK</div>
            <div style="font-size:1.55rem;font-weight:800;color:white;margin-top:4px">{level}</div>
            <div style="margin-top:8px;color:#cbd5e1">
                Estimated low-performance probability: <b>{row['low_performance_probability']:.1%}</b><br>
                Predicted final CGPA: <b>{row['predicted_final_cgpa']:.2f}</b><br>
                Intervention priority: <b>{priority}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def template_dataframe():
    row = {
        "student_id": "STU-001",
        "student_name": "Sample Student",
    }
    for f in FEATURES:
        if f == "programme":
            row[f] = "Computer Engineering"
        elif f == "entry_year":
            row[f] = 2023
        elif f == "gpa_100":
            row[f] = 2.80
        elif f == "gpa_200":
            row[f] = 2.45
        elif f == "gpa_300":
            row[f] = 2.20
        else:
            row[f] = ""
    return pd.DataFrame([row])



# ------------------------------------------------------------
# EXPLAINABILITY HELPERS
# ------------------------------------------------------------
def get_feature_importance_table(model_pipeline):
    """
    Extract transformed feature names and model importance/coefficients
    from a fitted sklearn Pipeline.
    """
    try:
        prep = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["model"]

        feature_names = prep.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            values = np.abs(coef[0] if coef.ndim > 1 else coef)
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        out = pd.DataFrame({
            "feature": feature_names,
            "importance": values
        }).sort_values("importance", ascending=False)

        # Clean sklearn prefixes for presentation
        out["feature"] = (
            out["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def explain_student(row):
    """
    Human-readable explanation from observable academic trajectory.
    This is not SHAP; it is a transparent decision-support explanation
    based on the student's recorded academic pattern.
    """
    notes = []
    vals = []

    for col, label in [
        ("gpa_100", "100-level GPA"),
        ("gpa_200", "200-level GPA"),
        ("gpa_300", "300-level GPA")
    ]:
        if col in row and pd.notna(row[col]):
            vals.append((label, float(row[col])))

    if vals:
        latest_label, latest = vals[-1]
        earliest_label, earliest = vals[0]

        if latest <= LOW_PERFORMANCE_CGPA:
            notes.append(
                f"{latest_label} ({latest:.2f}) is at or below the Nigerian dataset's "
                f"low-performance boundary ({LOW_PERFORMANCE_CGPA:.2f})."
            )
        elif latest <= MEDIAN_CGPA:
            notes.append(
                f"{latest_label} ({latest:.2f}) is below the dataset median ({MEDIAN_CGPA:.2f})."
            )
        else:
            notes.append(
                f"{latest_label} ({latest:.2f}) is above the dataset median ({MEDIAN_CGPA:.2f})."
            )

        delta = latest - earliest
        if delta <= -0.30:
            notes.append(
                f"Academic trajectory is declining by {abs(delta):.2f} GPA points from "
                f"{earliest_label} to {latest_label}."
            )
        elif delta >= 0.30:
            notes.append(
                f"Academic trajectory is improving by {delta:.2f} GPA points from "
                f"{earliest_label} to {latest_label}."
            )
        else:
            notes.append("Academic trajectory is relatively stable.")

    return notes


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="brand-wordmark">Edu<span>Pulse</span></div>
    <div class="brand-subtitle">Academic Performance Intelligence<br>and Early Warning System</div>
    """,
    unsafe_allow_html=True,
)


page = st.sidebar.radio(
    "Navigation",
    [
        "Assess Student",
        "Bulk Student Screening",
        "Early Warning Watchlist",
        "Intervention Management",
        "Follow-up Monitoring",
        "About EduPulse",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "EduPulse | Nigerian tertiary academic decision support"
)


# ------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------
if False and page == "Institutional Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Edu<span style="color:#38bdf8">Pulse</span></h1>
            <div style="font-size:1.15rem;font-weight:600;color:#cbd5e1;margin-top:6px">
                Academic Performance Intelligence & Early Warning System
            </div>
            <p>
                Institutional decision support for identifying emerging academic performance concerns,
                prioritising adviser attention, and supporting timely student intervention.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(screening)
    avg_pred = screening["predicted_final_cgpa"].mean()
    attention_mask = screening["risk_level"].isin(["MODERATE", "HIGH", "CRITICAL"])
    high_mask = screening["risk_level"].isin(["HIGH", "CRITICAL"])
    critical_mask = screening["risk_level"].eq("CRITICAL")

    attention_count = int(attention_mask.sum())
    high_count = int(high_mask.sum())
    critical_count = int(critical_mask.sum())
    attention_rate = attention_count / total if total else 0
    critical_rate = critical_count / total if total else 0

    st.markdown("### Institutional Academic Status")
    st.caption(
        "A high-level view of the screened student population and the records requiring academic attention."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students Screened", f"{total:,}")
    c2.metric("Average Predicted CGPA", f"{avg_pred:.2f}")
    c3.metric("Require Attention", f"{attention_count:,}", delta=f"{attention_rate:.1%} of cohort")
    c4.metric("High / Critical Risk", f"{high_count:,}")
    c5.metric("Critical Risk", f"{critical_count:,}", delta=f"{critical_rate:.1%} of cohort")

    st.markdown("### What Requires Attention Now")

    if critical_count > 0:
        st.error(
            f"{critical_count:,} student record(s) are currently classified as Critical. "
            "These records should receive the earliest academic adviser review."
        )
    elif high_count > 0:
        st.warning(
            f"{high_count:,} student record(s) are currently classified as High risk. "
            "Priority academic review is recommended."
        )
    else:
        st.success(
            "No High or Critical warning cases are present in the current screening output."
        )

    st.markdown("### Risk Distribution")
    risk_counts = (
        screening["risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
    )
    st.bar_chart(risk_counts)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("### Priority Student Queue")
        st.caption(
            "Students are ordered by low-performance probability so advisers can review the most concerning records first."
        )

        watch = screening[high_mask].copy()
        watch = watch.sort_values(
            ["risk_level", "low_performance_probability"],
            ascending=[True, False]
        )

        queue_cols = [
            c for c in [
                "student_id",
                "programme",
                "gpa_100",
                "gpa_200",
                "gpa_300",
                "predicted_final_cgpa",
                "low_performance_probability",
                "risk_level",
                "intervention_priority",
            ]
            if c in watch.columns
        ]

        if not watch.empty:
            display = watch[queue_cols].head(15).copy()
            if "low_performance_probability" in display.columns:
                display["low_performance_probability"] = display[
                    "low_performance_probability"
                ].map(lambda x: f"{x:.1%}")
            st.dataframe(display, width="stretch", hide_index=True)

            top_student = watch.iloc[0]
            st.markdown("#### Next Adviser Review")
            st.write(
                f"Start with **{top_student['student_id']}** "
                f"({top_student.get('programme', 'Programme not available')}). "
                f"Current warning: **{top_student['risk_level']}**; "
                f"estimated low-performance probability: "
                f"**{top_student['low_performance_probability']:.1%}**; "
                f"predicted final CGPA: **{top_student['predicted_final_cgpa']:.2f}**."
            )
        else:
            st.success("No High or Critical students are currently waiting for priority review.")

    with right:
        st.markdown("### Programme Risk Profile")
        st.caption(
            "Programme-level summaries help administrators identify where academic support demand is concentrated."
        )

        if "programme" in screening.columns:
            programme_profile = (
                screening.groupby("programme", dropna=False)
                .agg(
                    Students=("student_id", "count"),
                    Average_Risk=("low_performance_probability", "mean"),
                    Average_Predicted_CGPA=("predicted_final_cgpa", "mean"),
                    Require_Attention=("risk_level", lambda s: s.isin(["MODERATE", "HIGH", "CRITICAL"]).sum()),
                    High_Critical=("risk_level", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
                )
                .reset_index()
            )
            programme_profile["Attention_Rate"] = (
                programme_profile["Require_Attention"] /
                programme_profile["Students"]
            )
            programme_profile = programme_profile.sort_values(
                ["Attention_Rate", "Average_Risk"],
                ascending=False
            )

            programme_display = programme_profile.rename(columns={
                "programme": "Programme",
                "Average_Risk": "Average Risk",
                "Average_Predicted_CGPA": "Avg. Predicted CGPA",
                "Require_Attention": "Require Attention",
                "High_Critical": "High / Critical",
                "Attention_Rate": "Attention Rate",
            })

            st.dataframe(
                programme_display.style.format({
                    "Average Risk": "{:.1%}",
                    "Avg. Predicted CGPA": "{:.2f}",
                    "Attention Rate": "{:.1%}",
                }),
                width="stretch",
                hide_index=True,
            )

            highest = programme_profile.iloc[0]
            st.markdown("#### Programme Requiring Most Attention")
            st.write(
                f"**{highest['programme']}** currently has the highest relative concentration "
                f"of records requiring attention at **{highest['Attention_Rate']:.1%}**."
            )
        else:
            st.info("Programme information is not available in the model dataset.")

    st.markdown("### Recommended Institutional Workflow")
    w1, w2, w3, w4 = st.columns(4)
    w1.info("1. Review Critical and High cases")
    w2.info("2. Inspect individual academic trajectory")
    w3.info("3. Select appropriate intervention")
    w4.info("4. Monitor progress after follow-up")

    st.markdown(
        f"""
        <div class="footer-note">
        EduPulse is a decision-support system. Low academic performance is operationally defined
        from the Nigerian development dataset using its lower-quartile outcome boundary
        (<b>{LOW_PERFORMANCE_CGPA:.2f}</b>). Warning levels combine predicted final CGPA,
        low-performance probability, and academic trajectory. Final academic decisions remain
        with authorised institutional personnel.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# PAGE 2: SINGLE STUDENT RISK ASSESSMENT
# ------------------------------------------------------------
if page == "Assess Student":
    st.markdown(
        """
        <div class="hero">
            <h1>Assess Student</h1>
            <p>
                Enter a current student's academic record to predict later performance,
                analyse academic trajectory, generate an early warning, and recommend appropriate support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Student Identification")
    identity_a, identity_b = st.columns(2)
    with identity_a:
        student_id = st.text_input(
            "Student ID / Matric Number",
            value="STU-001",
            help="Enter the institution's unique identifier for this student."
        )
    with identity_b:
        student_name = st.text_input(
            "Student Name",
            value="",
            placeholder="Enter student's full name"
        )

    st.markdown("### Academic Record")
    values = {}

    col_a, col_b = st.columns(2)
    with col_a:
        if "programme" in FEATURES:
            choices = (
                sorted(screening["programme"].dropna().astype(str).unique().tolist())
                if "programme" in screening.columns else []
            )
            values["programme"] = st.selectbox(
                "Programme",
                choices if choices else ["Engineering"]
            )

    with col_b:
        if "entry_year" in FEATURES:
            values["entry_year"] = st.number_input(
                "Entry Year",
                min_value=2000,
                max_value=2035,
                value=2023,
                step=1
            )

    gpa_cols = st.columns(3)
    if "gpa_100" in FEATURES:
        with gpa_cols[0]:
            values["gpa_100"] = st.number_input(
                "100 Level GPA", 0.0, 5.0, 3.20, 0.01
            )
    if "gpa_200" in FEATURES:
        with gpa_cols[1]:
            values["gpa_200"] = st.number_input(
                "200 Level GPA", 0.0, 5.0, 2.80, 0.01
            )
    if "gpa_300" in FEATURES:
        with gpa_cols[2]:
            values["gpa_300"] = st.number_input(
                "300 Level GPA", 0.0, 5.0, 2.30, 0.01
            )

    rendered = {"programme", "entry_year", "gpa_100", "gpa_200", "gpa_300"}
    extras = [f for f in FEATURES if f not in rendered]
    if extras:
        st.markdown("### Additional Model Inputs")
        extra_cols = st.columns(2)
        for i, feature in enumerate(extras):
            with extra_cols[i % 2]:
                values[feature] = st.text_input(feature.replace("_", " ").title())

    if st.button("Analyse Student & Generate Early Warning", type="primary"):
        student_record = {
            "student_id": student_id.strip() or "UNASSIGNED",
            "student_name": student_name.strip() or "Not provided",
            **values,
        }
        assessed_df = predict_record(pd.DataFrame([student_record]))
        assessed_df["risk_evidence"] = assessed_df.apply(
            lambda r: " | ".join(explain_student(r)), axis=1
        )
        assessed_df["recommended_action"] = assessed_df.apply(
            lambda r: " | ".join(recommendations(r, r["risk_level"])), axis=1
        )
        st.session_state.pending_assessment = assessed_df.copy()

    # IMPORTANT: render the saved result OUTSIDE the Analyse button.
    # This makes the Add-to-Watchlist button work after Streamlit reruns.
    pending = st.session_state.pending_assessment

    if pending is not None and not pending.empty:
        assessed = pending.iloc[0]

        st.markdown("---")
        st.markdown("### Prediction Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Final CGPA", f"{assessed['predicted_final_cgpa']:.2f}")
        r2.metric(
            "Low Performance Probability",
            f"{assessed['low_performance_probability']:.1%}"
        )
        r3.metric("Early Warning", assessed["risk_level"])
        r4.metric("Intervention Priority", assessed["intervention_priority"])

        st.markdown("### Academic Trajectory")
        available_gpas = []
        for col, label in [
            ("gpa_100", "100 Level"),
            ("gpa_200", "200 Level"),
            ("gpa_300", "300 Level"),
        ]:
            if col in assessed and pd.notna(assessed[col]):
                available_gpas.append(
                    {"Academic Level": label, "GPA": float(assessed[col])}
                )

        if available_gpas:
            trajectory_df = pd.DataFrame(available_gpas)
            st.line_chart(trajectory_df.set_index("Academic Level")["GPA"])

            first_gpa = available_gpas[0]["GPA"]
            last_gpa = available_gpas[-1]["GPA"]
            delta = last_gpa - first_gpa

            t1, t2, t3 = st.columns(3)
            t1.metric("Earliest GPA", f"{first_gpa:.2f}")
            t2.metric("Latest GPA", f"{last_gpa:.2f}", delta=f"{delta:+.2f}")

            if delta >= 0.30:
                trajectory_status = "IMPROVING"
            elif delta <= -0.30:
                trajectory_status = "DECLINING"
            else:
                trajectory_status = "STABLE"
            t3.metric("Trajectory", trajectory_status)

        st.markdown("### Why This Student Was Flagged")
        for note in explain_student(assessed):
            st.write(f"- {note}")

        st.markdown("### Recommended Academic Intervention")
        for rec in recommendations(assessed, assessed["risk_level"]):
            st.write(f"- {rec}")

        st.markdown("### Next Institutional Action")
        if assessed["risk_level"] in ["MODERATE", "HIGH", "CRITICAL"]:
            if st.button(
                "Add Student to Early Warning Watchlist",
                type="primary",
                key="single_add_to_watchlist"
            ):
                save_live_cases(pending)
                saved = load_live_cases()
                sid = str(assessed.get("student_id", ""))
                if (
                    not saved.empty
                    and "student_id" in saved.columns
                    and sid in saved["student_id"].astype(str).values
                ):
                    st.success(
                        f"{assessed.get('student_name', 'Student')} ({sid}) is now on the Early Warning Watchlist."
                    )
                else:
                    st.error(
                        "The student could not be verified in the watchlist store. Please try again."
                    )
        else:
            st.success(
                "No immediate intervention is required. Continue routine academic monitoring."
            )

        st.markdown("### Academic Interpretation")
        st.info(
            "EduPulse provides academic decision support by identifying performance patterns "
            "that may require closer review. The early-warning result should be considered "
            "alongside the student's complete academic record, lecturer observations, and "
            "professional academic judgement. Decisions concerning support, referral, progression, "
            "or other institutional action remain with authorised academic personnel."
        )


# ------------------------------------------------------------
# PAGE 3: RECORDS UPLOAD
# ------------------------------------------------------------
elif page == "Bulk Student Screening":
    st.markdown(
        """
        <div class="hero">
            <h1>Bulk Student Screening</h1>
            <p>
                Upload current institutional student records to screen a class,
                department, or cohort for emerging academic risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Required Record Structure")
    st.caption(
        "Keep the required model column names exactly as shown. Include student_id "
        "or matric number and student_name so flagged cases can be linked to the correct student."
    )

    field_guide = []
    descriptions = {
        "programme": "Student programme or course of study",
        "entry_year": "Year of admission or entry",
        "gpa_100": "100 Level GPA",
        "gpa_200": "200 Level GPA",
        "gpa_300": "300 Level GPA",
    }

    for f in FEATURES:
        field_guide.append({
            "Required Field": f,
            "Description": descriptions.get(f, "Model input variable")
        })

    st.dataframe(pd.DataFrame(field_guide), hide_index=True, width="stretch")

    st.download_button(
        "Download EduPulse CSV Template",
        template_dataframe().to_csv(index=False).encode("utf-8"),
        file_name="edupulse_student_records_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Student CSV", type=["csv"])

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The uploaded CSV could not be read: {exc}")
            st.stop()

        st.markdown("### 2. Uploaded File Preview")
        st.dataframe(incoming.head(15), width="stretch", hide_index=True)

        missing_required = [f for f in FEATURES if f not in incoming.columns]
        duplicate_rows = int(incoming.duplicated().sum())
        available_required = [f for f in FEATURES if f in incoming.columns]
        missing_values = (
            int(incoming[available_required].isna().sum().sum())
            if available_required else 0
        )

        invalid_gpa = 0
        for gpa_col in ["gpa_100", "gpa_200", "gpa_300"]:
            if gpa_col in incoming.columns:
                vals = pd.to_numeric(incoming[gpa_col], errors="coerce")
                invalid_gpa += int(((vals < 0) | (vals > 5)).sum())

        st.markdown("### 3. Data Quality Check")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows Uploaded", f"{len(incoming):,}")
        q2.metric("Duplicate Rows", f"{duplicate_rows:,}")
        q3.metric("Missing Required Values", f"{missing_values:,}")
        q4.metric("Invalid GPA Values", f"{invalid_gpa:,}")

        if missing_required:
            st.error(
                "Missing required model column(s): " + ", ".join(missing_required)
            )
        elif invalid_gpa > 0:
            st.error(
                "One or more GPA values fall outside the expected 0.00 to 5.00 range."
            )
        else:
            if duplicate_rows > 0:
                st.warning(
                    f"{duplicate_rows} duplicate row(s) were detected."
                )
            if missing_values > 0:
                st.warning(
                    f"{missing_values} missing required value(s) were detected. "
                    "Correct institutional records where possible."
                )

            if st.button("Analyse Student Records", type="primary"):
                try:
                    analysed = predict_record(incoming)

                    if "student_id" not in analysed.columns:
                        analysed.insert(
                            0,
                            "student_id",
                            [f"UPLOAD-{i:04d}" for i in range(1, len(analysed) + 1)]
                        )
                    if "student_name" not in analysed.columns:
                        analysed.insert(1, "student_name", "Not provided")

                    analysed["risk_evidence"] = analysed.apply(
                        lambda r: " | ".join(explain_student(r)), axis=1
                    )
                    analysed["recommended_action"] = analysed.apply(
                        lambda r: " | ".join(recommendations(r, r["risk_level"])),
                        axis=1
                    )

                    risk_order = {
                        "CRITICAL": 0,
                        "HIGH": 1,
                        "MODERATE": 2,
                        "LOW": 3
                    }
                    analysed["_risk_order"] = (
                        analysed["risk_level"].map(risk_order).fillna(4)
                    )
                    analysed = analysed.sort_values(
                        ["_risk_order", "low_performance_probability"],
                        ascending=[True, False]
                    ).drop(columns=["_risk_order"])

                    st.session_state.pending_bulk_screening = analysed.copy()

                except Exception as exc:
                    st.error(f"Academic screening failed: {exc}")

    # IMPORTANT: render results/actions OUTSIDE the Analyse button.
    analysed = st.session_state.pending_bulk_screening

    if analysed is not None and not analysed.empty:
        total_students = len(analysed)
        attention = int(
            analysed["risk_level"]
            .isin(["MODERATE", "HIGH", "CRITICAL"])
            .sum()
        )
        high = int((analysed["risk_level"] == "HIGH").sum())
        critical = int((analysed["risk_level"] == "CRITICAL").sum())

        st.markdown("---")
        st.markdown("### 4. Screening Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Students Analysed", f"{total_students:,}")
        s2.metric("Require Attention", f"{attention:,}")
        s3.metric("High Risk", f"{high:,}")
        s4.metric("Critical Risk", f"{critical:,}")

        st.markdown("### 5. Prioritised Screening Results")
        display_cols = [
            c for c in [
                "student_id",
                "student_name",
                "programme",
                "gpa_100",
                "gpa_200",
                "gpa_300",
                "predicted_final_cgpa",
                "low_performance_probability",
                "risk_level",
                "intervention_priority",
            ]
            if c in analysed.columns
        ]

        display_df = analysed[display_cols].copy()
        if "low_performance_probability" in display_df.columns:
            display_df["low_performance_probability"] = display_df[
                "low_performance_probability"
            ].map(lambda x: f"{x:.1%}")

        st.dataframe(display_df, width="stretch", hide_index=True)

        flagged = analysed[
            analysed["risk_level"].isin(["MODERATE", "HIGH", "CRITICAL"])
        ].copy()

        st.markdown("### 6. Early Warning Action")
        if flagged.empty:
            st.success(
                "No students in this screening require an early-warning case."
            )
        else:
            if st.button(
                f"Add {len(flagged)} Flagged Student(s) to Watchlist",
                type="primary",
                key="bulk_add_to_watchlist"
            ):
                save_live_cases(flagged)
                saved = load_live_cases()

                flagged_ids = set(flagged["student_id"].astype(str))
                saved_ids = (
                    set(saved["student_id"].astype(str))
                    if not saved.empty and "student_id" in saved.columns
                    else set()
                )
                verified = flagged_ids.intersection(saved_ids)

                if len(verified) == len(flagged_ids):
                    st.success(
                        f"{len(verified)} flagged student(s) are now on the Early Warning Watchlist."
                    )
                else:
                    st.error(
                        f"Only {len(verified)} of {len(flagged_ids)} cases could be verified. "
                        "Please retry the watchlist action."
                    )

        st.markdown("### 7. Export Screening Results")
        st.download_button(
            "Download EduPulse Screening Results",
            analysed.to_csv(index=False).encode("utf-8"),
            file_name="edupulse_academic_screening_results.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# PAGE 4: EARLY WARNING WATCHLIST
# ------------------------------------------------------------
elif page == "Early Warning Watchlist":
    st.markdown(
        """
        <div class="hero">
            <h1>Early Warning Watchlist</h1>
            <p>
                Prioritise students requiring academic review, inspect the evidence behind each warning,
                and move directly into an intervention decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_cases = load_live_cases()

    if live_cases.empty:
        st.info(
            "No new student cases have been assessed yet. Go to Student Risk Assessment or "
            "Student Records Upload first. Only students assessed during actual use appear here."
        )
        st.stop()

    st.markdown("### Watchlist Filters")
    f1, f2 = st.columns([1, 1])

    with f1:
        selected_levels = st.multiselect(
            "Warning level",
            ["MODERATE", "HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"],
        )

    with f2:
        if "programme" in live_cases.columns:
            programmes = ["All Programmes"] + sorted(
                live_cases["programme"].dropna().astype(str).unique().tolist()
            )
            selected_programme = st.selectbox("Programme", programmes)
        else:
            selected_programme = "All Programmes"

    filtered = live_cases[live_cases["risk_level"].isin(selected_levels)].copy()

    if selected_programme != "All Programmes" and "programme" in filtered.columns:
        filtered = filtered[filtered["programme"].astype(str) == selected_programme]

    risk_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    filtered["_risk_order"] = filtered["risk_level"].map(risk_order).fillna(4)
    filtered = filtered.sort_values(
        ["_risk_order", "low_performance_probability"],
        ascending=[True, False]
    ).drop(columns=["_risk_order"])

    st.markdown("### Current Warning Queue")
    a, b, c, d = st.columns(4)
    a.metric("Students on Watchlist", f"{len(filtered):,}")
    b.metric("Moderate", int((filtered["risk_level"] == "MODERATE").sum()))
    c.metric("High", int((filtered["risk_level"] == "HIGH").sum()))
    d.metric("Critical", int((filtered["risk_level"] == "CRITICAL").sum()))

    if filtered.empty:
        st.info("No students match the selected warning criteria.")
    else:
        display_cols = [
            c for c in [
                "student_id",
                "programme",
                "gpa_100",
                "gpa_200",
                "gpa_300",
                "predicted_final_cgpa",
                "low_performance_probability",
                "risk_level",
                "intervention_priority",
            ]
            if c in filtered.columns
        ]

        display_df = filtered[display_cols].copy()
        if "low_performance_probability" in display_df.columns:
            display_df["low_performance_probability"] = display_df[
                "low_performance_probability"
            ].map(lambda x: f"{x:.1%}")

        st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("### Student Drill Down")
        student_id = st.selectbox(
            "Select student for review",
            filtered["student_id"].tolist(),
            key="watchlist_student_select"
        )
        row = filtered[filtered["student_id"] == student_id].iloc[0]

        r1, r2 = st.columns([1, 1])

        with r1:
            show_risk_summary(row)

            st.markdown("#### Academic Record")
            record_fields = []
            for col, label in [
                ("programme", "Programme"),
                ("gpa_100", "100 Level GPA"),
                ("gpa_200", "200 Level GPA"),
                ("gpa_300", "300 Level GPA"),
                ("predicted_final_cgpa", "Predicted Final CGPA"),
            ]:
                if col in row and pd.notna(row[col]):
                    value = row[col]
                    if isinstance(value, (float, np.floating)) and col != "programme":
                        value = f"{value:.2f}"
                    record_fields.append({"Field": label, "Value": value})

            st.dataframe(
                pd.DataFrame(record_fields),
                hide_index=True,
                width="stretch"
            )

        with r2:
            st.markdown("#### Why This Student Was Flagged")
            for factor in explain_student(row):
                st.write(f"- {factor}")

            st.markdown("#### Recommended Adviser Action")
            for rec in recommendations(row, row["risk_level"]):
                st.write(f"- {rec}")

            if row["risk_level"] == "CRITICAL":
                st.error("Urgent academic adviser review recommended.")
            elif row["risk_level"] == "HIGH":
                st.warning("Priority academic adviser review recommended.")
            else:
                st.info("Monitor this student's academic progression closely.")

        st.markdown("### What Happens Next?")
        st.write(
            "After reviewing a flagged student, continue to Intervention Management to record "
            "the academic support action, follow-up period, and adviser note."
        )

        st.markdown("### Review Guidance")
        st.caption(
            "A watchlist entry is a decision-support alert, not a disciplinary label. "
            "The adviser should review the student's wider academic circumstances before recording an intervention."
        )


# ------------------------------------------------------------
# PAGE 5: INTERVENTION MANAGEMENT
# ------------------------------------------------------------
elif page == "Intervention Management":
    st.markdown(
        """
        <div class="hero">
            <h1>Intervention Management</h1>
            <p>
                Convert an academic warning into a documented support plan.
                Review the evidence, select an intervention, define follow-up, and prepare the adviser record.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_cases = load_live_cases()

    if live_cases.empty:
        st.info(
            "No assessed students are available yet. Run Assess Student or Bulk Student Screening first."
        )
        st.stop()

    candidates = live_cases[
        live_cases["risk_level"].isin(["MODERATE", "HIGH", "CRITICAL"])
    ].copy()

    risk_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    candidates["_risk_order"] = candidates["risk_level"].map(risk_order).fillna(3)
    candidates = candidates.sort_values(
        ["_risk_order", "low_performance_probability"],
        ascending=[True, False]
    ).drop(columns=["_risk_order"])

    if candidates.empty:
        st.success("No students currently require an intervention review.")
    else:
        st.markdown("### 1. Select Student")
        student_id = st.selectbox(
            "Student requiring review",
            candidates["student_id"].tolist(),
            key="intervention_student_select"
        )

        row = candidates[candidates["student_id"] == student_id].iloc[0]

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 2. Review Warning")
            show_risk_summary(row)

            st.markdown("#### Evidence")
            for factor in explain_student(row):
                st.write(f"- {factor}")

        with right:
            st.markdown("### 3. Suggested Support Plan")
            suggested = recommendations(row, row["risk_level"])
            for i, rec in enumerate(suggested, start=1):
                st.write(f"{i}. {rec}")

        st.markdown("### 4. Record Adviser Decision")
        d1, d2 = st.columns(2)

        with d1:
            intervention_type = st.selectbox(
                "Selected intervention",
                [
                    "Academic adviser consultation",
                    "Targeted tutorial or remedial support",
                    "Structured academic support plan",
                    "Course performance review",
                    "Routine academic monitoring",
                ],
            )

        with d2:
            followup_period = st.selectbox(
                "Follow-up period",
                [
                    "After next assessment",
                    "Within 2 weeks",
                    "Within 4 weeks",
                    "End of semester",
                ],
            )

        intervention_status = st.selectbox(
            "Intervention status",
            ["Planned", "Initiated", "In Progress", "Completed"]
        )

        adviser_note = st.text_area(
            "Adviser note",
            placeholder=(
                "Record the reason for the selected intervention, relevant academic context, "
                "and what should be checked during follow-up."
            )
        )

        if st.button("Prepare Intervention Record", type="primary"):
            record = {
                "student_id": student_id,
                "student_name": row.get("student_name", "Not provided"),
                "programme": row.get("programme", "N/A"),
                "warning_level": row["risk_level"],
                "low_performance_probability": float(row["low_performance_probability"]),
                "predicted_final_cgpa": float(row["predicted_final_cgpa"]),
                "intervention_priority": row.get(
                    "intervention_priority",
                    intervention_priority(row["risk_level"])
                ),
                "selected_intervention": intervention_type,
                "follow_up": followup_period,
                "status": intervention_status,
                "adviser_note": adviser_note,
            }

            save_intervention_record(record)

            st.markdown("### 5. Prepared Intervention Record")
            st.success(
                "The intervention record has been prepared for authorised academic follow-up."
            )

            summary_df = pd.DataFrame([
                {"Field": "Student ID", "Value": record["student_id"]},
                {"Field": "Programme", "Value": record["programme"]},
                {"Field": "Warning Level", "Value": record["warning_level"]},
                {"Field": "Intervention Priority", "Value": record["intervention_priority"]},
                {"Field": "Predicted Final CGPA", "Value": f"{record['predicted_final_cgpa']:.2f}"},
                {"Field": "Low Performance Probability", "Value": f"{record['low_performance_probability']:.1%}"},
                {"Field": "Selected Intervention", "Value": record["selected_intervention"]},
                {"Field": "Follow-up", "Value": record["follow_up"]},
                {"Field": "Status", "Value": record["status"]},
                {"Field": "Adviser Note", "Value": record["adviser_note"] or "No note entered"},
            ])

            st.dataframe(summary_df, hide_index=True, width="stretch")

            record_csv = pd.DataFrame([record]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Intervention Record",
                record_csv,
                file_name=f"edupulse_intervention_{student_id}.csv",
                mime="text/csv",
            )

            st.caption(
                "EduPulse prepares the intervention record for the current session. "
                "A production institutional deployment should store adviser actions and follow-up history "
                "in an authorised persistent database."
            )


# ------------------------------------------------------------
# PAGE 6: STUDENT PROGRESS
# ------------------------------------------------------------
elif page == "Follow-up Monitoring":
    st.markdown(
        """
        <div class="hero">
            <h1>Follow-up Monitoring</h1>
            <p>
                Review the student after intervention and record whether academic performance is improving, stable, or declining.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Identify Student")

    intervention_records = load_interventions()
    live_cases = load_live_cases()

    if not intervention_records.empty:
        student_id = st.selectbox(
            "Student with recorded intervention",
            intervention_records["student_id"].astype(str).tolist()
        )
        matching = live_cases[live_cases["student_id"].astype(str) == str(student_id)]
        if not matching.empty:
            selected_case = matching.iloc[0]
            available_previous = [
                selected_case.get("gpa_300"),
                selected_case.get("gpa_200"),
                selected_case.get("gpa_100"),
            ]
            available_previous = [x for x in available_previous if pd.notna(x)]
            default_previous_gpa = float(available_previous[0]) if available_previous else 1.80
        else:
            default_previous_gpa = 1.80
    else:
        st.info(
            "No intervention has been recorded yet. You can still demonstrate progress monitoring manually, "
            "or create an intervention first from Intervention Management."
        )
        student_id = st.text_input("Student ID", value="STU-001")
        default_previous_gpa = 1.80

    st.markdown("### 2. Record Academic Change")
    c1, c2 = st.columns(2)

    with c1:
        previous_gpa = st.number_input(
            "GPA before intervention",
            min_value=0.0,
            max_value=5.0,
            value=float(default_previous_gpa),
            step=0.01
        )

    with c2:
        current_gpa = st.number_input(
            "Current GPA or assessment equivalent",
            min_value=0.0,
            max_value=5.0,
            value=2.30,
            step=0.01
        )

    intervention = st.selectbox(
        "Intervention received",
        [
            "Academic adviser consultation",
            "Targeted tutorial or remedial support",
            "Structured academic support plan",
            "Course performance review",
            "Routine academic monitoring",
        ],
    )

    followup_note = st.text_area(
        "Follow-up note",
        placeholder="Example: Student attended two tutorial sessions, met the academic adviser, submitted pending coursework, and showed improved attendance during the follow-up period."
    )

    if st.button("Evaluate Academic Progress", type="primary"):
        change = current_gpa - previous_gpa

        if change >= 0.30:
            status = "IMPROVING"
        elif change <= -0.30:
            status = "DECLINING"
        else:
            status = "STABLE"

        st.markdown("### 3. Progress Outcome")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Previous GPA", f"{previous_gpa:.2f}")
        p2.metric("Current GPA", f"{current_gpa:.2f}")
        p3.metric("GPA Change", f"{change:+.2f}")
        p4.metric("Progress Status", status)

        progress_df = pd.DataFrame({
            "Stage": ["Before Intervention", "Current"],
            "GPA": [previous_gpa, current_gpa]
        })
        st.line_chart(progress_df.set_index("Stage")["GPA"])

        if status == "IMPROVING":
            st.success(
                "The student's recorded academic performance has improved since the previous observation. Continue the current support plan and monitor whether the improvement is sustained."
            )
        elif status == "DECLINING":
            st.error(
                "The student's recorded academic performance has declined. "
                "Further academic review is recommended."
            )
        else:
            st.info(
                "The student's academic performance is relatively stable. "
                "Continue monitoring according to the agreed follow-up plan."
            )

        st.markdown("### 4. Follow-up Record")
        followup_record = pd.DataFrame([{
            "student_id": student_id,
            "intervention": intervention,
            "previous_gpa": previous_gpa,
            "current_gpa": current_gpa,
            "gpa_change": change,
            "progress_status": status,
            "followup_note": followup_note,
        }])

        append_followup_record(followup_record)

        st.dataframe(followup_record, hide_index=True, width="stretch")

        st.download_button(
            "Download Follow-up Record",
            followup_record.to_csv(index=False).encode("utf-8"),
            file_name=f"edupulse_followup_{student_id}.csv",
            mime="text/csv",
        )

        st.caption(
            "Progress monitoring records observed academic change. "
            "It does not prove that an intervention caused the improvement or decline."
        )


# ------------------------------------------------------------
# PAGE 7: EXPLAINABILITY
# ------------------------------------------------------------
elif page == "Explainability":
    st.markdown(
        """
        <div class="hero">
            <h1>Model Explainability</h1>
            <p>
                Understand which academic variables influence predictions most and inspect the
                observable academic evidence behind an individual student's warning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Classification Model Drivers")
    st.caption(
        "These values show predictive importance within the selected low-performance classifier. "
        "They do not establish that a variable causes academic success or failure."
    )

    cls_imp = get_feature_importance_table(risk_model)

    if not cls_imp.empty:
        top_cls = cls_imp.head(12).copy()
        e1, e2 = st.columns([1, 1])

        with e1:
            st.dataframe(
                top_cls.style.format({"importance": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )

        with e2:
            st.bar_chart(top_cls.set_index("feature")["importance"])
    else:
        st.info("Feature importance is not available for the selected classification model.")

    st.markdown("### CGPA Prediction Model Drivers")
    st.caption(
        "The regression importance profile shows which variables the selected model relies on most "
        "when estimating later or final CGPA."
    )

    reg_imp = get_feature_importance_table(cgpa_model)

    if not reg_imp.empty:
        top_reg = reg_imp.head(12).copy()
        e3, e4 = st.columns([1, 1])

        with e3:
            st.dataframe(
                top_reg.style.format({"importance": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )

        with e4:
            st.bar_chart(top_reg.set_index("feature")["importance"])
    else:
        st.info("Feature importance is not available for the selected regression model.")

    st.markdown("### Individual Student Explanation")
    student_id = st.selectbox(
        "Select screened student",
        screening["student_id"].tolist(),
        key="explain_student_select"
    )

    row = screening[screening["student_id"] == student_id].iloc[0]

    x1, x2 = st.columns([1, 1])

    with x1:
        show_risk_summary(row)

        academic_rows = []
        for col, label in [
            ("gpa_100", "100 Level GPA"),
            ("gpa_200", "200 Level GPA"),
            ("gpa_300", "300 Level GPA"),
            ("predicted_final_cgpa", "Predicted Final CGPA"),
        ]:
            if col in row and pd.notna(row[col]):
                academic_rows.append({
                    "Academic Indicator": label,
                    "Value": f"{float(row[col]):.2f}"
                })

        st.dataframe(
            pd.DataFrame(academic_rows),
            width="stretch",
            hide_index=True
        )

    with x2:
        st.markdown("#### Observable Evidence")
        for note in explain_student(row):
            st.write(f"- {note}")

        st.markdown("#### Interpretation")
        st.write(
            "EduPulse combines these observable academic patterns with the trained model outputs. "
            "The explanation is intended to help an adviser understand the warning, not to provide "
            "a causal diagnosis of why the student is performing at a particular level."
        )

    st.markdown("### Responsible Use of Explainability")
    st.info(
        "Feature importance indicates predictive influence within the trained model. "
        "It should not be interpreted as proof that a particular programme or GPA measurement "
        "independently causes a student's academic outcome."
    )


# ------------------------------------------------------------
# PAGE 8: MODEL VALIDATION
# ------------------------------------------------------------
elif page == "Model Validation":
    st.markdown(
        """
        <div class="hero">
            <h1>Model Validation</h1>
            <p>
                Technical evidence for the Nigerian machine learning pipeline used by EduPulse,
                including target definition, model comparison, evaluation metrics, and deployment limits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Development Dataset")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nigerian Records", f"{results.get('records', 0):,}")
    m2.metric("Low-Performance Boundary", f"{LOW_PERFORMANCE_CGPA:.2f}")
    m3.metric("Selected Classifier", results.get("selected_classifier", "N/A"))
    m4.metric("Selected Regressor", results.get("selected_regressor", "N/A"))

    st.caption(
        results.get(
            "target_definition",
            "Low academic performance is operationally defined from the distribution of final CGPA in the development dataset."
        )
    )

    st.markdown("### Variables Used")
    st.write(", ".join(FEATURES))

    st.markdown("### Classification Model Comparison")
    class_results = results.get("classification_results", {})

    if class_results:
        rows = []
        for model, payload in class_results.items():
            metrics = payload.get("tuned_threshold_metrics", {})
            rows.append({
                "Model": model,
                "CV F1": payload.get("cv_f1_mean"),
                "Threshold": payload.get("tuned_probability_threshold"),
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "Specificity": metrics.get("specificity"),
                "F1": metrics.get("f1"),
                "ROC AUC": metrics.get("roc_auc"),
            })

        cls_df = pd.DataFrame(rows)

        st.dataframe(
            cls_df.style.format({
                "CV F1": "{:.3f}",
                "Threshold": "{:.3f}",
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "Specificity": "{:.3f}",
                "F1": "{:.3f}",
                "ROC AUC": lambda x: "" if pd.isna(x) else f"{x:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )

        selected_cls = results.get("selected_classifier")
        selected_payload = class_results.get(selected_cls, {})
        selected_metrics = selected_payload.get("tuned_threshold_metrics", {})

        if selected_metrics:
            st.markdown("#### Selected Classifier Interpretation")
            st.write(
                f"EduPulse selected **{selected_cls}** using cross-validated F1 performance. "
                f"On the held-out evaluation data, recall was "
                f"**{selected_metrics.get('recall', 0):.3f}** and F1 was "
                f"**{selected_metrics.get('f1', 0):.3f}**. "
                "These metrics are emphasised because the purpose of the classifier is to identify "
                "students in the low-performance group, not simply maximise majority-class accuracy."
            )

    st.markdown("### CGPA Regression Model Comparison")
    reg_results = results.get("regression_results", {})

    if reg_results:
        rows = []
        for model, metrics in reg_results.items():
            rows.append({
                "Model": model,
                "CV RMSE": metrics.get("cv_rmse_mean"),
                "Test R²": metrics.get("test_r2"),
                "Test RMSE": metrics.get("test_rmse"),
                "Test MAE": metrics.get("test_mae"),
            })

        reg_df = pd.DataFrame(rows)

        st.dataframe(
            reg_df.style.format({
                "CV RMSE": "{:.3f}",
                "Test R²": "{:.3f}",
                "Test RMSE": "{:.3f}",
                "Test MAE": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )

        selected_reg = results.get("selected_regressor")
        selected_reg_metrics = reg_results.get(selected_reg, {})

        if selected_reg_metrics:
            st.markdown("#### Selected Regressor Interpretation")
            st.write(
                f"EduPulse selected **{selected_reg}** based on cross-validated regression error. "
                f"The held-out R² was **{selected_reg_metrics.get('test_r2', 0):.3f}**, "
                f"RMSE was **{selected_reg_metrics.get('test_rmse', 0):.3f}**, and "
                f"MAE was **{selected_reg_metrics.get('test_mae', 0):.3f}**."
            )

    st.markdown("### How the Final Warning Is Produced")
    st.write(
        "The early-warning level is not determined by classification probability alone. "
        "EduPulse combines predicted final CGPA, probability of a low-performance outcome, "
        "and academic trajectory to assign Low, Moderate, High, or Critical attention levels."
    )

    st.markdown("### Responsible Interpretation")
    st.warning(
        "The model was developed using Nigerian university academic records, but predictive performance "
        "should be revalidated before deployment in another institution. Differences in grading systems, "
        "programmes, student populations, and academic policies may change model behaviour."
    )

    st.info(
        "EduPulse is an academic decision-support system. Model outputs must not be used as automatic "
        "grounds for progression, suspension, disciplinary action, or compulsory counselling."
    )


# ------------------------------------------------------------
# PAGE 9: SYSTEM GUIDE
# ------------------------------------------------------------
elif page == "About EduPulse":
    st.markdown(
        """
        <div class="hero">
            <h1>About EduPulse</h1>
            <div style="font-size:1.10rem;font-weight:600;color:#cbd5e1;margin-top:6px">
                Academic Performance Intelligence & Early Warning System
            </div>
            <p>
                A machine learning based academic decision-support platform for Nigerian tertiary education.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How EduPulse Is Used in a Nigerian Institution")
    st.write(
        "The historical Nigerian dataset is used only to train and validate the machine learning models. "
        "During real use, the institution does not analyse those old training students again. "
        "A lecturer or academic adviser enters a NEW student's current academic record, or uploads current "
        "departmental records. EduPulse then predicts later performance, generates an early warning, "
        "recommends an intervention, and supports follow-up."
    )

    st.markdown("### What Problem Does EduPulse Solve?")
    st.write(
        "Tertiary institutions already collect academic records, but those records are often used mainly "
        "for reporting completed performance. EduPulse uses earlier academic records to help identify students "
        "whose performance pattern may require attention before the final outcome becomes more severe."
    )

    st.markdown("### Who Uses EduPulse?")
    users = pd.DataFrame([
        {
            "User": "Lecturer / Academic Adviser",
            "Use": "Review warnings, inspect academic trajectory, select support actions, and monitor follow-up."
        },
        {
            "User": "Academic Administrator",
            "Use": "Monitor institution or programme-level warning patterns and prioritise academic support."
        },
        {
            "User": "Student",
            "Use": "Receive academic guidance and support communicated by authorised personnel."
        },
    ])
    st.dataframe(users, width="stretch", hide_index=True)

    st.markdown("### EduPulse Workflow")
    workflow = pd.DataFrame([
        {"Stage": 1, "Process": "Student academic records are supplied to the system."},
        {"Stage": 2, "Process": "Required variables are validated and preprocessed."},
        {"Stage": 3, "Process": "The regression model predicts later or final CGPA."},
        {"Stage": 4, "Process": "The classifier estimates low-performance probability."},
        {"Stage": 5, "Process": "Academic trajectory is analysed across earlier GPA records."},
        {"Stage": 6, "Process": "EduPulse assigns an early-warning level and intervention priority."},
        {"Stage": 7, "Process": "Academic personnel review the warning and record an intervention."},
        {"Stage": 8, "Process": "Student progress can be monitored during follow-up."},
    ])
    st.dataframe(workflow, width="stretch", hide_index=True)

    st.markdown("### Main Modules")
    modules = pd.DataFrame([
        {"Module": "Assess Student", "Purpose": "Enter a new student's current record and predict later academic performance."},
        {"Module": "Bulk Student Screening", "Purpose": "Screen a class, department, or cohort using current institutional records."},
        {"Module": "Early Warning Watchlist", "Purpose": "Show only newly assessed students who require academic attention."},
        {"Module": "Intervention Management", "Purpose": "Record the academic support selected for a flagged student."},
        {"Module": "Follow-up Monitoring", "Purpose": "Monitor the student's academic change after intervention."},
    ])
    st.dataframe(modules, width="stretch", hide_index=True)

    st.markdown("### Responsible Use")
    st.warning(
        "EduPulse is a decision-support system. A warning level is not proof that a student will fail "
        "and must not be used as an automatic basis for progression, suspension, disciplinary action, "
        "or compulsory counselling."
    )

    st.markdown("### Nigerian Development Context")
    st.write(
        "The final machine learning pipeline was developed using Nigerian university academic records. "
        "However, any institution adopting EduPulse should revalidate the models using its own historical "
        "student records and align warning definitions with its academic regulations."
    )

    st.markdown("### System Identity")
    st.write(
        "Product name: EduPulse\n\n"
        "System type: Academic Performance Prediction, Intervention Recommendation and Early Warning System\n\n"
        "Primary users: Lecturers, academic advisers, and authorised academic administrators\n\n"
        "Core technology: Python, Pandas, NumPy, Scikit-learn, Joblib, and Streamlit"
    )