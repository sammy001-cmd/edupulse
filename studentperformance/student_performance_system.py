"""
Student Academic Performance Prediction, Analysis, Intervention
Recommendation and Early Warning System Using Machine Learning.

Purpose:
    Provide an academic decision-support pipeline that estimates final
    performance, estimates probability of failure, identifies observable
    risk factors, and recommends institutional interventions.

Important:
    The bundled UCI dataset is a benchmark/development dataset. It is not
    Nigerian institutional data. The application is designed so that an
    institution can later replace the benchmark data with local records
    and retrain the models.
"""

import os
import io
import zipfile
import requests
import warnings
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"

CATEGORICAL_COLS = [
    "sex", "school", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "subject"
]

NUMERICAL_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences", "G1", "G2"
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
def download_uci_data(data_dir="data"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    print("[INFO] Dataset files not found. Downloading UCI benchmark dataset...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(data_dir)

    print("[INFO] Dataset downloaded and extracted.")


def load_data(data_dir="data"):
    mat_path = os.path.join(data_dir, "student-mat.csv")
    por_path = os.path.join(data_dir, "student-por.csv")

    if not (os.path.exists(mat_path) and os.path.exists(por_path)):
        download_uci_data(data_dir)

    mat_df = pd.read_csv(mat_path, sep=";")
    por_df = pd.read_csv(por_path, sep=";")

    mat_df["subject"] = "math"
    por_df["subject"] = "portuguese"

    df = pd.concat([mat_df, por_df], ignore_index=True)

    required = set(FEATURE_COLS + ["G3"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["fail"] = (df["G3"] < 10).astype(int)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Pass rate: {(1 - df['fail'].mean()) * 100:.2f}%")
    return df


# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------
def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERICAL_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])


def prepare_features(df):
    X = df[FEATURE_COLS].copy()
    y_class = df["fail"].copy()
    y_reg = df["G3"].copy()
    return X, y_class, y_reg


# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------
def train_classification_models(X, y):
    """
    Evaluate models on a genuinely unseen test set.

    The preprocessor is fitted only on X_train during evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    results = {}
    best_name = None
    best_score = -1

    for name, model in models.items():
        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = metrics

        print(
            f"[RESULT] {name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_name = name

    print(f"[BEST] Classification model: {best_name}")

    # Refit the selected model on all available data for deployment.
    deployment_preprocessor = build_preprocessor()
    X_all_processed = deployment_preprocessor.fit_transform(X)
    deployment_model = models[best_name]
    deployment_model.fit(X_all_processed, y)

    return deployment_model, deployment_preprocessor, results, best_name


def train_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }

    results = {}
    best_name = None
    best_score = -np.inf

    for name, model in models.items():
        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }
        results[name] = metrics

        print(
            f"[RESULT] {name}: "
            f"R2={metrics['r2']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}"
        )

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_name = name

    print(f"[BEST] Regression model: {best_name}")

    # Refit selected model on all available data for deployment.
    deployment_preprocessor = build_preprocessor()
    X_all_processed = deployment_preprocessor.fit_transform(X)
    deployment_model = models[best_name]
    deployment_model.fit(X_all_processed, y)

    return deployment_model, deployment_preprocessor, results, best_name


# ---------------------------------------------------------------------
# DECISION-SUPPORT LOGIC
# ---------------------------------------------------------------------
def generate_early_warning(risk_score):
    risk_score = float(risk_score)

    if risk_score >= 0.80:
        return "CRITICAL"
    if risk_score >= 0.60:
        return "HIGH"
    if risk_score >= 0.30:
        return "MODERATE"
    return "LOW"


def identify_risk_factors(row):
    """
    Identify observable academic/engagement indicators.

    These are decision-support rules, not claims of causal relationships.
    """
    factors = []

    if row["G2"] < 10:
        factors.append("Low second-period performance")
    elif row["G2"] < 12:
        factors.append("Below-target second-period performance")

    if row["G1"] < 10:
        factors.append("Low first-period performance")

    if row["failures"] > 0:
        factors.append("Previous academic failures")

    if row["absences"] > 10:
        factors.append("High absenteeism")
    elif row["absences"] > 5:
        factors.append("Elevated absenteeism")

    if row["studytime"] < 2:
        factors.append("Low study time")

    if row["goout"] >= 4:
        factors.append("High social activity")

    if row["health"] <= 2:
        factors.append("Low reported health status")

    if row["famrel"] <= 2:
        factors.append("Low reported family support")

    if not factors:
        factors.append("No major rule-based risk indicators identified")

    return factors


def identify_performance_factors(row):
    factors = []

    if row["G2"] >= 12:
        factors.append("Strong second-period performance")
    if row["G1"] >= 12:
        factors.append("Strong first-period performance")
    if row["absences"] <= 2:
        factors.append("Low absenteeism")
    if row["studytime"] >= 3:
        factors.append("Consistent study time")
    if row["failures"] == 0:
        factors.append("No previous academic failures")
    if row["higher"] == "yes":
        factors.append("Positive higher-education aspiration")

    return factors


def generate_intervention(row, risk_score, risk_factors):
    """
    Convert detected indicators into practical academic actions.
    """
    warning = generate_early_warning(risk_score)

    if warning == "CRITICAL":
        priority = "URGENT"
    elif warning == "HIGH":
        priority = "HIGH"
    elif warning == "MODERATE":
        priority = "MEDIUM"
    else:
        priority = "LOW"

    interventions = []

    if warning in {"CRITICAL", "HIGH"}:
        interventions.append(
            "Schedule an academic adviser/counsellor review"
        )

    if "Previous academic failures" in risk_factors:
        interventions.append(
            "Refer the student for targeted remedial or tutorial support"
        )

    if "High absenteeism" in risk_factors or "Elevated absenteeism" in risk_factors:
        interventions.append(
            "Place the student on attendance monitoring"
        )

    if "Low study time" in risk_factors:
        interventions.append(
            "Create a structured study timetable and peer-study plan"
        )

    if "Low first-period performance" in risk_factors:
        interventions.append(
            "Provide revision support for foundational topics"
        )

    if "Low second-period performance" in risk_factors:
        interventions.append(
            "Provide targeted tutoring based on weak assessment areas"
        )

    if "High social activity" in risk_factors:
        interventions.append(
            "Discuss time management and study-life balance"
        )

    if "Low reported health status" in risk_factors:
        interventions.append(
            "Refer the student to the institution's appropriate welfare support"
        )

    if "Low reported family support" in risk_factors:
        interventions.append(
            "Consider adviser-led support and appropriate family engagement"
        )

    if not interventions:
        interventions.append(
            "Continue routine academic monitoring"
        )

    return priority, interventions


# ---------------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------------
def get_failure_probability(model, X_processed):
    """
    Robustly locate class 1 (fail) instead of assuming probability column 1.
    """
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Expected failure class 1, found classes: {classes}")

    fail_index = classes.index(1)
    return model.predict_proba(X_processed)[:, fail_index]


def predict_students(df, classifier, regressor, preprocessor):
    X = df[FEATURE_COLS].copy()
    X_processed = preprocessor.transform(X)

    risk_scores = get_failure_probability(classifier, X_processed)
    predicted_grades = np.clip(
        regressor.predict(X_processed),
        0,
        20,
    )

    result = df.copy()
    result["Predicted_Grade"] = predicted_grades
    result["Risk_Score"] = risk_scores
    result["Early_Warning"] = [
        generate_early_warning(score) for score in risk_scores
    ]
    result["Risk_Factors"] = result.apply(
        lambda row: " | ".join(identify_risk_factors(row)),
        axis=1,
    )
    result["Intervention_Priority"] = [
        generate_intervention(
            result.iloc[i],
            risk_scores[i],
            identify_risk_factors(result.iloc[i]),
        )[0]
        for i in range(len(result))
    ]
    result["Interventions"] = [
        " | ".join(
            generate_intervention(
                result.iloc[i],
                risk_scores[i],
                identify_risk_factors(result.iloc[i]),
            )[1]
        )
        for i in range(len(result))
    ]

    return result


# ---------------------------------------------------------------------
# ARTIFACTS / MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== STUDENT ACADEMIC PERFORMANCE SYSTEM ===")
    print("Development dataset: UCI Student Performance benchmark")
    print("Institutional target: adaptable to Nigerian educational settings\n")

    df = load_data()
    X, y_class, y_reg = prepare_features(df)

    print("\n=== CLASSIFICATION: FAILURE RISK ===")
    clf_model, clf_preprocessor, cls_results, best_cls = (
        train_classification_models(X, y_class)
    )

    print("\n=== REGRESSION: FINAL PERFORMANCE ===")
    reg_model, reg_preprocessor, reg_results, best_reg = (
        train_regression_models(X, y_reg)
    )

    # Both models use the same feature schema. The deployment preprocessors
    # should be equivalent, but save one canonical preprocessor for the app.
    # It is refitted on all available data only AFTER test-set evaluation.
    preprocessor = clf_preprocessor

    # Retrain the regression model using the canonical preprocessor so the
    # saved regression model and saved preprocessor always match.
    X_all_processed = preprocessor.transform(X)
    reg_model.fit(X_all_processed, y_reg)

    # Full-dataset predictions are for dashboard screening/visualisation.
    # They are NOT used as evaluation metrics.
    df_predictions = predict_students(
        df,
        clf_model,
        reg_model,
        preprocessor,
    )

    output_cols = [
        "school", "sex", "age", "subject",
        "G1", "G2", "G3", "fail",
        "Predicted_Grade", "Risk_Score", "Early_Warning",
        "Risk_Factors", "Intervention_Priority", "Interventions",
    ]
    df_predictions[output_cols].to_csv(
        os.path.join(OUTPUT_DIR, "student_predictions.csv"),
        index=False,
    )

    joblib.dump(clf_model, os.path.join(OUTPUT_DIR, "classification_model.pkl"))
    joblib.dump(reg_model, os.path.join(OUTPUT_DIR, "regression_model.pkl"))
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.pkl"))

    metadata = {
        "project": (
            "Student Academic Performance Prediction, Analysis, "
            "Intervention Recommendation and Early Warning System "
            "Using Machine Learning"
        ),
        "development_dataset": "UCI Student Performance benchmark dataset",
        "localisation_note": (
            "The benchmark data is not Nigerian institutional data. "
            "The system architecture is designed for later retraining "
            "with Nigerian institutional records."
        ),
        "pass_threshold": 10,
        "risk_levels": {
            "LOW": "0-29%",
            "MODERATE": "30-59%",
            "HIGH": "60-79%",
            "CRITICAL": "80-100%",
        },
        "classification_results": cls_results,
        "regression_results": reg_results,
        "best_classification_model": best_cls,
        "best_regression_model": best_reg,
        "features": FEATURE_COLS,
        "note": (
            "Evaluation metrics are calculated on held-out test data. "
            "Full-dataset predictions are for dashboard screening and "
            "are not used to report model performance."
        ),
    }

    with open(
        os.path.join(OUTPUT_DIR, "model_results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n[SUCCESS] Training and artifact generation completed.")
    print(f"[INFO] Best classification: {best_cls}")
    print(f"[INFO] Best regression: {best_reg}")
    print("[INFO] Evaluation metrics use unseen test data.")
    print("[INFO] Full predictions are for dashboard screening only.")


if __name__ == "__main__":
    main()