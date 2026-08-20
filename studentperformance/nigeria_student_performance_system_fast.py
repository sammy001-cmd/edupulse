import os
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve
)

DATA_PATH = "data/nigeria_student_performance.xlsx"
OUTPUT_DIR = "outputs_nigeria_tuned"

# Operational early-warning definition:
# students whose final CGPA falls in the lowest 25% of this Nigerian dataset.
LOW_PERFORMANCE_QUANTILE = 0.25


# ------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------
def _clean_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def _find_column(columns, candidates):
    normalized = {_clean_name(c): c for c in columns}

    for candidate in candidates:
        key = _clean_name(candidate)
        if key in normalized:
            return normalized[key]

    for c in columns:
        nc = _clean_name(c)
        for candidate in candidates:
            tokens = [t for t in re.split(r"[^a-z0-9]+", candidate.lower()) if t]
            if tokens and all(t in nc for t in tokens):
                return c
    return None


def load_nigerian_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    sheets = pd.read_excel(path, sheet_name=None)
    frames = []

    for sheet_name, raw in sheets.items():
        if raw.empty:
            continue

        df = raw.copy()
        cols = list(df.columns)

        mappings = {
            "programme": _find_column(
                cols, ["programme", "program", "course of study", "department"]
            ),
            "entry_year": _find_column(
                cols, ["entry year", "year of entry", "admission year"]
            ),
            "gpa_100": _find_column(
                cols, ["100 level gpa", "100l gpa", "first year gpa", "year 1 gpa", "gpa 100"]
            ),
            "gpa_200": _find_column(
                cols, ["200 level gpa", "200l gpa", "second year gpa", "year 2 gpa", "gpa 200"]
            ),
            "gpa_300": _find_column(
                cols, ["300 level gpa", "300l gpa", "third year gpa", "year 3 gpa", "gpa 300"]
            ),
            "final_cgpa": _find_column(
                cols, ["final cgpa", "overall cgpa", "graduation cgpa", "final cumulative gpa"]
            ),
        }

        if mappings["final_cgpa"] is None:
            for c in cols:
                if _clean_name(c) == "cgpa":
                    mappings["final_cgpa"] = c
                    break

        rename_map = {
            original: canonical
            for canonical, original in mappings.items()
            if original is not None
        }
        df = df.rename(columns=rename_map)

        if "programme" not in df.columns:
            df["programme"] = str(sheet_name)

        if "final_cgpa" in df.columns:
            frames.append(df)

    if not frames:
        raise ValueError("Could not identify final CGPA in the Nigerian workbook.")

    df = pd.concat(frames, ignore_index=True)

    numeric_cols = ["entry_year", "gpa_100", "gpa_200", "gpa_300", "final_cgpa"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    candidate_features = ["programme", "entry_year", "gpa_100", "gpa_200", "gpa_300"]
    features = [c for c in candidate_features if c in df.columns]

    # Remove completely empty fields.
    features = [c for c in features if df[c].notna().sum() > 0]

    if not any(c in features for c in ["gpa_100", "gpa_200", "gpa_300"]):
        raise ValueError("No usable early-year GPA predictors were detected.")

    df = df[features + ["final_cgpa"]].dropna(subset=["final_cgpa"]).copy()
    return df, features


# ------------------------------------------------------------
# ML HELPERS
# ------------------------------------------------------------
def build_preprocessor(df, features):
    categorical = [c for c in features if c == "programme" or df[c].dtype == "object"]
    numerical = [c for c in features if c not in categorical]

    transformers = []

    if numerical:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical,
        ))

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    return ColumnTransformer(transformers)


def choose_probability_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50

    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def classifier_metrics(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def warning_level(predicted_cgpa, probability, q25, q50, trajectory_delta):
    declining = pd.notna(trajectory_delta) and trajectory_delta <= -0.30

    # Data-driven early-warning bands.
    if predicted_cgpa <= q25 or probability >= 0.80:
        return "CRITICAL"
    if predicted_cgpa <= (q25 + q50) / 2 or probability >= 0.60:
        return "HIGH"
    if predicted_cgpa <= q50 or probability >= 0.35 or declining:
        return "MODERATE"
    return "LOW"


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, features = load_nigerian_data()

    q25 = float(df["final_cgpa"].quantile(0.25))
    q50 = float(df["final_cgpa"].quantile(0.50))
    q75 = float(df["final_cgpa"].quantile(0.75))

    # Core classification target:
    # bottom quartile = low academic performance.
    df["low_performance"] = (df["final_cgpa"] <= q25).astype(int)

    X = df[features]
    y_cls = df["low_performance"]
    y_reg = df["final_cgpa"]

    print("\n=== NIGERIAN CGPA DISTRIBUTION ===")
    print(f"Minimum CGPA: {df['final_cgpa'].min():.3f}")
    print(f"25th percentile: {q25:.3f}")
    print(f"Median CGPA: {q50:.3f}")
    print(f"75th percentile: {q75:.3f}")
    print(f"Maximum CGPA: {df['final_cgpa'].max():.3f}")

    print("\n=== CLASS BALANCE ===")
    print("Total records:", len(df))
    print("Low-performance cases:", int(y_cls.sum()))
    print("Other students:", int((y_cls == 0).sum()))
    print("Low-performance prevalence:", f"{y_cls.mean():.2%}")

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg,
        test_size=0.20,
        random_state=42,
        stratify=y_cls
    )

    cls_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    reg_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # ---------------- Classification ----------------
    classifiers = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=1
            ),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.10],
                "model__max_depth": [1, 2],
            },
        ),
    }

    cls_results = {}
    cls_models = {}

    for name, (model, params) in classifiers.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(df, features)),
            ("model", model),
        ])

        search = GridSearchCV(
            pipe,
            params,
            scoring="f1",
            cv=cls_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, yc_train)

        prob = search.best_estimator_.predict_proba(X_test)[:, 1]
        threshold = choose_probability_threshold(yc_test, prob)
        pred = (prob >= threshold).astype(int)

        cv_scores = cross_val_score(
            search.best_estimator_,
            X_train,
            yc_train,
            scoring="f1",
            cv=cls_cv,
            n_jobs=1,
        )

        cls_results[name] = {
            "best_params": search.best_params_,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "tuned_probability_threshold": threshold,
            "tuned_threshold_metrics": classifier_metrics(yc_test, pred, prob),
        }
        cls_models[name] = search.best_estimator_

        print(f"\n[CLASSIFIER] {name}")
        print("CV F1:", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("Tuned threshold:", f"{threshold:.3f}")
        print("Tuned metrics:", cls_results[name]["tuned_threshold_metrics"])

    best_cls_name = max(cls_results, key=lambda n: cls_results[n]["cv_f1_mean"])
    best_cls = cls_models[best_cls_name]
    best_prob_threshold = cls_results[best_cls_name]["tuned_probability_threshold"]

    # ---------------- Regression ----------------
    regressors = {
        "Random Forest Regressor": (
            RandomForestRegressor(random_state=42, n_jobs=1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 1.0],
            },
        ),
        "Gradient Boosting Regressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.10],
                "model__max_depth": [1, 2],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    reg_results = {}
    reg_models = {}

    for name, (model, params) in regressors.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(df, features)),
            ("model", model),
        ])

        search = GridSearchCV(
            pipe,
            params,
            scoring="neg_root_mean_squared_error",
            cv=reg_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, yr_train)

        pred = search.best_estimator_.predict(X_test)
        cv_rmse = -cross_val_score(
            search.best_estimator_,
            X_train,
            yr_train,
            scoring="neg_root_mean_squared_error",
            cv=reg_cv,
            n_jobs=1,
        )

        reg_results[name] = {
            "best_params": search.best_params_,
            "cv_rmse_mean": float(cv_rmse.mean()),
            "cv_rmse_std": float(cv_rmse.std()),
            "test_r2": float(r2_score(yr_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(yr_test, pred))),
            "test_mae": float(mean_absolute_error(yr_test, pred)),
        }
        reg_models[name] = search.best_estimator_

        print(f"\n[REGRESSOR] {name}")
        print("CV RMSE:", f"{cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print("Test R²:", f"{reg_results[name]['test_r2']:.4f}")
        print("Test RMSE:", f"{reg_results[name]['test_rmse']:.4f}")
        print("Test MAE:", f"{reg_results[name]['test_mae']:.4f}")

    best_reg_name = min(reg_results, key=lambda n: reg_results[n]["cv_rmse_mean"])
    best_reg = reg_models[best_reg_name]

    # ---------------- Deployment artifacts ----------------
    best_cls.fit(X, y_cls)
    best_reg.fit(X, y_reg)

    risk_prob = best_cls.predict_proba(X)[:, 1]
    predicted_cgpa = best_reg.predict(X)

    screening = df.copy()
    screening.insert(
        0,
        "student_id",
        [f"NG-STU-{i:04d}" for i in range(1, len(df) + 1)]
    )

    screening["predicted_final_cgpa"] = predicted_cgpa
    screening["low_performance_probability"] = risk_prob

    screening["trajectory_delta"] = np.nan
    if "gpa_100" in screening.columns and "gpa_300" in screening.columns:
        screening["trajectory_delta"] = screening["gpa_300"] - screening["gpa_100"]
    elif "gpa_100" in screening.columns and "gpa_200" in screening.columns:
        screening["trajectory_delta"] = screening["gpa_200"] - screening["gpa_100"]

    screening["risk_level"] = [
        warning_level(pcgpa, prob, q25, q50, delta)
        for pcgpa, prob, delta in zip(
            screening["predicted_final_cgpa"],
            screening["low_performance_probability"],
            screening["trajectory_delta"],
        )
    ]

    priority_map = {
        "LOW": "ROUTINE",
        "MODERATE": "MONITOR",
        "HIGH": "PRIORITY",
        "CRITICAL": "URGENT",
    }
    screening["intervention_priority"] = screening["risk_level"].map(priority_map)

    screening.to_csv(
        f"{OUTPUT_DIR}/student_risk_screening.csv",
        index=False
    )
    joblib.dump(best_cls, f"{OUTPUT_DIR}/risk_model.pkl")
    joblib.dump(best_reg, f"{OUTPUT_DIR}/cgpa_model.pkl")

    metadata = {
        "dataset_context": "Nigerian university academic records",
        "records": int(len(df)),
        "target_definition": (
            "Low academic performance is operationalised as final CGPA "
            "at or below the 25th percentile of this Nigerian institutional dataset."
        ),
        "low_performance_cgpa_threshold": q25,
        "median_cgpa": q50,
        "upper_quartile_cgpa": q75,
        "low_performance_cases": int(y_cls.sum()),
        "low_performance_prevalence": float(y_cls.mean()),
        "features": features,
        "selected_classifier": best_cls_name,
        "selected_classifier_probability_threshold": float(best_prob_threshold),
        "selected_regressor": best_reg_name,
        "classification_results": cls_results,
        "regression_results": reg_results,
        "warning_logic_note": (
            "Early-warning levels combine predicted final CGPA, probability of a "
            "bottom-quartile academic outcome, and declining GPA trajectory."
        ),
    }

    with open(
        f"{OUTPUT_DIR}/model_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n=== FINAL SELECTION ===")
    print("Selected classifier:", best_cls_name)
    print("Selected probability threshold:", f"{best_prob_threshold:.3f}")
    print("Selected regressor:", best_reg_name)
    print("Low-performance CGPA boundary:", f"{q25:.3f}")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()