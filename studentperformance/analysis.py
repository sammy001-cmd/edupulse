import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Run without opening a window
import matplotlib.pyplot as plt

from student_performance_system import load_data

os.makedirs("outputs/analysis", exist_ok=True)

# -------------------------------
# 1. Load data
# -------------------------------
df = load_data()

# -------------------------------
# 2. Grade distribution
# -------------------------------
plt.figure(figsize=(8, 5))
df["G3"].hist(bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/analysis/grade_distribution.png")
plt.close()

# -------------------------------
# 3. Pass vs Fail counts
# -------------------------------
pass_fail = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail").value_counts()
plt.figure(figsize=(6, 6))
pass_fail.plot(kind="bar", color=["green", "red"])
plt.title("Pass vs Fail Counts")
plt.xlabel("Outcome")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/analysis/pass_fail_counts.png")
plt.close()

# -------------------------------
# 4. Average grade by study time
# -------------------------------
plt.figure(figsize=(8, 5))
df.groupby("studytime")["G3"].mean().plot(kind="bar", color="orange", edgecolor="black")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time (1 = low, 4 = high)")
plt.ylabel("Average Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/analysis/grade_by_studytime.png")
plt.close()

# -------------------------------
# 5. Average grade by absences
# -------------------------------
df["absences_group"] = pd.cut(
    df["absences"],
    bins=[-1, 2, 5, 10, 100],
    labels=["0-2", "3-5", "6-10", "10+"]
)
plt.figure(figsize=(8, 5))
df.groupby("absences_group", observed=False)["G3"].mean().plot(
    kind="bar", color="purple", edgecolor="black"
)
plt.title("Average Final Grade by Number of Absences")
plt.xlabel("Number of Absences")
plt.ylabel("Average Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/analysis/grade_by_absences.png")
plt.close()
df.drop("absences_group", axis=1, inplace=True)

# -------------------------------
# 6. Feature importance
# -------------------------------
print("[INFO] Loading trained models and preprocessor...")
clf = joblib.load("outputs/classification_model.pkl")
reg = joblib.load("outputs/regression_model.pkl")
preprocessor = joblib.load("outputs/preprocessor.pkl")

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Classification feature importance
clf_importance = clf.feature_importances_
clf_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": clf_importance
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(
    clf_importance_df["feature"][::-1],
    clf_importance_df["importance"][::-1],
    color="teal"
)
plt.title("Top 20 Features for Pass/Fail Prediction (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/analysis/classification_feature_importance.png")
plt.close()

# Regression feature importance
reg_importance = reg.feature_importances_
reg_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": reg_importance
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(
    reg_importance_df["feature"][::-1],
    reg_importance_df["importance"][::-1],
    color="crimson"
)
plt.title("Top 20 Features for Final Grade Prediction (Gradient Boosting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/analysis/regression_feature_importance.png")
plt.close()

print("[SUCCESS] Analysis complete. Charts saved to outputs/analysis/")