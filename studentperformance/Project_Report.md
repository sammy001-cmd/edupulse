# Academic Performance Intelligence System

## Project Report

**Version:** 2.0  
**Date:** [Insert Date]

---

## 1. Executive Summary

This project develops an **Academic Performance Intelligence System** that uses machine learning to predict student academic outcomes, identify at-risk students early, and recommend targeted interventions. The system provides a web-based dashboard for lecturers, academic advisers, and university administrators to monitor student performance, prioritize support, and reduce failure and dropout rates.

The system is built using the UCI Student Performance dataset as a benchmark for model development. It trains multiple classification and regression models, selects the best-performing ones, and exposes predictions through an interactive Streamlit application. The final output includes risk scores, early warnings, identified risk factors, and prioritized intervention plans for each student.

---

## 2. Problem Statement

Tertiary institutions face significant challenges related to student underperformance and dropout. Manual monitoring is time-consuming, subjective, and often fails to detect at-risk students until it is too late. Without early identification and personalized support, many students fall through the cracks.

**Specific problems addressed:**
- Late identification of academically weak students.
- Generic teaching that does not adapt to individual needs.
- Inefficient use of educational data for decision-making.
- Lack of actionable recommendations for lecturers and advisers.

---

## 3. Objectives

### Main Objective
To design and implement a machine learning system that predicts student performance, provides early warnings, and recommends evidence-based interventions.

### Specific Objectives
1. Develop a robust data preprocessing pipeline.
2. Train and compare classification models to predict pass/fail.
3. Train regression models to predict exact final grades.
4. Generate risk scores and early warning categories.
5. Build a risk factor identification module.
6. Create an intervention recommendation engine.
7. Develop an interactive dashboard for institutional use.
8. Provide visual analytics and model performance metrics.

---

## 4. Target Audience

- **Primary:**  
  - Lecturers / Course advisers  
  - Academic advisers / Counsellors  
  - University administrators  

- **Secondary:**  
  - Students (via alerts)  
  - Parents / guardians  
  - Education researchers  

---

## 5. System Architecture

The system follows a modular pipeline:




### 5.1 Data Collection
- Uses the **UCI Student Performance** dataset (public benchmark).  
- Combines Mathematics and Portuguese student records (1044 total).

### 5.2 Preprocessing
- **Missing values:** median imputation for numerical, most frequent for categorical.  
- **Encoding:** one-hot encoding for categorical features.  
- **Scaling:** StandardScaler for numerical features.  
- **Targets:**  
  - `fail` (1 if final grade < 10, else 0) – classification  
  - `G3` (final grade 0–20) – regression  

### 5.3 Machine Learning Models
**Classification (Pass/Fail):**
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  

**Regression (Final Grade):**
- Random Forest Regressor  
- Gradient Boosting Regressor  

### 5.4 Evaluation
- **Classification:** Accuracy, F1-score (stratified train/test split).  
- **Regression:** R², RMSE.  
- Best models selected based on F1 (classification) and R² (regression).

### 5.5 Risk Scoring & Early Warning
- Failure probability from classifier = Risk Score.  
- Thresholds:  
  - `LOW` < 0.3  
  - `MODERATE` 0.3–0.5  
  - `HIGH` 0.5–0.7  
  - `CRITICAL` ≥ 0.7  

### 5.6 Risk Factor Identifier
Rule-based module that flags known risk indicators from raw features (e.g., previous failures, high absenteeism, low study time, low G1/G2, etc.).

### 5.7 Intervention Engine
Maps risk factors to specific, prioritized actions:
- Previous failures → remedial classes  
- High absenteeism → attendance warning to student/guardian  
- Low study time → structured timetable + peer mentoring  
- Low G1 → foundational revision  
- Low G2 → targeted tutoring  
- High social outings → time management advice  
- Alcohol consumption → wellness referral  
- Poor health → medical check-up  
- Poor family relationship → family engagement  

---

## 6. Dashboard Pages

The Streamlit dashboard consists of six pages:

1. **🏠 Academic Overview** – institutional metrics (total students, pass rate, risk distribution).  
2. **👨‍🎓 Student Assessment** – input a student’s data to get instant prediction, risk level, factors, and recommendations.  
3. **🚨 Early Warning Centre** – table of students flagged by risk, filterable, with detail view.  
4. **💡 Intervention Recommendations** – actionable plans for at-risk students, grouped by priority.  
5. **📊 Academic Analytics** – charts: grade distribution, pass/fail, study time, absences, feature importance.  
6. **🧠 Machine Learning Centre** – model comparison tables and performance metrics.

---

## 7. Results

### Classification (Pass/Fail)
| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | 89.95%   | 77.42%   |
| Random Forest       | 90.91%   | 79.12%   |
| Gradient Boosting   | 89.95%   | 78.35%   |

**Best:** Random Forest Classifier

### Regression (Final Grade)
| Model                     | R²     | RMSE   |
|---------------------------|--------|--------|
| Random Forest Regressor   | 0.8262 | 1.6393 |
| Gradient Boosting Regressor | 0.8314 | 1.6143 |

**Best:** Gradient Boosting Regressor

### Feature Importance
- Top predictors include previous grades (G1, G2), failures, study time, absences, and school support.

---

## 8. Conclusion

The Academic Performance Intelligence System successfully demonstrates the use of machine learning to predict student performance, flag at-risk students, and provide actionable interventions. The system achieves high accuracy and offers a user-friendly dashboard for institutional decision-making. It can be extended to real university data in the future.

---

## 9. Future Work & Nigeria Adaptation

- Collect and train on actual Nigerian university data (with ethical approval).  
- Adapt features: JAMB score, department, level, funding type, hostel status, etc.  
- Convert target to Nigerian GPA (5-point or percentage).  
- Deploy on university servers and integrate with student portals.  
- Add real-time alerts via email/SMS.  
- Implement hyperparameter tuning and advanced models (XGBoost, LightGBM).  

---

## 10. How to Run

Refer to `README.md` for installation and execution instructions.

---

## 11. References

- UCI Machine Learning Repository: Student Performance Dataset  
  https://archive.ics.uci.edu/ml/datasets/Student+Performance  
- scikit-learn documentation: https://scikit-learn.org/stable/  
- Streamlit documentation: https://docs.streamlit.io/

