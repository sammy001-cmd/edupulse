# Defense Preparation Guide

## Common Questions & Answers

### Q1. Why did you choose this project?
A: Student underperformance and dropout are serious issues. I wanted to build a system that uses data to help institutions intervene early.

### Q2. What is the main contribution?
A: A complete machine learning pipeline that not only predicts performance but also explains risk factors and recommends interventions.

### Q3. How did you handle the dataset?
A: I used a public benchmark dataset for development. I combined two subjects to increase diversity. I handled missing values, encoded categorical features, and scaled numerical features.

### Q4. Why Random Forest for classification?
A: It performed best on F1-score. It handles non-linear relationships and provides feature importance.

### Q5. What does the risk score mean?
A: It is the probability of a student failing (final grade < 10). We map it to four risk levels.

### Q6. How do you know the model is reliable?
A: I used a stratified train/test split and evaluated on unseen test data. The Random Forest achieved 90.91% accuracy and F1 of 79.12%. Regression R² is 0.83, meaning 83% of variance in final grades is explained.

### Q7. What are the limitations?
A: The dataset is from Portugal, not Nigeria. The recommendation engine is rule-based, not learned. The model may not generalize perfectly to local contexts.

### Q8. How would you deploy this in a Nigerian university?
A: Collect local data, adapt features (JAMB, GPA, etc.), retrain models, and host on university servers. The dashboard can be integrated with student information systems.

### Q9. What improvements can you make?
A: Use hyperparameter tuning, XGBoost, SHAP for individual explanations, real-time alerts, and more features.

### Q10. Why Streamlit?
A: It allows rapid development of a professional interactive dashboard without heavy front-end code, and it is sufficient for a prototype.

## Defense Tips
- Show the dashboard live if possible.
- Walk through the code structure.
- Be honest about limitations and future work.
- Emphasize the system architecture and real-world impact.