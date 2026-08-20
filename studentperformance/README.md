# Student Academic Performance Prediction, Analysis, Intervention Recommendation and Early Warning System Using Machine Learning

## Project Overview
This project builds a machine learning system that predicts student academic performance, identifies at-risk students early, and recommends personalized interventions. It uses the UCI Student Performance dataset and trains both classification (pass/fail) and regression (final grade) models.

## Features
- **Automated data collection**: Downloads public dataset from UCI repository.
- **Data preprocessing**: Handles missing values, encodes categorical features, scales numerical features.
- **Machine learning models**: Trains multiple models and selects the best based on evaluation metrics.
- **Risk scoring**: Computes the probability of failure for each student.
- **Early warning system**: Flags students as OK, WARNING, or CRITICAL based on risk threshold.
- **Intervention recommendation engine**: Suggests specific actions based on risk factors (e.g., attendance, past failures, study time).
- **Interactive dashboard**: Web interface for single student prediction, bulk prediction via CSV upload, and model analysis charts.
- **Model explainability**: Feature importance charts show the factors that influence predictions.

## Project Structure