## Fraud Detection System using Machine Learning
## Overview

This project focuses on building an end-to-end fraud detection system using machine learning techniques on a large-scale financial transaction dataset. The dataset contains over 6 million transactions with highly imbalanced class distribution, where fraudulent cases represent a very small fraction of the data.

The objective of this project is to accurately identify fraudulent transactions while minimizing false negatives, as missing fraud cases can lead to significant financial losses.

# Problem Statement

Financial fraud detection is a critical challenge due to:

Severe class imbalance
Evolving fraud patterns
High cost of misclassification

This project addresses these challenges by combining domain-driven feature engineering with advanced machine learning techniques.

# Dataset
Total records: ~6.3 million
Features include transaction type, amount, and account balances before and after transactions
Target variable: isFraud (0 = non-fraud, 1 = fraud)
Approach
1. Data Preprocessing
Reduced memory usage by optimizing data types
Selected relevant features for modeling
Handled categorical variables using one-hot encoding
2. Feature Engineering

Created domain-specific features to capture transaction behavior:

Balance differences between sender and receiver
Transaction inconsistency indicators
High-value transaction flags
Log transformation of transaction amount
Transaction-to-balance ratio

These features helped the model capture hidden fraud patterns more effectively.

3. Handling Class Imbalance

Due to extreme imbalance:

Applied SMOTE (Synthetic Minority Oversampling Technique)
Used class weighting in XGBoost (scale_pos_weight)
4. Model Building

Trained an XGBoost classifier with:

Controlled tree depth
Subsampling for generalization
Feature sampling for robustness
5. Evaluation Metrics

# Focused on:

Recall (primary metric to detect fraud cases)
Precision (to control false positives)

# Achieved:

Recall: ~99.8%
Precision: ~93%
6. Threshold Optimization

Instead of relying on the default classification threshold:

Evaluated multiple thresholds
Analyzed precision-recall tradeoff
Selected threshold based on business impact
7. Model Explainability

Used SHAP (SHapley Additive exPlanations) to:

Interpret model predictions
Identify key drivers of fraud
Validate feature importance
Results

The model successfully captures fraudulent behavior patterns with high recall, ensuring that most fraud cases are detected. Feature engineering played a critical role in improving performance.

# Project Structure
fraud-detection/
│
├── data/
├── notebook.ipynb
├── fraud_model.pkl
├── app.py
├── requirements.txt
└── README.md

# Key Learnings
Importance of domain knowledge in feature engineering
Handling extreme class imbalance in real-world datasets
Trade-offs between precision and recall in business scenarios
Model interpretability using SHAP
Future Improvements
Real-time fraud detection pipeline
Model monitoring and drift detection
Integration with APIs for production deployment
