# Diabetes Prediction Model

A machine learning classification model that predicts whether a person is **diabetic or non-diabetic** based on easily obtainable health indicators.

## Problem Statement
Diabetes affects millions worldwide, yet many cases go undetected due to limited healthcare access. This model serves as an **early warning system** using features that can be measured at home or through routine checkups.

## Dataset Features
- Age, BMI, HbA1c Level, Blood Glucose Level
- Hypertension, Heart Disease, Smoking History, Gender

## Approach
- Handled severe class imbalance (91.5% vs 8.5%) using **SMOTEENN**
- Applied **Box-Cox transformation** for skewed features
- Compared 6 algorithms: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, CatBoost
- Tuned hyperparameters using **GridSearchCV**

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CatBoost ✅ | ~97% | 97.6% | — | 79.6% |

**CatBoost** was selected as the final model for its superior F1-Score and robustness with imbalanced data.

## Tech Stack
`Python` `CatBoost` `Scikit-learn` `Pandas` `NumPy` `SMOTEENN` `Matplotlib` `Seaborn`
