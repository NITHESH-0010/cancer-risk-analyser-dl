# Healthcare Cancer Risk Platform (DL Version)

## Project Goal
This project aims to transition an existing machine learning pipeline into a deep learning approach for predicting cancer risk. The initial step sets up a robust baseline by cleaning the data properly and evaluating standard machine learning models before implementing neural networks.

## Dataset
The data used in this project is the Kaggle Cancer Prediction Dataset.
*Note: I do not claim ownership of this dataset.*

## What I fixed from v1
In the previous version of the pipeline, there was a significant data leakage/preprocessing bug related to the `CancerHistory` column. 
The old notebook applied IQR (Interquartile Range) outlier capping to all numeric columns. Because `CancerHistory` is a binary column where both Q1 and Q3 are 0, every `1` (indicating a history of cancer) was capped to `0`. Consequently, the models were trained without the `CancerHistory` feature since it became a constant column. 
In this updated version, binary columns are explicitly excluded from scaling and outlier capping.

## Pipeline So Far
1. **Data Loading & Verification:** Loaded data and verified missing values, duplicates, and column variations.
2. **Preprocessing:** Split the data into 70/15/15 train/validation/test stratified sets. Standardized only the non-binary numeric features based on the training split.
3. **Baselines:** Trained basic ML models (Logistic Regression, Random Forest, Gradient Boosting) on the new splits to set a performance benchmark for the upcoming Deep Learning models.

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8311 | 0.8000 | 0.7229 | 0.7595 | 0.9057 |
| Random Forest | 0.9289 | 0.9589 | 0.8434 | 0.8974 | 0.9474 |
| Gradient Boosting | 0.9422 | 0.9487 | 0.8916 | 0.9193 | 0.9570 |

## Original Repository
Link to the original Machine Learning repository:
[Cancer-Prediction-ML](https://github.com/NITHESH-0010/Cancer-Prediction-ML)
