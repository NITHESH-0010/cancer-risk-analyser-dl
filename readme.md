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


## Deep Learning Approach
A Multi-Layer Perceptron (MLP) was implemented using Keras. The architecture uses BatchNormalization, Dropout, and L2 regularization to prevent overfitting on this small dataset.
A hyperparameter search was conducted on the validation set, and the best configuration was evaluated using multiple random seeds for robustness.

### Deep Learning Results
The test results and 5-fold cross-validation scores indicate the model's performance compared to classical ML models.
*(See the CSV files in `results/` for full tabular metrics and standard deviations)*

### Conclusion
Tabular datasets of this size (~1500 rows) typically favor tree-based models like Gradient Boosting. The Deep Learning model performs competitively, but depending on the specific hyperparameters, it might only match or slightly underperform the tree ensembles. This confirms the standard wisdom that Neural Networks require larger datasets to significantly outperform boosted trees on tabular data.

### Limitations
- **Small Dataset:** Neural networks usually require more data to generalize effectively.
- **Synthetic-looking Data:** Some feature relationships might not perfectly mimic real-world clinical distributions.
- **Not a Clinical Tool:** This model is for educational and demonstrative purposes only and should not be used for actual medical diagnosis.


## Why the first MLP lagged
A deep dive into the features revealed that the diagnosis rates change in sharp steps rather than smoothly (e.g., around Age 50, BMI 25). These sharp steps suggest synthetic, rule-based data generation.

Because standard Neural Networks struggle to learn sharp step functions from raw numeric features, we introduced a `QuantilePiecewiseLinearEncoder`. This explicitly models these step-like nonlinearities by mapping the numeric features into bins.

Here are the reconciled results for the new Encoded MLP on both the Test split and 5-Fold Cross-Validation, presented side by side with classical models:

### Test Split Results
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| MLP raw features | 0.8569 +/- 0.0119 | 0.8097 +/- 0.0098 | 0.8000 +/- 0.0326 | 0.8046 +/- 0.0194 | 0.9257 +/- 0.0036 |
| MLP encoded features | 0.9129 +/- 0.0132 | 0.9015 +/- 0.0233 | 0.8578 +/- 0.0132 | 0.8791 +/- 0.0177 | 0.9498 +/- 0.0050 |
| Logistic Regression | 0.8311 | 0.8000 | 0.7229 | 0.7595 | 0.9057 |
| Random Forest | 0.9289 | 0.9589 | 0.8434 | 0.8974 | 0.9474 |
| Gradient Boosting | 0.9422 | 0.9487 | 0.8916 | 0.9193 | 0.9570 |

### 5-Fold Cross-Validation Results
| Model | Accuracy (CV) | ROC-AUC (CV) |
| --- | --- | --- |
| Logistic Regression | 0.8473 +/- 0.0183 | 0.9179 +/- 0.0134 |
| Random Forest | 0.9227 +/- 0.0071 | 0.9508 +/- 0.0113 |
| Gradient Boosting | 0.9307 +/- 0.0083 | 0.9531 +/- 0.0119 |
| MLP (DL) | 0.8820 +/- 0.0105 | 0.9391 +/- 0.0133 |
| MLP encoded features | 0.9120 +/- 0.0117 | 0.9510 +/- 0.0095 |

### Conclusion (Updated)
The engineered features improved the MLP's performance over raw features. However, Gradient Boosting and Random Forest remain very competitive (and often superior) on this tabular dataset, reinforcing that tree ensembles are a highly robust default for such data.
