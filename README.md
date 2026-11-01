# 2026ak05691_Bank_Marketing_Classification

Bank Marketing Binary Classification using five custom machine learning algorithms implemented from scratch.

## 1. Problem Statement

The objective of this project is to develop machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer demographic, financial, and campaign-related information.

This is a binary classification problem where the target variable is `y`:
- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

## 2. Dataset Description

- **Dataset:** Bank Marketing Dataset
- **Number of instances:** 4,521
- **Number of original features:** 16
- **Target variable:** `y`
- **Number of classes:** 2 (`yes` and `no`)
- **Missing values:** No missing values
- **Training and testing split:** 80:20
- **Training samples:** 3,616
- **Testing samples:** 905
- **Processed features:** 42 after one-hot encoding

The dataset contains customer information such as age, job, marital status, education, balance, housing loan, personal loan, contact type, campaign details, and previous marketing outcomes.

## 3. GitHub Repository

[2026ak05691_Bank_Marketing_Classification](https://github.com/2026ak05691/2026ak05691_Bank_Marketing_Classification)

## 4. Machine Learning Models Used

The following five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Model Performance Comparison

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 89.50% | 88.49% | 58.82% | 28.85% | 38.71% | 0.3627 |
| Decision Tree | 88.73% | 81.98% | 51.52% | 32.69% | 40.00% | 0.3520 |
| KNN | 88.40% | 74.15% | 48.78% | 19.23% | 27.59% | 0.2547 |
| Gaussian Naive Bayes | 82.98% | 78.98% | 32.14% | 43.27% | 36.89% | 0.2770 |
| SVM | 89.06% | 88.78% | 57.14% | 19.23% | 28.78% | 0.2871 |

## 6. Model-wise Observations

1. **Logistic Regression:** Achieved the highest accuracy (89.50%), precision (58.82%), and MCC (0.3627). It provides the strongest MCC performance among the five models, although its recall is relatively low.

2. **Decision Tree:** Achieved the highest F1 Score (40.00%), providing the best balance between precision and recall according to this metric.

3. **KNN:** Achieved 88.40% accuracy but relatively low recall (19.23%) and the lowest ROC-AUC (74.15%).

4. **Gaussian Naive Bayes:** Achieved the highest recall (43.27%), identifying more actual positive customers. However, it has lower accuracy and precision than the other models.

5. **SVM:** Achieved the highest ROC-AUC (88.78%), indicating good ability to distinguish between positive and negative classes. However, its recall is relatively low (19.23%).

## 7. Overall Model Recommendation

The Decision Tree model is recommended when F1 Score is the primary evaluation metric because it achieved the highest F1 Score of 40.00%.

However, Logistic Regression achieved the highest accuracy (89.50%), precision (58.82%), and MCC (0.3627). Gaussian Naive Bayes is preferable when identifying more actual positive customers is the main business priority because it achieved the highest recall (43.27%).

Since the dataset is imbalanced, accuracy alone should not be used to determine the best model.
