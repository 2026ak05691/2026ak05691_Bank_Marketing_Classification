
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from model_classes import (
    LogisticRegressionScratch,
    DecisionTreeScratch,
    KNNScratch,
    GaussianNaiveBayesScratch,
    SVMScratch
)

# -------------------------------------------------
# 1. Set project paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "bank_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------

data = pd.read_csv(DATA_PATH, sep=";")

print("Dataset shape:", data.shape)

X = data.drop("y", axis=1)

y = data["y"].map({
    "no": 0,
    "yes": 1
}).to_numpy()

# -------------------------------------------------
# 3. Split training and testing rows
# -------------------------------------------------

train_indices, test_indices = train_test_split(
    np.arange(len(data)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 4. Convert categorical features into numbers
# -------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

feature_columns = X_encoded.columns.tolist()

X_train = X_encoded.iloc[train_indices].to_numpy(dtype=float)
X_test = X_encoded.iloc[test_indices].to_numpy(dtype=float)

y_train = y[train_indices]
y_test = y[test_indices]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of encoded features:", X_train.shape[1])

# -------------------------------------------------
# 5. Standardize features
# -------------------------------------------------

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Avoid division by zero
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Save preprocessing information
preprocessing = {
    "feature_columns": feature_columns,
    "mean": mean,
    "std": std
}

joblib.dump(
    preprocessing,
    os.path.join(MODEL_DIR, "preprocessing.pkl")
)

# -------------------------------------------------
# 6. Initialize all five custom models
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegressionScratch(),
    "Decision Tree": DecisionTreeScratch(max_depth=5),
    "KNN": KNNScratch(k=5),
    "Gaussian Naive Bayes": GaussianNaiveBayesScratch(),
    "SVM": SVMScratch()
}

# -------------------------------------------------
# 7. Train and save all five models
# -------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    trained_models[name] = model

    file_name = model_files[name]

    save_path = os.path.join(MODEL_DIR, file_name)

    joblib.dump(model, save_path)

    print("Saved:", save_path)

# -------------------------------------------------
# 8. Save the held-out test dataset
# -------------------------------------------------

test_data = data.iloc[test_indices].copy()

test_data.to_csv(
    os.path.join(BASE_DIR, "test_data.csv"),
    index=False,
    sep=";"
)

# -------------------------------------------------
# 9. Training completed
# -------------------------------------------------

print("\n===================================")
print("ALL FIVE MODELS TRAINED SUCCESSFULLY")
print("===================================")

print("\nSaved files:")

for file_name in os.listdir(MODEL_DIR):
    print(file_name)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
