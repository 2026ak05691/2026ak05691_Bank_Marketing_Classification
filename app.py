
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
     matthews_corrcoef
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification")
st.write(
    "Upload your test dataset, select a machine learning model, "
    "and evaluate its performance."
)

# --------------------------------------------------
# 2. LOAD SAVED MODELS AND PREPROCESSING
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "logistic_regression.pkl")
        ),

        "Decision Tree": joblib.load(
            os.path.join(MODEL_DIR, "decision_tree.pkl")
        ),

        "KNN": joblib.load(
            os.path.join(MODEL_DIR, "knn.pkl")
        ),

        "Gaussian Naive Bayes": joblib.load(
            os.path.join(MODEL_DIR, "naive_bayes.pkl")
        ),

        "SVM": joblib.load(
            os.path.join(MODEL_DIR, "svm.pkl")
        )
    }

    preprocessing = joblib.load(
        os.path.join(MODEL_DIR, "preprocessing.pkl")
    )

    return models, preprocessing


try:
    models, preprocessing = load_models()
    st.success("All five trained models loaded successfully.")

except Exception as e:
    st.error(f"Error loading saved models: {e}")
    st.stop()


# --------------------------------------------------
# 3. UPLOAD TEST DATASET
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(
            uploaded_file,
            sep=";"
        )

        st.success("Test dataset uploaded successfully.")

        st.write("Dataset shape:", test_data.shape)
        st.write("Preview of uploaded data:")
        st.dataframe(test_data.head())

        # Check target column
        if "y" not in test_data.columns:
            st.error("The uploaded CSV must contain target column 'y'.")
            st.stop()

        # Check target values
        if test_data["y"].isnull().any():
            st.error("Target column contains missing values.")
            st.stop()

        if not test_data["y"].isin(["yes", "no"]).all():
            st.error("Target column must contain only 'yes' and 'no'.")
            st.stop()

        # --------------------------------------------------
        # 4. PREPROCESS TEST DATA
        # --------------------------------------------------

        st.header("2. Data Preprocessing")

        X_test = test_data.drop("y", axis=1)
        y_test = test_data["y"].map({"no": 0, "yes": 1}).to_numpy()

        # Apply one-hot encoding
        X_test = pd.get_dummies(
            X_test,
            drop_first=True,
            dtype=int
        )

        # Use exactly the same feature columns as training
        feature_columns = preprocessing["feature_columns"]

        X_test = X_test.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Convert to NumPy array
        X_test = X_test.to_numpy(dtype=float)

        # Apply saved training mean and standard deviation
        mean = preprocessing["mean"]
        std = preprocessing["std"]

        # Prevent division by zero
        safe_std = np.where(std == 0, 1, std)

        X_test_scaled = (X_test - mean) / safe_std

        st.write(
            "Number of test samples:",
            X_test_scaled.shape[0]
        )

        st.write(
            "Number of processed features:",
            X_test_scaled.shape[1]
        )

        # --------------------------------------------------
        # 5. MODEL SELECTION
        # --------------------------------------------------

        st.header("3. Select Machine Learning Model")

        selected_model_name = st.selectbox(
            "Choose a model",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        # --------------------------------------------------
        # 6. PREDICTION
        # --------------------------------------------------

        y_pred = selected_model.predict(X_test_scaled)

        y_prob = selected_model.predict_proba(X_test_scaled)

        y_pred = np.asarray(y_pred).reshape(-1).astype(int)
        y_prob = np.asarray(y_prob).reshape(-1)

        if len(y_pred) != len(y_test) or len(y_prob) != len(y_test):
            st.error("Prediction output size does not match test data.")
            st.stop()

        if not np.isfinite(y_prob).all():
            st.error("Model generated invalid probability values.")
            st.stop()

        if ((y_prob < 0) | (y_prob > 1)).any():
            st.error("Model probabilities must be between 0 and 1.")
            st.stop()

        # --------------------------------------------------
        # 7. EVALUATION METRICS
        # --------------------------------------------------

        st.header("4. Model Evaluation")

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("ROC-AUC", f"{auc:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 8. CONFUSION MATRIX
        # --------------------------------------------------

        st.header("5. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # 9. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.header("6. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())

        # --------------------------------------------------
        # 10. MODEL COMPARISON
        # --------------------------------------------------

        st.header("7. Comparison of All Five Models")

        comparison_results = []

        for model_name, model in models.items():

            predictions = np.asarray(
                model.predict(X_test_scaled)
            ).reshape(-1).astype(int)

            probabilities = np.asarray(
                model.predict_proba(X_test_scaled)
            ).reshape(-1)

            comparison_results.append({

                "Model": model_name,

                "Accuracy": accuracy_score(
                    y_test, predictions
                ),

                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

                "ROC-AUC": roc_auc_score(
                    y_test,
                    probabilities
                )
                "MCC": matthews_corrcoef(
                    y_test,
                    predictions
                )
            })

        comparison_df = pd.DataFrame(comparison_results)

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.success("Evaluation completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:

    st.info("Please upload your test_data.csv file to begin.")
