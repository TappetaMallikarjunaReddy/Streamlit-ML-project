# app.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
                         classification_report, ConfusionMatrixDisplay
                         )

import matplotlib.pyplot as plt

from train import FEATURE_NAMES, TARGET_COLUMN



# Paths and model names


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}



# Load saved model

@st.cache_resource
def load_model(model_path):
    """Load a saved, trained sklearn pipeline."""
    return joblib.load(model_path)


# Read and validate uploaded data

def prepare_uploaded_data(uploaded_file):
    """Validate the CSV and return feature data and encoded target."""

    data = pd.read_csv(uploaded_file)

    required_columns = FEATURE_NAMES + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )

    if data.empty:
        raise ValueError("The uploaded CSV has no rows.")

    # Keep only the required features, in the same order used during training.
    X = data[FEATURE_NAMES].copy()

    # Convert feature columns to numeric values.
    # Blank cells are allowed because the saved pipeline has an imputer.
    for column in FEATURE_NAMES:
        original = X[column]
        converted = pd.to_numeric(original, errors="coerce")

        invalid_values = original.notna() & converted.isna()
        if invalid_values.any():
            examples = original[invalid_values].astype(str).head(3).tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values, "
                f"for example: {examples}"
            )

        X[column] = converted

    # Reject infinity values; the trained pipeline can impute missing values,
    # but infinity is not a valid measurement.
    if X.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("The feature data contains infinity values.")

    # Encode the target: benign = 0, malignant = 1.
    raw_target = data[TARGET_COLUMN]

    if raw_target.isna().any():
        raise ValueError(
            f"'{TARGET_COLUMN}' contains missing target values. "
            "A target is required for evaluation."
        )

    target_text = raw_target.astype(str).str.strip().str.upper()
    target_mapping = {
        "B": 0,
        "M": 1,
        "0": 0,
        "1": 1,
    }

    unknown_labels = sorted(set(target_text) - set(target_mapping))
    if unknown_labels:
        raise ValueError(
            "Unsupported target labels: "
            + ", ".join(unknown_labels)
            + ". Use B/M or 0/1."
        )

    y = target_text.map(target_mapping).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The uploaded data must contain both classes (B and M) "
            "to calculate all evaluation metrics."
        )

    return X, y


# Streamlit interface

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide",
)

st.title("Breast Cancer Classification Model Evaluation")
st.write(
    "Upload a labelled CSV file, select one of the trained models, "
    "and evaluate its predictions. The app uses saved models and "
    "does not retrain them."
)

st.info(
    "The target column must be named `Diagnosis` and contain B/M "
    "or 0/1 labels. The CSV must include all 30 trained feature columns."
)

model_name = st.selectbox(
    "Select a trained model",
    list(MODEL_FILES.keys()),
)

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        X_test, y_test = prepare_uploaded_data(uploaded_file)

        model_path = MODEL_DIR / MODEL_FILES[model_name]

        if not model_path.exists():
            st.error(
                f"Saved model not found: {model_path}. "
                "Run train.py first to create the model files."
            )
            st.stop()

        model = load_model(str(model_path))

        # Predict with the selected saved pipeline.
        y_pred = model.predict(X_test)

        # These models were trained with class 1 = malignant.
        y_score = model.predict_proba(X_test)[:, 1]

        # Calculate requested metrics.
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_score),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }

        st.subheader(f"Results: {model_name}")
        st.caption(
            f"Evaluated {len(X_test)} rows. "
            "Positive class: malignant (M / 1)."
        )

        metric_columns = st.columns(3)
        for index, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_columns[index % 3]:
                st.metric(metric_name, f"{metric_value:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        figure, axis = plt.subplots()
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign (B)", "Malignant (M)"],
        )
        display.plot(ax=axis, colorbar=False)
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        # Classification report
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign (B)", "Malignant (M)"],
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), width="stretch")

    except Exception as error:
        st.error(f"Could not evaluate the uploaded CSV: {error}")
        st.exception(error)  # temporary: shows the app-level traceback