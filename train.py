from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.metrics import ( accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, 
                                matthews_corrcoef, confusion_matrix, classification_report
                            )

import matplotlib.pyplot as plt
import seaborn as sns
# 1. Project configuration

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "wdbc.data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Diagnosis"



# 2. Dataset feature names

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# 3. Load and validate dataset


def load_dataset():

    columns = ["ID", TARGET_COLUMN] + FEATURE_NAMES

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=columns,
        na_values=["", "NA", "N/A", "null", "?"]
    )

    # Remove identifier column
    df = df.drop(columns=["ID"])

    # Convert all features to numerical values
    for column in FEATURE_NAMES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert target labels
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "B": 0,
        "M": 1
    })

    # Target cannot be missing
    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Missing or invalid target values found."
        )

    # Check for infinite values
    if df[FEATURE_NAMES].isin(
        [float("inf"), float("-inf")]
    ).any().any():
        raise ValueError(
            "Infinite values found in the dataset."
        )

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN].astype(int)

    return X, y

# 4. Train-test split

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# 5. Define preprocessing pipeline

def create_preprocessor():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])


# 6. Define five classification models

def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    return models


# 7. Train all five models

def train_models(X_train, y_train):

    models = create_models()

    trained_models = {}

    for model_name, classifier in models.items():

        print(f"\nTraining {model_name}...")

        # Each classifier gets its own preprocessing pipeline
        pipeline = Pipeline([
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ])

        # Fit preprocessing and classifier using training data
        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        print(f"{model_name} trained successfully.")

    return trained_models


# 8. Save trained models

def save_models(trained_models):

    for model_name, pipeline in trained_models.items():

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            pipeline,
            model_path
        )

        print(f"Saved model: {model_path}")



# 9. Evaluate all trained models


def evaluate_models(trained_models, X_test, y_test):

    results = []

    for model_name, pipeline in trained_models.items():

        print(f"\nEvaluating {model_name}...")

        # Predict class labels
        y_pred = pipeline.predict(X_test)

        # Predict probabilities for AUC
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        metrics = {
            "Model": model_name,

            "Accuracy": accuracy_score(
                y_test, y_pred
            ),

            "AUC": roc_auc_score(
                y_test, y_prob
            ),

            "Precision": precision_score(
                y_test, y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test, y_pred,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, y_pred,
                zero_division=0
            ),

            "MCC": matthews_corrcoef(
                y_test, y_pred
            )
        }

        results.append(metrics)

        # Generate confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"]
        )

        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            BASE_DIR / "outputs" / filename
        )

        plt.close()

        # Print classification report
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Benign", "Malignant"],
                zero_division=0
            )
        )

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)

    # Save evaluation results
    results_df.to_csv(
        BASE_DIR / "outputs" / "model_comparison.csv",
        index=False
    )

    print("\nModel Comparison:")
    print(
        results_df.round(4).to_string(index=False)
    )

    return results_df


# 10. Main execution

if __name__ == "__main__":

    # 1. Load dataset
    X, y = load_dataset()

    print("Dataset loaded successfully.")
    print("Total instances:", len(X))
    print("Total features:", X.shape[1])

    print("\nMissing values before preprocessing:")
    print(X.isnull().sum().sum())

    print("\nTarget distribution:")
    print(y.value_counts())

    # 2. Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(
        X, y
    )

    print("\nTraining instances:", len(X_train))
    print("Testing instances:", len(X_test))

    # 3. Save the test dataset for Streamlit
    test_data = X_test.copy()

    test_data[TARGET_COLUMN] = y_test.map({
        0: "B",
        1: "M"
    })

    test_data.to_csv(
        BASE_DIR / "test_data.csv",
        index=False
    )

    print("\nTest dataset saved successfully.")
    print("File location:", BASE_DIR / "test_data.csv")

    # 4. Create output directory
    (BASE_DIR / "outputs").mkdir(
        parents=True,
        exist_ok=True
    )

    # 5. Train all five models
    trained_models = train_models(
        X_train,
        y_train
    )

    # 6. Save all trained models
    save_models(trained_models)

    # 7. Evaluate all five models
    results_df = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    # 8. Display final model comparison
    print("\nFinal Model Comparison:")
    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nAll models trained and evaluated successfully.")