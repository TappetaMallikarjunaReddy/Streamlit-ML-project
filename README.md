# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

This project implements and compares five machine learning classification models to predict whether a breast tumour is **benign** or **malignant** using diagnostic measurements.

The models are trained and evaluated on the same dataset and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). A Streamlit application allows users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Number of instances:** 569
- **Number of input features:** 30 numeric features
- **Target:** Diagnosis
  - `B` — Benign
  - `M` — Malignant

The dataset contains measurements computed from digitized images of breast fine needle aspirates. The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

For model training, the diagnosis labels are encoded as `0` for benign and `1` for malignant. Malignant is treated as the positive class when calculating classification metrics.

## 3. Repository and Application Links

- **GitHub Repository:** https://github.com/TappetaMallikarjunaReddy/Streamlit-ML-project
- **Live Streamlit App:** https://app-ml-project-aam7zjpcaa2qbzp89nzje7.streamlit.app/

## 4. Models and Evaluation

All five models use the same dataset and stratified 80:20 train-test split, with `random_state=42`. Missing values are handled using median imputation, and features are standardized where applicable. The preprocessing steps are saved with each trained model.

### Model comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine (SVM) | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

*Metrics are from the current training run; verify them against the generated metrics output before submission.*

### Observations

- **Logistic Regression:** Achieved strong overall performance and the highest AUC in this run, indicating excellent discrimination between the two classes.
- **Decision Tree:** Produced lower performance than the other models on this test split. Its MCC was also the lowest, indicating comparatively weaker overall classification agreement.
- **K-Nearest Neighbors:** Performed well on accuracy and AUC, but its recall was lower than Logistic Regression and SVM.
- **Gaussian Naive Bayes:** Achieved a high AUC, but its recall and F1 score were comparatively lower on this test split.
- **Support Vector Machine:** Achieved the highest accuracy, precision, F1 score, and MCC in this run. Its recall was also high, though slightly below Logistic Regression.

### Overall winner

**Support Vector Machine (SVM)** was the strongest overall model on this test split, with the highest accuracy, precision, F1 score, and MCC. Logistic Regression achieved the highest AUC. These results describe this particular split and should not be interpreted as a guarantee of performance on other data.

## 5. Project Structure

```text
ML_Assignment_2/
├── app.py
├── train.py
├── requirements.txt
├── test_data.csv
├── data/
│   └── wdbc.data
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
└── outputs/
    ├── model_comparison.csv
    └── confusion_matrices/
```

*Update the filenames above if your actual saved model or output filenames differ.*

## 6. How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/TappetaMallikarjunaReddy/Streamlit-ML-project.git
   cd ML_Assignment_2
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. Upload a test CSV containing all required input features and the target column. Select a trained model to view its evaluation metrics and confusion matrix or classification report.

The application loads the saved model and evaluates the uploaded test data. It does not retrain the model during evaluation.

## 7. Streamlit Application Features

- Upload test data in CSV format.
- Select one of the five trained classification models.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View a confusion matrix and/or classification report for the selected model.

## 8. Tools and Libraries

- Python
- pandas and NumPy
- scikit-learn
- joblib
- Matplotlib and Seaborn
- Streamlit

## 9. Conclusion

Five classification algorithms were implemented and evaluated on the Wisconsin Diagnostic Breast Cancer dataset. SVM provided the strongest overall result on the current test split, while Logistic Regression achieved the highest AUC. The Streamlit application makes it possible to evaluate the saved models interactively using uploaded test data.
