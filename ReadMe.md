# Early Risk Prediction for Chronic Disease Using Data Mining

## 📋 Project Overview
This project applies supervised data mining and machine learning techniques to predict the likelihood of diabetes onset based on clinical and diagnostic measurements. Using the benchmark **Pima Indians Diabetes Dataset**, the project builds, optimizes, and compares multiple classification models to automate early-stage health risk assessment.

---

## 📂 Dataset Description
* **Dataset Name:** Pima Indians Diabetes Dataset
* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (Hosted on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))
* **Size:** 768 patient records with 9 attributes.
* **Features:**
  * `Pregnancies`: Number of times pregnant
  * `Glucose`: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
  * `BloodPressure`: Resting blood pressure (mm Hg)
  * `SkinThickness`: Triceps skin fold thickness (mm)
  * `Insulin`: 2-Hour serum insulin (mu U/ml)
  * `BMI`: Body mass index ($\text{weight in kg}/(\text{height in m})^2$)
  * `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic scoring)
  * `Age`: Age (years)
  * `Outcome`: Target variable ($0$ = Non-Diabetic, $1$ = Diabetic)

---

## 🛠️ Project Pipeline & Methodology
1. **Data Cleaning & Missing Value Imputation:**
   * Identified and replaced biologically impossible zero-values in clinical features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with `NaN`.
   * Imputed missing values using column medians to avoid skewness distortion.
2. **Feature Scaling:**
   * Applied `StandardScaler` (Z-score normalization) to center feature distributions and prevent scale bias.
3. **Model Training & Comparison:**
   * **Logistic Regression:** Serves as a baseline linear model.
   * **Random Forest Classifier:** An ensemble method handling non-linear interactions.
   * **Support Vector Machine (SVM):** Finds optimal hyperplanes using an RBF kernel.
4. **Evaluation Metrics:**
   * Evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## 🗂️ Repository Directory Structure
```text
diabetes-prediction-project/
│
├── data/
│   └── diabetes.csv                # Raw dataset file
│
├── notebooks/
│   └── diabetes_analysis.ipynb     # Jupyter Notebook containing full pipeline code
│
├── src/
│   ├── preprocess.py               # Data cleaning and imputation scripts
│   └── train.py                    # Model training and evaluation scripts
│
├── README.md                       # Comprehensive project documentation
└── requirements.txt                # List of Python dependencies