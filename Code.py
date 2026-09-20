# ==============================================================================
# PROJECT: Early Risk Prediction for Chronic Disease Using Data Mining
# DATASET: Pima Indians Diabetes Dataset
# DESCRIPTION: End-to-end Python script covering data preprocessing, feature scaling,
#              model training (Logistic Regression, Random Forest, SVM), evaluation,
#              and feature importance visualization.
# ==============================================================================

# --- STEP 1: IMPORT REQUIRED LIBRARIES ---
# Import data manipulation and numerical computation libraries
import pandas as pd
import numpy as np

# Import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning preprocessing modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import machine learning classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import model evaluation and performance metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

print("All necessary libraries imported successfully.")


# --- STEP 2: LOAD THE DATASET ---
# Load the raw dataset from a CSV file into a Pandas DataFrame
# Ensure 'diabetes.csv' is in your working directory or Colab file storage
dataset_filename = 'diabetes.csv'
df = pd.read_csv(dataset_filename)

# Display basic structural information about the dataset
print(f"Dataset loaded successfully with shape: {df.shape}")
print("First 5 rows of the dataset:")
display(df.head())


# --- STEP 3: DATA CLEANING & HANDLING MISSING VALUES ---
# In the Pima Indians Diabetes dataset, clinical measurements such as Glucose, 
# BloodPressure, SkinThickness, Insulin, and BMI cannot physically have a value of 0.
# These zeros represent unrecorded or missing data entries.

# Define the columns that contain biologically impossible zero values
columns_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Step 3.1: Replace 0 values with NaN (Null) so they can be properly imputed
df[columns_with_invalid_zeros] = df[columns_with_invalid_zeros].replace(0, np.nan)

# Step 3.2: Initialize a SimpleImputer using the 'median' strategy
# Median is preferred over mean to prevent distortion from skewed clinical distributions
median_imputer = SimpleImputer(strategy='median')

# Step 3.3: Apply the imputer to fill missing (NaN) values with column medians
df[columns_with_invalid_zeros] = median_imputer.fit_transform(df[columns_with_invalid_zeros])

# Verify that all missing values have been successfully handled
print("\nMissing values remaining per column after median imputation:")
print(df.isnull().sum())


# --- STEP 4: FEATURE SELECTION & TRAIN/TEST SPLIT ---
# Separate the independent features (X) from the target outcome variable (y)
# 'Outcome' indicates whether the patient is diabetic (1) or non-diabetic (0)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into training sets (80%) and testing sets (20%)
# stratify=y ensures that the proportion of diabetic/non-diabetic classes is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTraining set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")


# --- STEP 5: FEATURE SCALING ---
# Features have varying numeric ranges (e.g., Age ranges from 21-80, while Insulin ranges up to 800+).
# Distance-based models (like SVM) and gradient-based models require feature scaling.
# StandardScaler standardizes features by removing the mean and scaling to unit variance (Z-score).

feature_scaler = StandardScaler()

# Fit the scaler ONLY on the training data to prevent data leakage, then transform both train and test sets
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

print("\nFeature scaling (StandardScaler) applied successfully.")


# --- STEP 6: MODEL TRAINING & EVALUATION ---
# Define a dictionary of three distinct machine learning classifiers to compare:
# 1. Logistic Regression: Serves as an interpretable linear baseline model.
# 2. Random Forest Classifier: Robust ensemble model capturing non-linear interactions.
# 3. Support Vector Machine (SVC): Effective for finding optimal hyperplanes in complex spaces.

classification_models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42)
}

# Iterate through each model, train it, make predictions, and calculate performance metrics
model_performance_results = []

for model_name, classifier in classification_models.items():
    print(f"\n{'='*10} Training and Evaluating: {model_name} {'='*10}")
    
    # Step 6.1: Train (fit) the classifier using the scaled training data
    classifier.fit(X_train_scaled, y_train)
    
    # Step 6.2: Predict class labels and probabilities on the test set
    y_predictions = classifier.predict(X_test_scaled)
    y_probabilities = classifier.predict_proba(X_test_scaled)[:, 1] # Probability of positive class (Outcome = 1)
    
    # Step 6.3: Compute evaluation metrics
    model_accuracy = accuracy_score(y_test, y_predictions)
    model_precision = precision_score(y_test, y_predictions)
    model_recall = recall_score(y_test, y_predictions)
    model_f1 = f1_score(y_test, y_predictions)
    model_roc_auc = roc_auc_score(y_test, y_probabilities)
    
    # Store results for summary analysis
    model_performance_results.append({
        "Model Name": model_name,
        "Accuracy": model_accuracy,
        "Precision": model_precision,
        "Recall": model_recall,
        "F1-Score": model_f1,
        "ROC-AUC": model_roc_auc
    })
    
    # Print detailed classification report metrics
    print(f"Accuracy:  {model_accuracy:.4f}")
    print(f"ROC-AUC:   {model_roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_predictions))

# Convert performance results list into a clean DataFrame for side-by-side comparison
performance_summary_df = pd.DataFrame(model_performance_results)
print("\n" + "="*40)
print("FINAL MODEL PERFORMANCE SUMMARY TABLE")
print("="*40)
display(performance_summary_df)


# --- STEP 7: FEATURE IMPORTANCE VISUALIZATION (RANDOM FOREST) ---
# Extract feature importance scores from the trained Random Forest model
# This allows us to identify which clinical attributes (e.g., Glucose, BMI) are most critical for prediction.

trained_random_forest = classification_models["Random Forest"]
feature_importance_scores = trained_random_forest.feature_importances_
feature_column_names = X.columns

# Create a horizontal bar plot to visualize feature importance rankings
plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance_scores, 
    y=feature_column_names, 
    palette="viridis"
)

# Customize plot titles and axis labels
plt.title("Feature Importance Ranking derived from Random Forest Model", fontsize=14, fontweight='bold')
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Clinical Features", fontsize=12)
plt.tight_layout()

# Display the feature importance chart
plt.show()
print("\nProject pipeline execution completed successfully!")