""" Street Vendor Survival Prediction
    Phase 6 - Machine Learning
    Model Training, Evaluation and Selection """

# Importing Libraries
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay)

# Loading Feature Engineered Dataset
dataset_path = (
    r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/"
    r"Minor Project/Street-Vendor-Survival-Prediction/"
    r"datasets/processed_dataset/street_vendor_survival_feature_engineered.csv")
df = pd.read_csv(dataset_path)
print("Feature Engineered Dataset Loaded Successfully")
print(f"Dataset Shape : {df.shape}")

# Separating Features and Target
X = df.drop(columns=["vendor_survived"])
y = df["vendor_survived"]
print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# Checking Target Distribution
print("\nTarget Distribution: \n", y.value_counts())
print("\nTarget Percentage: \n", (y.value_counts(normalize=True) * 100).round(2))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y)
print("\nData Split Completed")
print(f"Training Data : {X_train.shape}")
print(f"Testing Data  : {X_test.shape}")

# Defining Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss")
}

# Training and Evaluating Models
results = []
trained_models = {}

for model_name, model in models.items():
    print("\n" + model_name)
    print("Training model...")
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    trained_models[model_name] = model
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

# Model Comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="ROC AUC",
    ascending=False
).reset_index(drop=True)
print("\nModel Comparison")
print(results_df.to_string(index=False))

# Saving Model Comparison Results
results_path = (
    r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/"
    r"Minor Project/Street-Vendor-Survival-Prediction/reports")

os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(
    results_path,
    "model_comparison.csv")

results_df.to_csv(
    results_file,
    index=False)
print(f"\nModel comparison saved at:")
print(results_file)

# Selecting the Best Model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print("\nBest Model Selected")
print(f"Model : {best_model_name}")

# Detailed Evaluation of Best Model
best_predictions = best_model.predict(X_test)
print("\nClassification Report")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Closed", "Active"],
        zero_division=0))

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    best_predictions
)
print("\nConfusion Matrix: \n", cm)

# Save Confusion Matrix
figures_path = (
    r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/"
    r"Minor Project/Street-Vendor-Survival-Prediction/reports/figures/ML")
os.makedirs(figures_path, exist_ok=True)
plt.figure(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Closed", "Active"]).plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

confusion_matrix_file = os.path.join(
    figures_path,
    "confusion_matrix.png")

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight")

plt.close()
print("\nConfusion matrix saved at:")
print(confusion_matrix_file)

# Saving the Best Trained Model
models_path = (
    r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/"
    r"Minor Project/Street-Vendor-Survival-Prediction/models")

os.makedirs(models_path, exist_ok=True)
model_file = os.path.join(models_path, "trained_model.pkl")
joblib.dump(best_model, model_file)

print("\nBest Model Saved Successfully")
print(f"Model : {best_model_name}")
print(f"Location : {model_file}")

# Final Summary
best_row = results_df.iloc[0]
print("\nFinal Model Performance")
print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_row['Accuracy']:.4f}")
print(f"Precision  : {best_row['Precision']:.4f}")
print(f"Recall     : {best_row['Recall']:.4f}")
print(f"F1 Score   : {best_row['F1 Score']:.4f}")
print(f"ROC AUC    : {best_row['ROC AUC']:.4f}")
print("\nPhase 6 Machine Learning Completed Successfully")