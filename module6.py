# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
# Replace 'Telco-Customer-Churn.csv' with your file path
data = pd.read_csv('Telco-Customer-Churn.csv')

# Preview the dataset
print("Initial Dataset:\n", data.head())

# Data Cleaning
# Handle missing data in 'TotalCharges'
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Convert 'Churn' to binary
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop irrelevant column
data.drop(['customerID'], axis=1, inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

print("Cleaned Dataset Info:\n", data.info())

# Define Features (X) and Target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model Training: Random Forest with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Best Model
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Model Evaluation
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Print Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC Curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_rf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importances")
plt.show()

# Analyze Misclassified Samples
X_test['Actual'] = y_test.values
X_test['Predicted'] = y_pred
misclassified = X_test[X_test['Actual'] != X_test['Predicted']]

print("Misclassified Samples:\n", misclassified.head(5))

# Save the Results to CSV (optional)
misclassified.to_csv('misclassified_samples.csv', index=False)

# Save Code and Dataset to GitHub
# Refer to the README in your GitHub repository to explain the analysis process.
