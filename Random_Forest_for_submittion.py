import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop unnecessary columns in train and test datasets
X = train_df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = train_df['Diagnosis']
X_test = test_df.drop(columns=['PatientID', 'DoctorInCharge'])

# Feature selection with SelectKBest
best_k = 15  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=best_k)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

# Scaling features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define Random Forest model and hyperparameter grid
rf_model = RandomForestClassifier(random_state=100, class_weight='balanced_subsample')
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform GridSearchCV on all training data
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

# Best model from GridSearchCV
best_rf_model = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')

# Train the final model on all available training data
best_rf_model.fit(X_scaled, y)

# Predict on the test set
y_test_pred = best_rf_model.predict(X_test_scaled)

# Save predictions to CSV
submission_df = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'Diagnosis': y_test_pred
})
submission_df.to_csv('sub_predictions.csv', index=False)
print("Predictions saved to 'sub_predictions.csv'.")

# Retrieve feature importances and selected feature names
feature_importances = best_rf_model.feature_importances_
selected_feature_names = X.columns[selector.get_support()]

# Create a DataFrame for better sorting and visualization
importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)  # Set ascending=False for descending order

# Plot the feature importance diagram
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Names')
plt.title('Feature Importance Diagram')
plt.tight_layout()
plt.show()

# Predict probabilities for the entire training set
y_probs = best_rf_model.predict_proba(X_scaled)

# Calculate ROC curve and AUC score
fpr, tpr, _ = roc_curve(y, y_probs[:, 1], pos_label=np.unique(y)[1])
roc_auc = auc(fpr, tpr)

# Plot the AUC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Chance Line (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
