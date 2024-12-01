import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop unnecessary columns like 'PatientID' and 'DoctorInCharge'
X_train = train_df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y_train = train_df['Diagnosis']
X_test = test_df.drop(columns=['PatientID', 'DoctorInCharge'])

# Feature selection with a tuned 'k' value for SelectKBest
best_k = 15  # experimenting with more features, adjust based on dataset and feature importances
selector = SelectKBest(score_func=f_classif, k=best_k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Experimenting with MinMaxScaler for scaled data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define a Random Forest model with a wider hyperparameter grid for tuning
rf_model = RandomForestClassifier(random_state=100, class_weight='balanced_subsample')  # Adjusting class weights
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Use GridSearchCV with cross-validation and more hyperparameter options
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model from cross-validation
best_rf_model = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')

# Evaluate the best model on the training data
y_train_pred = best_rf_model.predict(X_train_scaled)
print('Classification Report on Training Data:')
print(classification_report(y_train, y_train_pred))

# Predict on test data using the best model
y_test_pred = best_rf_model.predict(X_test_scaled)

# Save the predictions in the required format
submission_df = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'Diagnosis': y_test_pred
})
submission_df.to_csv('sub_predictions.csv', index=False)

selected_features = X_train.columns[selector.get_support()]
print("Selected Features:")
print(selected_features)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Feature Importance
importances = best_rf_model.feature_importances_

# Create DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Check the feature importance DataFrame
print("Feature Importance DataFrame:")
print(feature_importance_df)

# Plot Feature Importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='brown')
plt.xlabel('Variable Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance in Random Forest', fontsize=14)
plt.gca().invert_yaxis()  # Flip y-axis for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


if set(y_train.unique()) != {0, 1}:
    y_train = (y_train == y_train.unique().max()).astype(int)  # 转为二分类标签


y_train_proba = best_rf_model.predict_proba(X_train_scaled)[:, 1]

# ROC 曲线和 AUC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, pos_label=1)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Training ROC curve (AUC = {roc_auc_train:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Data')
plt.legend(loc="lower right")
plt.grid()
plt.show()