import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop unnecessary columns like 'PatientID' and 'DoctorInCharge'
X = train_df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = train_df['Diagnosis']
X_test = test_df.drop(columns=['PatientID', 'DoctorInCharge'])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define and train the Support Vector Machine (SVM) model
# Use probability=True to enable probability estimation for AUC calculation
svm = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# Evaluate the model on the validation data
y_val_pred = svm.predict(X_val_scaled)
y_val_pred_proba = svm.predict_proba(X_val_scaled)[:, 1]  # Probabilities for the positive class

print('Classification Report on Validation Data:')
print(classification_report(y_val, y_val_pred))

# Calculate AUC on the validation data
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f'AUC (Area Under the Curve) on Validation Data: {auc_score:.4f}')

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Predict on test data using the trained model
y_test_pred = svm.predict(X_test_scaled)

# Save the predictions in the required format
submission_df = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'Diagnosis': y_test_pred
})
submission_df.to_csv('svm_predictions.csv', index=False)

print("Predictions saved to 'svm_predictions.csv'.")
