import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns

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

# Define and train the SVM model
svm_model = SVC(
    kernel='rbf',
    probability=True,  # Required for probability predictions
    class_weight='balanced',
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model on the validation data
y_val_pred = svm_model.predict(X_val_scaled)
y_val_pred_proba = svm_model.predict_proba(X_val_scaled)[:, 1]  # Probabilities for the positive class

# Classification Report
print('Classification Report on Validation Data:')
print(classification_report(y_val, y_val_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix as a Heatmap
plt.figure(figsize=(10, 8))  # Widescreen aspect ratio
sns.set_context("poster")  # Set context for presentation-quality visuals
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.title('Confusion Matrix', fontsize=20, weight='bold', pad=20)
plt.xlabel('Predicted Labels', fontsize=16, labelpad=15)
plt.ylabel('True Labels', fontsize=16, labelpad=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Calculate AUC on the validation data
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f'AUC (Area Under the Curve) on Validation Data: {auc_score:.4f}')

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure(figsize=(10, 8))
sns.lineplot(x=fpr, y=tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color='red', label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=16, labelpad=15)
plt.ylabel('True Positive Rate', fontsize=16, labelpad=15)
plt.title('ROC Curve', fontsize=20, weight='bold', pad=20)
plt.legend(loc='lower right', fontsize=14)
plt.grid(alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Predict on test data using the trained model
y_test_pred = svm_model.predict(X_test_scaled)

# Save the predictions in the required format
submission_df = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'Diagnosis': y_test_pred
})
submission_df.to_csv('sub_predictions_svm.csv', index=False)

print("Predictions saved to 'sub_predictions_svm.csv'.")
