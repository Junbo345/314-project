import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prepare datasets
X = train_df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = train_df['Diagnosis']
X_test = test_df.drop(columns=['PatientID', 'DoctorInCharge'])

# Feature selection using statistical tests
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)


# Store selected feature names for later use
selected_feature_names = X.columns[selector.get_support()].tolist()

# Scale features to [0,1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(X_test_selected)


def test_model_stability(X, y, n_iterations=10):
    """
    Tests model performance stability across different data splits

    Parameters:
    X : feature matrix
    y : target variable
    n_iterations : number of test iterations

    Returns:
    feature_importances_matrix : matrix of feature importance scores across iterations
    """
    accuracies = []
    auc_scores = []
    feature_importances_matrix = []

    print("\n=== Model Stability Test ===")
    print(f"Running {n_iterations} iterations with different random splits...")

    for i in range(n_iterations):
        # Create new model with same parameters but different random seed
        model = RandomForestClassifier(
            max_depth=6,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=300,
            random_state=i * 100,
            class_weight='balanced_subsample'
        )

        # Split data with different random seed each time
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=i * 100
        )

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model and make predictions
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        y_probs = model.predict_proba(X_val_scaled)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        fpr, tpr, _ = roc_curve(y_val, y_probs)
        auc_score = auc(fpr, tpr)

        # Store results
        accuracies.append(accuracy)
        auc_scores.append(auc_score)
        feature_importances_matrix.append(model.feature_importances_)

    # Analyze results
    print("\nPerformance Stability Results:")
    print(f"Accuracy: mean={np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"AUC Score: mean={np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

    return np.array(feature_importances_matrix)


def analyze_feature_importance_stability(feature_importances_matrix, feature_names):
    """
    Analyzes the stability of feature importances across iterations

    Parameters:
    feature_importances_matrix : matrix of feature importance scores
    feature_names : list of feature names
    """
    print("\n=== Feature Importance Stability Analysis ===")

    # Calculate mean and standard deviation of feature importances
    importance_means = np.mean(feature_importances_matrix, axis=0)
    importance_stds = np.std(feature_importances_matrix, axis=0)

    # Calculate coefficient of variation (CV)
    cvs = importance_stds / importance_means

    # Create and sort results
    stability_results = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance': importance_means,
        'Std Dev': importance_stds,
        'CV': cvs
    }).sort_values('Mean Importance', ascending=False)

    print("\nFeature Importance Stability:")
    print(stability_results)

    # Visualize feature importance stability
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        range(len(feature_names)),
        importance_means,
        yerr=importance_stds,
        fmt='o',
        capsize=5
    )
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Stability\n(Error bars show standard deviation)')
    plt.tight_layout()
    plt.show()


# Create and train the optimized Random Forest model
rf_model = RandomForestClassifier(
    max_depth=6,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=300,
    random_state=100,
    class_weight='balanced_subsample'
)

# Train the model
rf_model.fit(X_scaled, y)

# Run stability tests
print("\nRunning stability tests...")
feature_importances_matrix = test_model_stability(X_selected, y)

# Analyze feature importance stability
analyze_feature_importance_stability(
    feature_importances_matrix,
    selected_feature_names
)

# Generate predictions
y_test_pred = rf_model.predict(X_test_scaled)

# Save predictions
submission_df = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'Diagnosis': y_test_pred
})
submission_df.to_csv('sub_predictions8.csv', index=False)
print("\nPredictions saved to 'sub_predictions.csv'")

# Create feature importance visualization
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Names')
plt.title('Feature Importance Diagram')
plt.tight_layout()
plt.show()

# Generate and plot ROC curve
y_probs = rf_model.predict_proba(X_scaled)
fpr, tpr, _ = roc_curve(y, y_probs[:, 1], pos_label=np.unique(y)[1])
roc_auc = auc(fpr, tpr)

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