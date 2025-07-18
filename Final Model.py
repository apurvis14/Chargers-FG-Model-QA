import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, r2_score
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# Load and prepare Data
data = pd.read_csv('data/FG Data1.csv')

# Feature and Target Separation
X = data.drop(columns=['success', 'kicker_player_name', 'season', 
                       'distance_bin', 'roof', 'posteam_type', 
                       'cumulative_attempts', 'cumulative_made', 
                       'season_cum_attempts', 'season_cum_made',
                       'bin_cum_made', 'bin_cum_attempts',
                       'kicker_player_id', 'stadium_id', 'special',
                       'career_fg_pct', 'last_5_fg_pct', 'clutch_kick', 'fg_prob', 'ep', 'posteam']).astype(float)
y = data['success'].values

# Training Set Split for XGBoost
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the Data for XGBoost
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


# Logistic Regression Model within Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Scale within pipeline
    ('logreg', LogisticRegression(max_iter=1000))
])

# Train and Test of Logistic Regression Model
pipeline.fit(X, y)  # Fit on full data
log_probs = pipeline.predict_proba(X_test)[:, 1]  # Predict on test set

# Save Logistic Regression Model for Future Use
joblib.dump(pipeline, 'logistic_regression_model.joblib')

# XGBoost Model with Hyperparameters
xgb_model = XGBClassifier(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Train and Test of XGBoost Model
xgb_model.fit(X_train_scaled, y_train.ravel())
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Save XGBoost Model for Future Use
joblib.dump(xgb_model, 'xgboost_model.joblib')

print(f"Logistic Regression Test AUC: {roc_auc_score(y_test, log_probs):.4f}")
print(f"XGBoost Test AUC: {roc_auc_score(y_test, xgb_probs):.4f}")

# Ensemble Model (80% Logistic, 20% XGBoost)
ensemble_prob = 0.8 * log_probs + 0.2 * xgb_probs

# Evaluation of Ensemble Model
test_auc = roc_auc_score(y_test, ensemble_prob)
test_accuracy = accuracy_score(y_test, (ensemble_prob > 0.5).astype(int))
print(f"Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Calibration Curve for Ensemble Model
prob_true, prob_pred = calibration_curve(y_test, ensemble_prob, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Ensemble Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('Model Results Images/Final Model Calibration Curve.png')
plt.show()

# AUC/ROC Curve for Ensemble Model
fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o', label='Ensemble Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('Model Results Images/Final Model ROC Curve.png')
plt.show()

# Save Test Predictions
results = pd.DataFrame(X_test, columns=X.columns)
results['logistic_prob'] = log_probs
results['xgb_prob'] = xgb_probs
results['ensemble_prob'] = ensemble_prob
results['fg_prob'] = data.loc[X_test.index, 'fg_prob'].values
results['actual'] = y_test
results.to_csv('data/FG Predictions.csv', index=False)

# Scatter Plot Ensemble Probability vs FG Probability with R^2 Adjusted Value
# Calculate Adjusted R^2
r2 = r2_score(data.loc[X_test.index, 'fg_prob'], ensemble_prob)
n = len(ensemble_prob)
p = 1 
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(data.loc[X_test.index, 'fg_prob'], ensemble_prob, alpha=0.5)
plt.text(0.1, 0.8, f'$R^2$ = {adjusted_r2:.4f}', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))
plt.title('Ensemble Predicted Probability vs Online FG Probability Data')
plt.xlabel('Online FG Probability Data')
plt.ylabel('Ensemble Predicted Probability')
plt.grid(True)
plt.savefig('Model Results Images/Final Comparison (Ensemble vs NFLVerse) R2.png') 
plt.show()

# Apply Ensemble Model to Full Data for Future Use and Save as CSV
x_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
full_ensemble_prob = 0.8 * pipeline.predict_proba(X)[:, 1] + 0.2 * xgb_model.predict_proba(x_scaled)[:, 1]

data['predicted_probability'] = full_ensemble_prob
data.to_csv('data/Full FG Data with Predictions.csv', index=False)
