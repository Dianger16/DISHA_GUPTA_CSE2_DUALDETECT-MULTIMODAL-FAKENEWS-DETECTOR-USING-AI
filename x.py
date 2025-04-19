import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the best model
best_model = joblib.load("best_random_forest_model.pkl")

# Load test data
X_train, X_test, y_train, y_test = joblib.load("train_test_data.pkl")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Final Classification Report:\n", classification_report(y_test, y_pred))
