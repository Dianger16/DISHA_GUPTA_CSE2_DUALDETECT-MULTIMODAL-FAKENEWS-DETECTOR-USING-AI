import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (Replace with your actual data)
X, y = ...  # Your feature matrix and target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
rf = RandomForestClassifier()

# Define hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

# Set up RandomizedSearchCV
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)

# Train the model
search.fit(X_train, y_train)

# Get best parameters
print("Best Parameters:", search.best_params_)

# Evaluate model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "best_random_forest_model.pkl")
print("Model saved successfully!")
