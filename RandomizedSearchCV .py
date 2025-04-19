from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define the model
rf = RandomForestClassifier()

# Define hyperparameter search space
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Run Randomized Search
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring="f1", n_jobs=-1, verbose=2)
search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", search.best_params_)
