import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint
from collections import Counter

# Load the dataset
df = pd.read_csv("D:\\Documents\\MScComputerScience\\AppliedAI\\nasdq.csv")

# Convert Date to datetime format and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Feature engineering: Create moving averages
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Lagged features
df['Prev_Close'] = df['Close'].shift(1)

# Handle missing values
df = df.dropna()

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'Prev_Close']].copy()
y = df['Close'].copy()

# Define the pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestRegressor())
])

# Define the hyperparameter grid
param_distributions = {
    'model__n_estimators': randint(50, 1000),           # Number of trees
    'model__max_depth': randint(5, 30),                # Maximum depth of trees
    'model__min_samples_split': randint(2, 20),        # Minimum number of samples to split
    'model__min_samples_leaf': randint(1, 10),         # Minimum number of samples in a leaf
    'model__max_features': [6],    # Valid values for max_features
    'model__bootstrap': [True, False]                  # Whether to use bootstrap sampling
}

#'sqrt', 'log2', None, 1, 2, 3, 4, 5,

# Initialize storage for results
n_runs = 100
results = {
    "RMSE": [],
    "MAE": [],
    "R_squared": [],
    "Best_Parameters": []
}

# Run the process 100 times
for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=5,       # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Optimizing for RMSE
        n_jobs=-1,  # Use all available processors
        error_score='raise'  # Raise errors instead of setting NaN scores
    )

    # Fit Randomized Search
    random_search.fit(X_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_
    results["Best_Parameters"].append(best_params)

    # Evaluate the best model on the test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results["RMSE"].append(rmse)
    results["MAE"].append(mae)
    results["R_squared"].append(r2)

    print(f"  RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# Calculate average performance metrics
avg_rmse = np.mean(results["RMSE"])
avg_mae = np.mean(results["MAE"])
avg_r2 = np.mean(results["R_squared"])

# Calculate the most frequently selected best parameters
param_counts = Counter(tuple(sorted(best_param.items())) for best_param in results["Best_Parameters"])
most_common_params = param_counts.most_common(1)[0]

# Output results
print("\nAverage Metrics Over All Runs:")
print(f"  Average RMSE: {avg_rmse:.5f}")
print(f"  Average MAE: {avg_mae:.5f}")
print(f"  Average R²: {avg_r2:.5f}")

print("\nMost Frequently Selected Best Hyperparameters:")
for param, value in dict(most_common_params[0]).items():
    print(f"  {param}: {value}")
print(f"  Frequency: {most_common_params[1]}")
