import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

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
X = df[['Open', 'High', 'Low', 'MA_5', 'MA_20', 'Prev_Close']].copy()
y = df['Close'].copy()

# Initialize storage for results
results = {
    "RMSE": [],
    "MAE": [],
    "R_squared": [],
    "Next_Day_Prediction": []
}

# Number of runs
n_runs = 100

# Run the model multiple times
for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=405,
        max_depth=13,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features=6,
        bootstrap=True
    )
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results["RMSE"].append(rmse)
    results["MAE"].append(mae)
    results["R_squared"].append(r2)

    print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

    # Predict the next day's closing price
    latest_data = df.iloc[-1]
    next_day_features = pd.DataFrame([{
        'Open': latest_data['Open'],
        'High': latest_data['High'],
        'Low': latest_data['Low'],
        'MA_5': latest_data['MA_5'],
        'MA_20': latest_data['MA_20'],
        'Prev_Close': latest_data['Close']
    }])
    next_day_features_scaled = scaler.transform(next_day_features)
    next_day_prediction = model.predict(next_day_features_scaled)

    # Store the next day's prediction
    results["Next_Day_Prediction"].append(next_day_prediction[0])

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Print the average performance metrics
print("\nAverage Metrics Over All Runs:")
print(results_df[["RMSE", "MAE", "R_squared"]].mean())

# Cross-Validation with pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestRegressor(n_estimators=405,
        max_depth=13,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features=6,
        bootstrap=True))
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Cross-Validated RMSE: {rmse_scores.mean():.5f}")

# Validation set evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
valid_rmse = np.sqrt(mean_squared_error(y_valid, model.predict(X_valid)))
print(f"Train RMSE: {train_rmse:.5f}, Validation RMSE: {valid_rmse:.5f}")

# Print the average next-day prediction
avg_next_day_prediction = np.mean(results_df["Next_Day_Prediction"])
print(f"Average Predicted Closing Price for the Next Day: ${avg_next_day_prediction:.2f}")

importances = model.feature_importances_
feature_names = X.columns
important_features = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(important_features)

