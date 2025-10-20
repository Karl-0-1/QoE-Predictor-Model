import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib # For saving the model

# 1. Load the data
try:
    df = pd.read_csv('qoe_data.csv')
except FileNotFoundError:
    print("Error: qoe_data.csv not found.")
    print("Please run generate_data.py first.")
    exit()

# 2. Define Features (X) and Target (y)
features = ['channel_width', 'tx_power', 'num_clients', 'interference_level']
target = 'avg_throughput'

X = df[features]
y = df[target]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# n_estimators=100 is a good default
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training complete.")

# 5. Evaluate the Model
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
# R-squared should be very high (e.g., > 0.95) since the data has a clear formula
print(f"R-squared (R²): {r2:.4f}") 
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 6. Save the trained model for the bonus step
joblib.dump(model, 'qoe_predictor_model.pkl')
print(f"\nModel saved to qoe_predictor_model.pkl")
