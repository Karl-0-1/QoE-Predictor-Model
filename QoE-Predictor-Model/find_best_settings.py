import numpy as np
import joblib

def find_best_settings(model, current_clients, current_interference):
    """
    Loops through all controllable parameters to find the best
    predicted throughput using the trained model.
    """
    
    # Define the search space for controllable parameters
    possible_widths = [20, 40, 80]
    possible_powers = range(10, 21) # 10 to 20 inclusive
    
    best_settings = {}
    best_predicted_throughput = -1.0
    
    print(f"\nOptimizing for: {current_clients} clients and {current_interference} interference...")
    
    # Loop through all combinations
    for width in possible_widths:
        for power in possible_powers:
            # Create the feature array for the model
            # Order must match the training: 
            # ['channel_width', 'tx_power', 'num_clients', 'interference_level']
            features = np.array([[width, power, current_clients, current_interference]])
            
            # Use the model to predict
            predicted_throughput = model.predict(features)[0]
            
            # Check if this is the best one so far
            if predicted_throughput > best_predicted_throughput:
                best_predicted_throughput = predicted_throughput
                best_settings = {
                    'channel_width': width,
                    'tx_power': power
                }
                
    return best_settings, best_predicted_throughput

# --- Main execution ---
try:
    # 1. Load the trained model
    qoe_model = joblib.load('qoe_predictor_model.pkl')
except FileNotFoundError:
    print("Error: qoe_predictor_model.pkl not found.")
    print("Please run train_model.py first.")
    exit()

# 2. EXAMPLE 1: Moderate conditions
clients_now = 25
interference_now = 30
best_config, best_qoe = find_best_settings(qoe_model, clients_now, interference_now)

print("\n--- Recommendation 1 ---")
print(f"Current Conditions: {clients_now} clients, {interference_now} interference")
print(f"Recommended Settings: {best_config}")
print(f"Predicted Throughput: {best_qoe:.2f} Mbps")


# 3. EXAMPLE 2: High-contention conditions
clients_now = 45
interference_now = 80
best_config, best_qoe = find_best_settings(qoe_model, clients_now, interference_now)

print("\n--- Recommendation 2 ---")
print(f"Current Conditions: {clients_now} clients, {interference_now} interference")
print(f"Recommended Settings: {best_config}")
print(f"Predicted Throughput: {best_qoe:.2f} Mbps")
