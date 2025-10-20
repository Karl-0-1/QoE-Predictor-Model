import pandas as pd
import numpy as np

num_samples = 1000

# 1. Create the features
channel_width = np.random.choice([20, 40, 80], num_samples)
# Use randint(10, 21) to include 20
tx_power = np.random.randint(10, 21, num_samples) 
num_clients = np.random.randint(5, 51, num_samples)
interference_level = np.random.randint(0, 101, num_samples)

# 2. Create the target (avg_throughput)
# We create a formula where width and power are 'good',
# and clients/interference are 'bad'.
# add 1 to the denominator to avoid division by zero.
base_signal = (channel_width * 5) + (tx_power - 10) # tx_power now adds a small boost
contention = num_clients + interference_level
noise = np.random.normal(0, 2, num_samples) # Add some random noise

avg_throughput = (base_signal / (contention + 1)) + noise
# Ensure throughput can't be negative
avg_throughput = np.maximum(0, avg_throughput) 

# 3. Create DataFrame and save
df = pd.DataFrame({
    'channel_width': channel_width,
    'tx_power': tx_power,
    'num_clients': num_clients,
    'interference_level': interference_level,
    'avg_throughput': avg_throughput
})

df.to_csv('qoe_data.csv', index=False)
print("qoe_data.csv created successfully with 1000 samples.")
