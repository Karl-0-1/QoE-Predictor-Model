# QoE Predictor Model

This project builds a simple Machine Learning model (Random Forest) to predict network Quality of Experience (QoE) based on Radio Resource Management (RRM) parameters like channel width, power, and client load.

This model serves as the core for an AI-assisted optimization engine.

## Features
* **Data Synthesis**: `generate_data.py` creates a realistic 1000-sample dataset.
* **Model Training**: `train_model.py` trains a `RandomForestRegressor` on the data and saves the model.
* **Optimization**: `find_best_settings.py` loads the trained model and runs a simple "brute-force" search to find the optimal Wi-Fi settings for a given network condition.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Karl-0-1/QoE-Predictor-Model.git](https://github.com/Karl-0-1/QoE-Predictor-Model.git)
    cd QoE-Predictor-Model
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the full pipeline:**

    **Step 1. Generate the data:**
    ```bash
    python generate_data.py
    ```
    *This will create `qoe_data.csv` (which is ignored by git).*

    **Step 2. Train the model:**
    ```bash
    python train_model.py
    ```
    *This will train the model, show the R² score, and save `qoe_predictor_model.pkl`.*

    **Step 3. Run the optimizer:**
    ```bash
    python find_best_settings.py
    ```

## Example Output

Running the optimizer (`find_best_settings.py`) will give you a recommendation like this:

```
Optimizing for: 25 clients and 30 interference...

--- Recommendation 1 ---
Current Conditions: 25 clients, 30 interference
Recommended Settings: {'channel_width': 80, 'tx_power': 20}
Predicted Throughput: 7.79 Mbps

Optimizing for: 45 clients and 80 interference...

--- Recommendation 2 ---
Current Conditions: 45 clients, 80 interference
Recommended Settings: {'channel_width': 80, 'tx_power': 20}
Predicted Throughput: 3.51 Mbps
```
