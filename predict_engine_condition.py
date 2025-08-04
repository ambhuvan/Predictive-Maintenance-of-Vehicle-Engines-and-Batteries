import pandas as pd
import joblib
import os

# --- Configuration ---
# Path to your saved model
MODEL_PATH = 'models/best_tuned_engine_classifier_with_smote_pipeline.joblib'

# Define the feature columns (all columns except 'engineCondition')
# Make sure these match the columns used for training (X_train)
FEATURE_COLUMNS = [
    'engineRpm', 'lubOilPressure', 'fuelPressure', 'coolantPressure',
    'lubOilTemp', 'coolantTemp', 'viscosity'
]

# Mapping for the output predictions to readable labels
CONDITION_LABELS = {
    0: 'Bad',
    1: 'Good',
    2: 'Moderate'
}

def load_model(path):
    """Loads a trained model from a .joblib file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    print(f"Loading model from: {path}")
    model = joblib.load(path)
    print("Model loaded successfully.")
    return model

def predict_new_data(model, new_data_df):
    """
    Makes predictions on new data using the loaded model.
    The model (pipeline) handles scaling automatically.
    """
    # Ensure the new data has the same feature columns as the training data
    # Drop any 'engineCondition' or 'timestamp' columns if present in new data
    data_for_prediction = new_data_df.copy()
    if 'engineCondition' in data_for_prediction.columns:
        data_for_prediction = data_for_prediction.drop(columns=['engineCondition'])
    if 'timestamp' in data_for_prediction.columns:
        data_for_prediction = data_for_prediction.drop(columns=['timestamp'])
    elif 'Timestamp' in data_for_prediction.columns:
        data_for_prediction = data_for_prediction.drop(columns=['Timestamp'])

    # Reorder columns to match the training data's feature order
    # This is crucial for consistent predictions
    try:
        data_for_prediction = data_for_prediction[FEATURE_COLUMNS]
    except KeyError as e:
        print(f"Error: Missing expected feature column(s) in new data: {e}")
        print(f"Expected columns: {FEATURE_COLUMNS}")
        print(f"New data columns: {data_for_prediction.columns.tolist()}")
        return None

    print(f"\nMaking predictions on {len(data_for_prediction)} new engine records...")
    predictions_numeric = model.predict(data_for_prediction)

    # Convert numeric predictions to readable labels
    predictions_labels = [CONDITION_LABELS.get(p, 'Unknown') for p in predictions_numeric]

    return predictions_labels, predictions_numeric

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Load your trained model
        best_model = load_model(MODEL_PATH)

        # --- Create some dummy new engine data for demonstration ---
        # In a real scenario, this would come from sensors, a new file, etc.
        # Ensure column names match your training data features!
        new_engine_data = pd.DataFrame({
            'engineRpm': [2500, 2000, 3000, 1800],
            'lubOilPressure': [2.5, 1.8, 3.2, 1.5],
            'fuelPressure': [3.0, 2.5, 3.5, 2.0],
            'coolantPressure': [1.5, 1.2, 1.8, 1.0],
            'lubOilTemp': [85.0, 75.0, 95.0, 70.0],
            'coolantTemp': [90.0, 80.0, 100.0, 78.0],
            'viscosity': [10.0, 8.0, 12.0, 7.5]
            # No 'engineCondition' here, as we're predicting it!
        })

        # You can also load new data from a CSV, for example:
        # new_engine_data = pd.read_csv('path/to/your/new_engine_data.csv')


        # Make predictions
        predicted_labels, predicted_numeric = predict_new_data(best_model, new_engine_data)

        if predicted_labels is not None:
            print("\n--- Predictions for New Engine Data ---")
            for i, (label, num) in enumerate(zip(predicted_labels, predicted_numeric)):
                print(f"Engine Record {i+1}: Predicted Condition: {label} (Code: {num})")

            # You might want to combine the new data with predictions for output
            new_engine_data['Predicted_Condition_Code'] = predicted_numeric
            new_engine_data['Predicted_Condition_Label'] = predicted_labels
            print("\nNew Data with Predictions:")
            print(new_engine_data)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'models' directory and 'best_tuned_engine_classifier_with_smote_pipeline.joblib' file exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")