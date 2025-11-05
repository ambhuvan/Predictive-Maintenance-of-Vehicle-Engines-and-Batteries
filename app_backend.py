from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import math
from datetime import datetime, timedelta
import os
import traceback

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'models/best_tuned_engine_classifier_with_smote_pipeline.joblib')

AVG_SPEED_KPH = 50.0
CONDITION_LABELS = {0: 'Bad', 1: 'Good', 2: 'Moderate'}

EXPECTED_FEATURE_COLUMNS = [
    'engineRpm', 'lubOilPressure', 'fuelPressure', 'coolantPressure',
    'lubOilTemp', 'coolantTemp', 'viscosity'
]

# --- Load Model ---
try:
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model file not found at expected path: {MODEL_PATH}")
    model_pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ FATAL ERROR: {e}")
    model_pipeline = None
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model: {e}")
    model_pipeline = None

# --- Helper Functions ---
def calculate_viscosity(temp_celsius):
    if pd.isna(temp_celsius) or temp_celsius <= -273: return 10.0 # Return default on invalid input
    try:
         temperature_kelvin = temp_celsius + 273
         viscosity = 0.7 * math.exp(1500 / temperature_kelvin)
         return viscosity
    except (ValueError, OverflowError, TypeError, ZeroDivisionError):
         print(f"Warning: Viscosity calculation error for temp {temp_celsius}")
         return 10.0 # Return default

def calculate_part_health_scores(sensor_data):
    parts_health = {}
    defaults = {
        'lubOilPressure': 50, 'lubOilTemp': 75, 'viscosity': 10,
        'coolantPressure': 2, 'coolantTemp': 85, 'fuelPressure': 3.5
    }
    try:
        lop = sensor_data.get('lubOilPressure', defaults['lubOilPressure'])
        lot = sensor_data.get('lubOilTemp', defaults['lubOilTemp'])
        visc = sensor_data.get('viscosity', defaults['viscosity'])
        cp = sensor_data.get('coolantPressure', defaults['coolantPressure'])
        ct = sensor_data.get('coolantTemp', defaults['coolantTemp'])
        fp = sensor_data.get('fuelPressure', defaults['fuelPressure'])

        parts_health['oil_pump'] = np.clip(lop / 60, 0, 1) * 100
        parts_health['oil_filter'] = np.clip((15 - visc) / 10, 0, 1) * 100
        parts_health['coolant_pump'] = np.clip(cp / 3, 0, 1) * 100
        parts_health['thermostat'] = np.clip(1 - abs(ct - 85) / 20, 0, 1) * 100
        parts_health['fuel_pump'] = np.clip(fp / 4, 0, 1) * 100
        parts_health['fuel_filter'] = np.clip(fp / 4.5, 0, 1) * 100
    except (TypeError, ValueError) as e:
         print(f"Warning: Health score calculation error: {e}")
         return {k: 50.0 for k in ['oil_pump', 'oil_filter', 'coolant_pump', 'thermostat', 'fuel_pump', 'fuel_filter']}
    return {k: round(v, 1) for k, v in parts_health.items()}

def calculate_approx_time_left(distance_km, avg_speed_kph=AVG_SPEED_KPH):
    if pd.isna(distance_km) or distance_km <= 0 or avg_speed_kph <= 0: return "N/A"
    try:
        hours = distance_km / avg_speed_kph
        total_minutes = hours * 60
        h = int(total_minutes // 60)
        m = int(total_minutes % 60)
        time_str = f"{h} hours"
        if m > 0: time_str += f" and {m} mins"
        return time_str
    except (ValueError, TypeError): return "N/A"

def format_remaining_distance(km):
    if pd.isna(km) or km <= 0: return "0 km (Maintenance Due)"
    if km < 1000: return f"{km:.0f} km"
    if km < 10000: return f"{km/1000:.1f}k km"
    return f"{km/1000:.0f}k+ km"

def generate_dummy_recommendations(parts_health):
    recs = []
    now = datetime.now()
    for part, health in parts_health.items():
         if health <= 0: dist = 0
         elif health >= 100: dist = 50000
         else: dist = (health / 100) ** 2 * 5000

         time_left = calculate_approx_time_left(dist)
         if health <= 40:
             status = "🚨 CRITICAL"
             date = (now + timedelta(days=max(1, int(dist / 50)))).strftime('%Y-%m-%d')
             priority_level = "CRITICAL"
         elif health <= 70:
             status = "⚠️ MODERATE"
             date = (now + timedelta(days=max(7, int(dist / 50)))).strftime('%Y-%m-%d')
             priority_level = "MODERATE"
         else:
             status = "✅ GOOD"
             date = (now + timedelta(days=max(30, int(dist / 50)))).strftime('%Y-%m-%d')
             priority_level = "GOOD"
         
         recs.append({
             'part': part.replace('_', ' ').title(),
             'current_health': f"{health:.1f}%",
             'health_score': health, # <--- THIS IS THE FIX. ADDED THE RAW HEALTH SCORE.
             'priority': status,
             'priority_level': priority_level,
             'recommended_replacement_date': date,
             'remaining_kilometers': dist,
             'remaining_km_formatted': format_remaining_distance(dist),
             'approx_time_left': time_left
         })
    
    # This sorting line will now work because 'health_score' exists in the dictionary
    priority_map = {"CRITICAL": 0, "MODERATE": 1, "GOOD": 2}
    return sorted(recs, key=lambda x: (priority_map.get(x['priority_level'], 99), x['health_score']))


# --- Flask App Initialization ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Engine Health Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_condition():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded properly on startup"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # --- Prepare data for prediction ---
        input_data_dict = {}
        missing_in_request = []
        default_vals_for_missing = {
             'engineRpm': 1500, 'lubOilPressure': 50, 'fuelPressure': 3.5,
             'coolantPressure': 2.0, 'lubOilTemp': 80, 'coolantTemp': 85, 'viscosity': 10.0
        }

        if 'lubOilTemp' in data and pd.notna(data['lubOilTemp']):
             data['viscosity'] = calculate_viscosity(data['lubOilTemp'])
        else:
             data['viscosity'] = default_vals_for_missing['viscosity']
             if 'lubOilTemp' not in data:
                 missing_in_request.append('lubOilTemp')

        for feature in EXPECTED_FEATURE_COLUMNS:
             if feature in data and pd.notna(data[feature]):
                  input_data_dict[feature] = data[feature]
             else:
                  input_data_dict[feature] = default_vals_for_missing.get(feature, 0)
                  if feature not in missing_in_request:
                      missing_in_request.append(feature)

        if missing_in_request:
             print(f"Warning: Features missing/NaN in request, used defaults: {missing_in_request}")

        input_df = pd.DataFrame([input_data_dict], columns=EXPECTED_FEATURE_COLUMNS)

        # --- Make prediction ---
        prediction_code = model_pipeline.predict(input_df)[0]
        prediction_label = CONDITION_LABELS.get(int(prediction_code), 'Unknown')

        # --- Generate Health Scores & Recommendations ---
        parts_health = calculate_part_health_scores(input_data_dict)
        recommendations = generate_dummy_recommendations(parts_health)

        # --- Prepare response for the app ---
        response = {
            'overall_status': prediction_label,
            'parts_health': parts_health,
            'recommendations': recommendations,
            'current_readings': {k: round(v,2) if isinstance(v, (int, float)) else v for k,v in input_data_dict.items()}
        }

        print(f"DEBUG: Sending this JSON response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: An internal error occurred."}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)