from confluent_kafka import Consumer, KafkaException, KafkaError
import json
import joblib
import math
import numpy as np
import pandas as pd
import time

# Load trained models
try:
    engine_model = joblib.load('engine_condition_classifier.joblib')
    kmeans_model = joblib.load('viscosity_classifier.joblib')
    
    # Load individual parameter models if they exist
    parameter_models = {}
    for param in ["lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
        try:
            parameter_models[param] = joblib.load(f'{param}_classifier.joblib')
        except FileNotFoundError:
            print(f"Warning: {param} classifier not found. Will use only the main engine model.")
            
    print("Models loaded.")

except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please run model.py first to generate the models")
    exit(1)

# Kafka Consumer setup
kafka_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'engine-data-group',
    'auto.offset.reset': 'latest'
}
consumer = Consumer(kafka_config)
topic = 'test-topic'
consumer.subscribe([topic])

# Store latest values and timestamps
sensor_buffer = {}
last_known_values = {}  # Store previous values to handle missing data
last_flush_time = time.time()

# Sensor field mapping
field_map = {
    "Engine rpm": "engineRpm",
    "Lub oil pressure": "lubOilPressure",
    "Fuel pressure": "fuelPressure",
    "Coolant pressure": "coolantPressure",
    "lub oil temp": "lubOilTemp",
    "Coolant temp": "coolantTemp"
}

# Default fallback values if missing values persist
default_values = {
    "engineRpm": 1000,        # Default RPM
    "lubOilPressure": 50,     # Default Pressure
    "fuelPressure": 3.5,      # Default Fuel Pressure
    "coolantPressure": 2.0,   # Default Coolant Pressure
    "lubOilTemp": 75,         # Default Oil Temperature
    "coolantTemp": 80         # Default Coolant Temperature
}

# Color formatting for final condition display
color_map = {
    "Good": "\033[92m",     # Green
    "Moderate": "\033[93m", # Yellow
    "Bad": "\033[91m"       # Red
}
reset_color = "\033[0m"

def calculate_viscosity(temp_celsius):
    """Calculate viscosity and classify using K-Means."""
    temperature_kelvin = temp_celsius + 273  
    viscosity = 0.7 * math.exp(1500 / temperature_kelvin)  

    # Predict viscosity cluster
    cluster_label = kmeans_model.predict(np.array([[viscosity]]))[0]

    # Map clusters to conditions
    cluster_map = {0: "Low", 1: "Moderate", 2: "High"} # Assuming these maps to Good, Moderate, Bad
    viscosity_condition = cluster_map.get(cluster_label, "Unknown")

    return viscosity, viscosity_condition

def get_condition_text(condition_code):
    """Convert numeric condition code to text."""
    condition_map = {0: "Bad", 1: "Moderate", 2: "Good"}
    return condition_map.get(condition_code, "Unknown")

# Simple function to classify parameters directly
def classify_parameter_directly(param_name, value):
    """Directly classify a parameter value without ML model."""
    # Define thresholds for each parameter (adjust based on domain knowledge)
    thresholds = {
        "lubOilPressure": [(0, 30, "Bad"), (30, 60, "Moderate"), (60, float('inf'), "Good")],
        "fuelPressure": [(0, 2, "Bad"), (2, 4, "Moderate"), (4, float('inf'), "Good")],
        "coolantPressure": [(0, 1, "Bad"), (1, 3, "Moderate"), (3, float('inf'), "Good")],
        "lubOilTemp": [(0, 50, "Bad"), (50, 90, "Good"), (90, float('inf'), "Moderate")],  # Middle range is better for temp
        "coolantTemp": [(0, 60, "Bad"), (60, 95, "Good"), (95, float('inf'), "Moderate")],  # Middle range is better for temp
    }
    
    for lower, upper, category in thresholds[param_name]:
        if lower <= value < upper:
            return category
    return "Moderate"  # Default to moderate if no match

def process_collected_data():
    """Process buffered sensor data every 1.5 seconds."""
    global last_flush_time

    # Check if 1.5 seconds have passed
    if time.time() - last_flush_time < 1.5:
        return

    last_flush_time = time.time()

    # Check if we have data to process
    if not sensor_buffer:
        return

    # Ensure all required sensors have values
    complete_data = {}
    missing_fields = []

    for key in field_map.values():
        if key in sensor_buffer:
            complete_data[key] = sensor_buffer[key]
            last_known_values[key] = sensor_buffer[key]  # Update last known value
        elif key in last_known_values:
            complete_data[key] = last_known_values[key]  # Use last known value
        else:
            complete_data[key] = default_values[key]  # Use default value
            missing_fields.append(key)

    # Log missing fields
    if missing_fields:
        print(f"⚠️ Warning: Missing fields {missing_fields}, using last known/default values.")

    # Convert to feature vector
    feature_vector = [[
        complete_data["engineRpm"],
        complete_data["lubOilPressure"],
        complete_data["fuelPressure"],
        complete_data["coolantPressure"],
        complete_data["lubOilTemp"],
        complete_data["coolantTemp"]
    ]]

    # Convert to DataFrame to avoid warnings
    feature_df = pd.DataFrame(feature_vector, columns=field_map.values())

    # Get the number of features expected by the model
    try:
        num_features_expected = engine_model.n_features_in_
        # print(f"Model expects {num_features_expected} features. We have {len(feature_df.columns)}.") # Muted this line for cleaner output
        
        # If there's a feature mismatch, let's use a different approach
        if num_features_expected != len(feature_df.columns):
            # Use a direct classification approach instead of the model
            engine_condition_text = "Unknown"
            
            # Simple algorithm: if most parameters are good, engine is good
            param_conditions_temp = {} # Use temp var to avoid conflict with final parameter_conditions
            for param in ["lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
                if param in complete_data:
                    param_conditions_temp[param] = classify_parameter_directly(param, complete_data[param])
            
            # Count conditions
            condition_counts = {"Good": 0, "Moderate": 0, "Bad": 0}
            for condition in param_conditions_temp.values():
                condition_counts[condition] += 1
                
            # Determine overall condition based on counts
            if condition_counts["Bad"] >= 2:
                engine_condition_text = "Bad"
            elif condition_counts["Good"] >= 3:
                engine_condition_text = "Good"
            else:
                engine_condition_text = "Moderate"
        else:
            # Use the model as intended
            engine_condition_code = engine_model.predict(feature_df)[0]
            engine_condition_text = get_condition_text(engine_condition_code)
    except Exception as e:
        print(f"Error using model: {e}")
        # Fallback to direct classification
        engine_condition_text = "Unknown"
        
        # Simple algorithm: if most parameters are good, engine is good
        param_conditions_temp = {} # Use temp var to avoid conflict with final parameter_conditions
        for param in ["lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
            if param in complete_data:
                param_conditions_temp[param] = classify_parameter_directly(param, complete_data[param])
        
        # Count conditions
        condition_counts = {"Good": 0, "Moderate": 0, "Bad": 0}
        for condition in param_conditions_temp.values():
            condition_counts[condition] += 1
            
        # Determine overall condition based on counts
        if condition_counts["Bad"] >= 2:
            engine_condition_text = "Bad"
        elif condition_counts["Good"] >= 3:
            engine_condition_text = "Good"
        else:
            engine_condition_text = "Moderate"

    # Calculate viscosity
    viscosity, viscosity_condition = calculate_viscosity(complete_data['lubOilTemp'])

    # Analyze individual parameters (multi-class for each parameter)
    parameter_conditions = {}
    for param in ["lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
        if param in complete_data:
            try:
                if param in parameter_models:
                    # Use the ML model if available
                    param_condition_code = parameter_models[param].predict(feature_df)[0]
                    param_condition_text = get_condition_text(param_condition_code)
                else:
                    # Direct classification if model not available
                    param_condition_text = classify_parameter_directly(param, complete_data[param])
                    
                parameter_conditions[param] = param_condition_text
            except Exception as e:
                print(f"Error predicting {param}: {e}")
                # Direct classification as fallback
                parameter_conditions[param] = classify_parameter_directly(param, complete_data[param])

    print("\n" + "="*80)
    print(f"               {color_map.get(engine_condition_text, '')}REAL-TIME ENGINE HEALTH REPORT{reset_color}")
    print("="*80 + "\n")

    print("Overall Engine Status:")
    print(f"  Current Condition: {color_map.get(engine_condition_text, '')}{engine_condition_text.upper()}{reset_color}")
    
    # --- START REFINED OVERALL ENGINE STATUS OUTPUT ---
    if engine_condition_text == 'Good':
        print("  Outlook: All primary parameters are within healthy operating ranges. Optimal performance expected.")
        print("  Action Recommendation: Continue regular monitoring. No immediate intervention required.")
    elif engine_condition_text == 'Moderate':
        # Customizing Moderate overall status
        bad_count = sum(1 for cond in parameter_conditions.values() if cond == 'Bad')
        moderate_count = sum(1 for cond in parameter_conditions.values() if cond == 'Moderate')

        if bad_count > 0:
            print("  Outlook: **ATTENTION REQUIRED.** One or more critical parameters are showing 'Bad' readings, leading to an overall Moderate status.")
            print("  Action Recommendation: IMMEDIATE review of the 'Bad' parameter alerts below is crucial. Address those specific faults as a priority to prevent escalation.")
        elif moderate_count >= 2:
            print("  Outlook: Several parameters are trending towards suboptimal performance, indicating a developing pattern of concern.")
            print("  Action Recommendation: PROMPT review of all 'Moderate' parameter alerts below is recommended. Plan for pre-emptive maintenance to avoid future critical issues.")
        else: # Default moderate if only one minor moderate parameter
            print("  Outlook: One or more parameters show developing concerns. Vigilance is advised.")
            print("  Action Recommendation: Review individual parameter alerts below and address promptly. Increased operational awareness.")
    elif engine_condition_text == 'Bad':
        # Customizing Bad overall status
        bad_count = sum(1 for cond in parameter_conditions.values() if cond == 'Bad')

        if bad_count >= 2:
            print("  Outlook: **CRITICAL: MULTIPLE FAILURES DETECTED.** Multiple primary parameters are reporting 'Bad' conditions, indicating widespread system distress.")
            print("  Action Recommendation: IMMEDIATE SHUTDOWN AND ISOLATION OF THE ENGINE IS MANDATORY. Catastrophic failure is imminent. Consult individual alerts for specific component failures.")
        elif 'lubOilPressure' in parameter_conditions and parameter_conditions['lubOilPressure'] == 'Bad':
            print("  Outlook: **CRITICAL: LUBRICATION FAILURE.** The engine is experiencing critically low lubrication oil pressure, leading to an overall Bad status.")
            print("  Action Recommendation: IMMEDIATE EMERGENCY SHUTDOWN. Continued operation will lead to severe and irreversible internal engine damage. Prioritize oil system inspection.")
        elif 'coolantPressure' in parameter_conditions and parameter_conditions['coolantPressure'] == 'Bad':
            print("  Outlook: **CRITICAL: COOLING SYSTEM FAILURE.** The engine is experiencing critically low/high coolant pressure, leading to an overall Bad status.")
            print("  Action Recommendation: IMMEDIATE EMERGENCY SHUTDOWN. Overheating will occur rapidly, causing severe engine damage. Prioritize cooling system inspection.")
        else: # Default bad if no specific critical parameter identified, or single bad
            print("  Outlook: CRITICAL ISSUES DETECTED. High risk of immediate and severe engine damage or failure.")
            print("  Action Recommendation: IMMEDIATE ATTENTION REQUIRED. Consult individual parameter alerts for specific critical faults and illustrative urgency. Prepare for potential shutdown.")
    # --- END REFINED OVERALL ENGINE STATUS OUTPUT ---
    
    print("\n" + "-"*80)

    print("Individual Parameter Conditions:")
    print("-" * 80)
    
    # Process and print each parameter condition with detailed messages
    for param, condition in parameter_conditions.items():
        param_display = next((k for k, v in field_map.items() if v == param), param) # Get user-friendly name
        value = complete_data[param]
        print(f"\n- {param_display}: {value:.2f} -> {color_map.get(condition, '')}{condition}{reset_color}")
        
        # --- Predictive Distance Variable ---
        predictive_distance = "N/A" # Default for Good or unclassified

        # Specific messages for each parameter and condition
        if param == 'lubOilPressure':
            if condition == 'Bad':
                if value < 5: # Extremely critically low pressure
                    print(f"    Diagnosis: CRITICAL: Extremely low lub oil pressure ({value:.2f}). Imminent failure risk.")
                    print("    Impact: This value is dangerously low, suggesting severe oil pump failure or major leak. Bearings and camshafts are at extreme risk.")
                    print("    Recommended Action & Illustrative Urgency: **EMERGENCY SHUTDOWN REQUIRED IMMEDIATELY.** Contact maintenance. Engine will likely seize within minutes (illustrative: <0.5 hours).")
                    predictive_distance = "approx. <10 KMs" # Very short distance
                elif value < 15: # Critically low pressure (e.g., 5-15)
                    print(f"    Diagnosis: Critically low lub oil pressure ({value:.2f}). Severe and immediate issue detected.")
                    print("    Impact: High risk of rapid and significant engine damage, particularly affecting bearings and other lubricated components.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the fault within approximately 0.5 to 1 hour before irreversible engine breakdown occurs. Continued operation beyond this illustrative timeframe is highly risky.")
                    predictive_distance = "approx. 10-50 KMs"
                elif value < 30: # Low pressure (e.g., 15-30)
                    print(f"    Diagnosis: Low lub oil pressure ({value:.2f}). Immediate attention required.")
                    print("    Impact: High risk of rapid engine damage if prolonged. Bearings and camshafts are under stress.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the fault (e.g., in the oil pump, oil filter, or internal engine parts) within approximately 1 to 2 hours.")
                    predictive_distance = "approx. 50-100 KMs"
                else: # Fallback for 'Bad' (should be covered by ranges, but good to have)
                    print(f"    Diagnosis: Critically low lub oil pressure ({value:.2f}). Immediate attention required.")
                    print("    Impact: High risk of rapid engine damage.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address within approximately 0.5 to 2 hours.")
                    predictive_distance = "approx. <100 KMs"
            elif condition == 'Moderate':
                if value >= 30 and value < 45: # Moderate-low pressure range
                    print(f"    Diagnosis: Lub oil pressure ({value:.2f}) is moderately low. Developing concern.")
                    print("    Impact: Suboptimal lubrication leading to increased wear over prolonged operation. Check for minor leaks or a slightly worn pump.")
                    print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues in the oil system. This issue should be addressed within approximately 4 to 12 hours.")
                    predictive_distance = "approx. 200-500 KMs"
                else: # Moderate-high pressure range (if applicable, or general moderate)
                    print(f"    Diagnosis: Lub oil pressure ({value:.2f}) is slightly outside optimal range. Ongoing concern.")
                    print("    Impact: Reduced lubrication effectiveness which could lead to increased wear over prolonged operation. ")
                    print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues in the oil system. This issue should be addressed within approximately 12 to 24 hours to prevent escalation to a critical fault.")
                    predictive_distance = "approx. 500-1000 KMs"
            else: # Good
                print(f"    Diagnosis: Lub oil pressure ({value:.2f}) is within healthy operating parameters.")
                print("    Impact: Optimal lubrication is being maintained.")
                print("    Recommended Action & Urgency: Monitor regularly. No immediate action required.")
                predictive_distance = "approx. >5000 KMs"
        
        elif param == 'fuelPressure':
            if condition == 'Bad':
                print(f"    Diagnosis: Critically low/high fuel pressure ({value:.2f}). Indicates a severe fuel system issue.")
                print("    Impact: Risk of engine misfires, stalling, or fuel pump damage.")
                print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the fuel system fault within approximately 0.5 to 2 hours to prevent engine failure or damage.")
                predictive_distance = "approx. <100 KMs"
            elif condition == 'Moderate':
                print(f"    Diagnosis: Fuel pressure ({value:.2f}) is outside optimal range, indicating a potential issue.")
                print("    Impact: May lead to decreased engine performance or efficiency over time.")
                print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues in the fuel lines, filter, or pump. Address within approximately 4 to 24 hours.")
                predictive_distance = "approx. 200-1000 KMs"
            else: # Good
                print(f"    Diagnosis: Fuel pressure ({value:.2f}) is within healthy operating parameters.")
                print("    Impact: Optimal fuel delivery is being maintained.")
                print("    Recommended Action & Urgency: Monitor regularly. No immediate action required.")
                predictive_distance = "approx. >5000 KMs"
        
        elif param == 'coolantPressure':
            if condition == 'Bad':
                if value < 0.2: # Extremely critically low coolant pressure
                    print(f"    Diagnosis: CRITICAL: Extremely low coolant pressure ({value:.2f}). Immediate and severe risk of engine overheating.")
                    print("    Impact: High probability of head gasket failure, engine block cracking, or internal seizing due to extreme heat.")
                    print("    Recommended Action & Illustrative Urgency: **EMERGENCY SHUTDOWN REQUIRED IMMEDIATELY.** Contact maintenance. Catastrophic engine failure is likely within minutes (illustrative: < 0.25 hours).")
                    predictive_distance = "approx. <5 KMs"
                elif value < 0.7: # Critically low pressure (e.g., 0.2-0.7)
                    print(f"    Diagnosis: Critically low coolant pressure ({value:.2f}). Severe cooling system issue.")
                    print("    Impact: High risk of engine overheating, head gasket failure, or severe internal damage.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the cooling system fault (e.g., leaks, pump, or radiator issues) within approximately 0.5 to 1 hour.")
                    predictive_distance = "approx. 5-25 KMs"
                elif value < 1.0: # Low pressure (e.g., 0.7-1.0)
                    print(f"    Diagnosis: Low coolant pressure ({value:.2f}). Immediate attention required.")
                    print("    Impact: High risk of rapid engine overheating if prolonged.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the cooling system fault within approximately 1 to 2 hours.")
                    predictive_distance = "approx. 25-75 KMs"
                else: # Fallback for 'Bad'
                    print(f"    Diagnosis: Critically low/high coolant pressure ({value:.2f}). Immediate attention required.")
                    print("    Impact: High risk of engine overheating.")
                    print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address within approximately 0.5 to 2 hours.")
                    predictive_distance = "approx. <75 KMs"
            elif condition == 'Moderate':
                if value >= 1.0 and value < 2.0: # Moderate-low pressure range
                    print(f"    Diagnosis: Coolant pressure ({value:.2f}) is moderately low. Indicates a developing concern.")
                    print("    Impact: Potential for reduced cooling efficiency. Check for minor leaks, air pockets, or a slightly weak water pump.")
                    print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues in the cooling system. This issue should be addressed within approximately 4 to 12 hours.")
                    predictive_distance = "approx. 100-300 KMs"
                else: # Moderate-high pressure range (if applicable, or general moderate)
                    print(f"    Diagnosis: Coolant pressure ({value:.2f}) is slightly outside optimal range, indicating a developing concern that needs attention.")
                    print("    Impact: Potential for reduced cooling efficiency, leading to increased wear or overheating over prolonged periods. If ignored, this could escalate to a critical fault.")
                    print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues in the cooling system (e.g., leaks, pump, or blockages). This issue should be addressed within approximately 12 to 24 hours, or before the next operational cycle, to prevent further degradation and potential breakdown.")
                    predictive_distance = "approx. 300-800 KMs"
            else: # Good
                print(f"    Diagnosis: Coolant pressure ({value:.2f}) is within healthy operating parameters.")
                print("    Impact: Optimal cooling system function is being maintained.")
                print("    Recommended Action & Urgency: Monitor regularly. No immediate action required.")
                predictive_distance = "approx. >5000 KMs"
                
        elif param == 'lubOilTemp':
            if condition == 'Bad':
                print(f"    Diagnosis: Critically high/low lub oil temperature ({value:.2f}). Indicates a severe lubrication system issue.")
                print("    Impact: High risk of oil breakdown, increased friction, and severe engine component wear.")
                print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the oil temperature fault (e.g., cooling system, oil level, or sensor issues) within approximately 0.5 to 2 hours to prevent irreversible engine damage.")
                predictive_distance = "approx. <100 KMs"
            elif condition == 'Moderate':
                print(f"    Diagnosis: Lub oil temperature ({value:.2f}) is outside optimal range, indicating a potential concern.")
                print("    Impact: Suboptimal lubrication leading to increased engine wear over time.")
                print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues with oil cooling, level, or sensor calibration. Address within approximately 4 to 24 hours.")
                predictive_distance = "approx. 200-1000 KMs"
            else: # Good
                print(f"    Diagnosis: Lub oil temperature ({value:.2f}) is within healthy operating parameters.")
                print("    Impact: Optimal oil viscosity and lubrication effectiveness.")
                print("    Recommended Action & Urgency: Monitor regularly. No immediate action required.")
                predictive_distance = "approx. >5000 KMs"

        elif param == 'coolantTemp':
            if condition == 'Bad':
                print(f"    Diagnosis: Critically high/low coolant temperature ({value:.2f}). Indicates a severe engine cooling issue.")
                print("    Impact: High risk of engine overheating, seizing, or thermal stress damage.")
                print("    Recommended Action & Illustrative Urgency: CRITICAL: IMMEDIATE SHUTDOWN AND INSPECTION REQUIRED. Address the cooling system fault (e.g., thermostat, radiator, water pump issues) within approximately 0.5 to 2 hours to prevent catastrophic engine failure.")
                predictive_distance = "approx. <100 KMs"
            elif condition == 'Moderate':
                print(f"    Diagnosis: Coolant temperature ({value:.2f}) is outside optimal range, indicating a potential concern.")
                print("    Impact: Suboptimal cooling which could lead to increased engine wear or reduced efficiency over time.")
                print("    Recommended Action & Illustrative Urgency: CONCERNING: PROMPT INSPECTION RECOMMENDED. Investigate potential issues with cooling system efficiency or sensor calibration. Address within approximately 4 to 24 hours.")
                predictive_distance = "approx. 200-1000 KMs"
            else: # Good
                print(f"    Diagnosis: Coolant temperature ({value:.2f}) is within healthy operating parameters.")
                print("    Impact: Optimal engine temperature regulation.")
                print("    Recommended Action & Urgency: Monitor regularly. No immediate action required.")
                predictive_distance = "approx. >5000 KMs"

        # Print Predictive Distance for all conditions
        print(f"    Predictive Distance (Illustrative): {predictive_distance}")


    # Clear buffer after processing
    sensor_buffer.clear()

def process_message(message):
    """Store received sensor data in buffer."""
    try:
        data = json.loads(message)

        # Extract values
        sensor_id = data.get("sensor_id")
        value = data.get("value")

        # Validate incoming data
        if sensor_id not in field_map.values():
            print(f"Skipping invalid sensor: {sensor_id}")
            return

        # Store latest value in buffer
        sensor_buffer[sensor_id] = value

    except json.JSONDecodeError:
        print("Invalid JSON received!")
    except KeyError as e:
        print(f"Missing field: {e}")

try:
    print(f"Subscribed to topic: {topic}")
    print("Monitoring engine data with multi-class classification for each parameter...")
    while True:
        msg = consumer.poll(1.0)  # Polls messages every 1 second

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"End of partition reached: {msg.topic()} {msg.partition()}")
            else:
                raise KafkaException(msg.error())
        else:
            process_message(msg.value().decode('utf-8'))

        # Process collected data every 1.5 seconds
        process_collected_data()

except KeyboardInterrupt:
    print("Consumer interrupted by user")
finally:
    consumer.close()
    print("Kafka consumer closed.")