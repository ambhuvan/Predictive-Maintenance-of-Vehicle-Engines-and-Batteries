import pandas as pd
import joblib
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = "engines_dataset-train_multi_class.csv" 
df = pd.read_csv(data_path)

# Define expected field names and map them to correct ones
field_map = {
    "Engine rpm": "engineRpm",
    "Lub oil pressure": "lubOilPressure",
    "Fuel pressure": "fuelPressure",
    "Coolant pressure": "coolantPressure",
    "lub oil temp": "lubOilTemp",
    "Coolant temp": "coolantTemp",
    "Engine Condition": "engineCondition" 
}

df.rename(columns=field_map, inplace=True)

# Function to calculate viscosity from temperature
def calculate_viscosity(temp_celsius):
    temperature_kelvin = temp_celsius + 273  # Convert °C to K
    return 0.7 * math.exp(1500 / temperature_kelvin)  # Empirical formula

# Compute viscosity for training data
df["viscosity"] = df["lubOilTemp"].apply(calculate_viscosity)

# Train K-Means model on viscosity for multi-class classification
viscosity_values = df["viscosity"].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters: Low, Moderate, High
kmeans.fit(viscosity_values)

# Add viscosity class to the dataframe
df["viscosity_class"] = kmeans.predict(viscosity_values)

# Save trained K-Means model
joblib.dump(kmeans, "viscosity_classifier.joblib")
print("K-Means viscosity classifier trained and saved.")

# Train individual parameter classifiers for multi-class classification
# Define parameters to classify
parameters = ["lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]
parameter_models = {}

# Function to categorize parameters into multiple classes
def categorize_parameter(param_name, value):
    """Categorize parameters into three classes: 0 (Bad), 1 (Moderate), 2 (Good)"""
    # Define thresholds for each parameter (these should be adjusted based on domain knowledge)
    thresholds = {
        "lubOilPressure": [(0, 30, 0), (30, 60, 1), (60, float('inf'), 2)],
        "fuelPressure": [(0, 2, 0), (2, 4, 1), (4, float('inf'), 2)],
        "coolantPressure": [(0, 1, 0), (1, 3, 1), (3, float('inf'), 2)],
        "lubOilTemp": [(0, 50, 0), (50, 90, 2), (90, float('inf'), 1)],  # Middle range is better for temp
        "coolantTemp": [(0, 60, 0), (60, 95, 2), (95, float('inf'), 1)],  # Middle range is better for temp
    }
    
    for lower, upper, category in thresholds[param_name]:
        if lower <= value < upper:
            return category
    return 1  # Default to moderate if no match

# Create labeled data for each parameter
for param in parameters:
    df[f"{param}_class"] = df[param].apply(lambda x: categorize_parameter(param, x))
    
    # Train a RandomForest classifier for each parameter
    X_param = df[["engineRpm", "lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]]
    y_param = df[f"{param}_class"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_param, y_param, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{param} Classification Model Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, f"{param}_classifier.joblib")
    parameter_models[param] = model
    print(f"{param} model saved.")

# Define features & target for overall engine condition classification
X = df[["engineRpm", "lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]]
y = df["engineCondition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest for engine condition (already multi-class)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Engine Condition Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# Save trained RandomForest model
joblib.dump(model, "engine_condition_classifier.joblib")
print("Engine condition model saved.")

# Optional: Save all models together in a dictionary
all_models = {
    "engine_condition": model,
    "viscosity": kmeans,
    **parameter_models
}
joblib.dump(all_models, "all_classifiers.joblib")
print("All models saved in a single file.")