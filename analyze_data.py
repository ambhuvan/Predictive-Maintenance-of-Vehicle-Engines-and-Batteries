import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def calculate_viscosity(temp_celsius):
    """Calculate viscosity from temperature using empirical formula."""
    temperature_kelvin = temp_celsius + 273  # Convert °C to K
    return 0.7 * math.exp(1500 / temperature_kelvin)

def analyze_engine_data(data_path="engines_dataset-train_multi_class.csv"):
    print(f"Loading data from {data_path} for analysis...")
    df = pd.read_csv(data_path)

    # Ensure consistent column names (same as in prepare_multi_class_data.py)
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

    # Compute viscosity if not already in the CSV (should be if prepare_multi_class_data.py ran)
    if 'viscosity' not in df.columns:
        df["viscosity"] = df["lubOilTemp"].apply(calculate_viscosity)

    # Map numerical conditions to names for better plot labels
    condition_map = {0: 'Bad', 1: 'Good', 2: 'Moderate'}
    df['engineCondition_label'] = df['engineCondition'].map(condition_map)

    print("\n--- Value Counts by Engine Condition ---")
    print(df['engineCondition_label'].value_counts())

    print("\n--- Plotting Distributions by Engine Condition ---")
    sensor_features = ['engineRpm', 'lubOilPressure', 'fuelPressure',
                       'coolantPressure', 'lubOilTemp', 'coolantTemp', 'viscosity']

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(sensor_features):
        plt.subplot(3, 3, i + 1) # Adjust subplot grid as needed
        sns.boxplot(x='engineCondition_label', y=feature, data=df, order=['Bad', 'Moderate', 'Good'])
        plt.title(f'{feature} by Engine Condition')
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete. Examine the plots to refine your 'Moderate' rules.")
    print("Look for where 'Moderate' overlaps with 'Good' or 'Bad' and adjust thresholds.")

if __name__ == "__main__":
    analyze_engine_data()