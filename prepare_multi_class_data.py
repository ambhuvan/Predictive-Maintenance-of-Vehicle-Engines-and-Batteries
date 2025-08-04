import pandas as pd
import math

def calculate_viscosity(temp_celsius):
    """Calculate viscosity from temperature using empirical formula."""
    temperature_kelvin = temp_celsius + 273  # Convert °C to K
    return 0.7 * math.exp(1500 / temperature_kelvin)

def prepare_multi_class_data(input_csv_path="engines_dataset-train.csv", output_csv_path="engines_dataset-train_multi_class.csv"):
    """
    Loads the engine dataset, computes viscosity, and re-labels a portion of 'Good' (1)
    and 'Bad' (0) records as 'Moderate' (2) based on specific sensor value criteria.
    Saves the modified data to a new CSV.
    """
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

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

    # Compute viscosity
    df["viscosity"] = df["lubOilTemp"].apply(calculate_viscosity)

    original_counts = df['engineCondition'].value_counts()
    print(f"Original engineCondition distribution:\n{original_counts}")

    # --- Define new rules for 'Moderate' (class 2) based on plot analysis ---
    # We are trying to make the 'Moderate' band more distinct,
    # and potentially pull fewer 'Bad' records into 'Moderate'.

    # Re-label 'Good' (1) records that are "slightly off" as 'Moderate' (2)
    # Adjusted lubOilPressure and coolantTemp thresholds based on image_f05d05.jpg
    moderate_conditions_from_good = (
        (df['engineCondition'] == 1) &
        (df['lubOilPressure'] < 4.0) & (df['lubOilPressure'] >= 3.0) & # From plot: Good is >3.5-4.0, Moderate 2.5-4.0
        (df['coolantTemp'] > 80.0) & (df['coolantTemp'] <= 85.0)     # From plot: Good is <80.0, Moderate 80-87.5
    )

    # Re-label 'Bad' (0) records that are "not quite critical" as 'Moderate' (2)
    # Adjusted lubOilPressure and coolantTemp thresholds based on image_f05d05.jpg
    moderate_conditions_from_bad = (
        (df['engineCondition'] == 0) &
        (df['lubOilPressure'] >= 2.0) & (df['lubOilPressure'] < 3.0) & # From plot: Bad is <2.5, Moderate 2.5-4.0. Making this stricter.
        (df['coolantTemp'] > 85.0) & (df['coolantTemp'] <= 88.0)      # From plot: Bad is >87.5, Moderate 80-87.5. Making this stricter.
    )

    # Apply re-labeling
    df.loc[moderate_conditions_from_good, 'engineCondition'] = 2
    df.loc[moderate_conditions_from_bad, 'engineCondition'] = 2

    final_counts = df['engineCondition'].value_counts()
    print(f"\nModified engineCondition distribution:\n{final_counts}")

    # Save the modified DataFrame to a new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nModified dataset saved to {output_csv_path}")

    # Instruction for next steps (already done these, but for reminder)
    print("\n--- Next Steps (Re-confirming for your next actions) ---")
    print(f"1. **Confirm `data_path` in `model.py` is `\"{output_csv_path}\"`**")
    print(f"2. **Run `model.py`:** `python model.py`")
    print(f"3. **Confirm `data_path` in `model_comparison.py` is `\"{output_csv_path}\"`**")
    print(f"4. **Run `model_comparison.py`:** `python model_comparison.py`")
    print("\nAfter these runs, check the 'Engine Condition Model' accuracy and recall/precision for 'Bad' (0) again!")

if __name__ == "__main__":
    prepare_multi_class_data()