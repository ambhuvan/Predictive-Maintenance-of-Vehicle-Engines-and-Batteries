import pandas as pd

# Make sure this path is correct for your file
DATA_PATH = 'engines_dataset-train_multi_class.csv'
TARGET_COLUMN = 'engineCondition'

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded {DATA_PATH}. Shape: {df.shape}")

    if TARGET_COLUMN in df.columns:
        print(f"\nUnique values and their counts in '{TARGET_COLUMN}' column:")
        print(df[TARGET_COLUMN].value_counts(dropna=False)) # dropna=False includes NaNs if any
        print("\nData types of all columns:")
        print(df.dtypes)
    else:
        print(f"\nError: Column '{TARGET_COLUMN}' not found in the CSV.")
        print("Available columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")