"""
Usage Example for Optimized Engine Condition Prediction Models

This script demonstrates how to use the optimized models for engine condition prediction.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer

def load_optimized_model():
    """Load the best optimized model and preprocessor"""
    try:
        model = joblib.load('ultimate_optimized_model.joblib')
        preprocessor = joblib.load('ultimate_preprocessor.joblib')
        print("✓ Optimized LightGBM model loaded successfully")
        return model, preprocessor
    except FileNotFoundError:
        print("✗ Optimized model files not found. Please run ultimate_model.py first.")
        return None, None

def create_advanced_features(df):
    """Create the same advanced features used in training"""
    df_advanced = df.copy()
    
    # Remove viscosity if present
    if 'viscosity' in df_advanced.columns:
        df_advanced = df_advanced.drop('viscosity', axis=1)
    
    # Top 3 separable features based on analysis
    top_features = ['coolantTemp', 'engineRpm', 'fuelPressure']
    
    # Advanced transformations
    for feature in top_features:
        df_advanced[f'{feature}_log'] = np.log1p(df_advanced[feature])
        df_advanced[f'{feature}_sqrt'] = np.sqrt(df_advanced[feature])
        df_advanced[f'{feature}_square'] = df_advanced[feature] ** 2
        
        # Binning (simplified to avoid edge cases with single values)
        try:
            quartiles = df_advanced[feature].quantile([0.25, 0.5, 0.75])
            if len(quartiles.unique()) > 1:  # Only bin if we have different quartile values
                df_advanced[f'{feature}_bin'] = pd.cut(df_advanced[feature], 
                                                      bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                                                      labels=[0, 1, 2, 3], duplicates='drop')
            else:
                df_advanced[f'{feature}_bin'] = 1  # Default bin for single values
        except:
            df_advanced[f'{feature}_bin'] = 1  # Default bin if binning fails
    
    # Ratios and interactions
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            df_advanced[f'{feat1}_{feat2}_ratio'] = df_advanced[feat1] / (df_advanced[feat2] + 1e-8)
            df_advanced[f'{feat1}_{feat2}_diff'] = df_advanced[feat1] - df_advanced[feat2]
            df_advanced[f'{feat1}_{feat2}_product'] = df_advanced[feat1] * df_advanced[feat2]
    
    # Distance features (using training set statistics)
    # Note: In production, these should be calculated from training data
    training_stats = {
        'coolantTemp': {'class_0_mean': 80.448, 'class_1_mean': 80.011, 'class_2_mean': 83.588},
        'engineRpm': {'class_0_mean': 884.468, 'class_1_mean': 734.663, 'class_2_mean': 800.032},
        'fuelPressure': {'class_0_mean': 6.288, 'class_1_mean': 6.929, 'class_2_mean': 6.702}
    }
    
    for cls in [0, 1, 2]:
        for feature in top_features:
            class_mean = training_stats[feature][f'class_{cls}_mean']
            df_advanced[f'dist_to_class_{cls}_{feature}'] = np.abs(df_advanced[feature] - class_mean)
    
    return df_advanced

def predict_engine_condition(model, preprocessor, sensor_data):
    """
    Predict engine condition from sensor data
    
    Parameters:
    - model: trained model
    - preprocessor: fitted preprocessor
    - sensor_data: dict with sensor readings
    
    Returns:
    - prediction: 0 (Bad), 1 (Good), 2 (Moderate)
    - probabilities: probability for each class
    """
    
    # Create DataFrame from sensor data
    df = pd.DataFrame([sensor_data])
    
    # Map column names if needed
    field_map = {
        "Engine rpm": "engineRpm",
        "Lub oil pressure": "lubOilPressure", 
        "Fuel pressure": "fuelPressure",
        "Coolant pressure": "coolantPressure",
        "lub oil temp": "lubOilTemp",
        "Coolant temp": "coolantTemp"
    }
    df.rename(columns=field_map, inplace=True)
    
    # Create advanced features
    df_advanced = create_advanced_features(df)
    
    # Preprocess
    X_processed = preprocessor.transform(df_advanced)
    
    # Predict
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    
    # Map prediction to meaningful labels
    condition_map = {0: 'Bad', 1: 'Good', 2: 'Moderate'}
    condition_label = condition_map[prediction]
    
    return prediction, condition_label, probabilities

def demonstrate_usage():
    """Demonstrate usage with example data"""
    print("=== ENGINE CONDITION PREDICTION DEMO ===")
    
    # Load model
    model, preprocessor = load_optimized_model()
    if model is None:
        return
    
    # Example sensor readings
    examples = [
        {
            'name': 'Engine in Good Condition',
            'data': {
                'engineRpm': 750,
                'lubOilPressure': 3.5,
                'fuelPressure': 7.0,
                'coolantPressure': 2.3,
                'lubOilTemp': 79.0,
                'coolantTemp': 78.0
            }
        },
        {
            'name': 'Engine in Bad Condition', 
            'data': {
                'engineRpm': 900,
                'lubOilPressure': 3.0,
                'fuelPressure': 6.0,
                'coolantPressure': 2.5,
                'lubOilTemp': 82.0,
                'coolantTemp': 85.0
            }
        },
        {
            'name': 'Engine in Moderate Condition',
            'data': {
                'engineRpm': 800,
                'lubOilPressure': 3.2,
                'fuelPressure': 6.7,
                'coolantPressure': 2.3,
                'lubOilTemp': 80.0,
                'coolantTemp': 84.0
            }
        }
    ]
    
    # Make predictions
    for example in examples:
        print(f"\n--- {example['name']} ---")
        print("Sensor readings:")
        for sensor, value in example['data'].items():
            print(f"  {sensor}: {value}")
        
        # Predict
        prediction, condition_label, probabilities = predict_engine_condition(
            model, preprocessor, example['data']
        )
        
        print(f"\nPrediction: {condition_label} (Class {prediction})")
        print("Probabilities:")
        print(f"  Bad: {probabilities[0]:.3f}")
        print(f"  Good: {probabilities[1]:.3f}")
        print(f"  Moderate: {probabilities[2]:.3f}")
        
        # Confidence assessment
        confidence = max(probabilities)
        if confidence > 0.7:
            confidence_level = "High"
        elif confidence > 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        print(f"Confidence: {confidence_level} ({confidence:.3f})")

def batch_prediction_example():
    """Example of batch prediction from CSV file"""
    print("\n=== BATCH PREDICTION EXAMPLE ===")
    
    # Load model
    model, preprocessor = load_optimized_model()
    if model is None:
        return
    
    try:
        # Load test data
        df_test = pd.read_csv("engines_dataset-test.csv")
        print(f"Loaded test data: {df_test.shape}")
        
        # Create advanced features
        df_advanced = create_advanced_features(df_test)
        
        # Preprocess
        X_processed = preprocessor.transform(df_advanced)
        
        # Predict
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        # Add results to dataframe
        condition_map = {0: 'Bad', 1: 'Good', 2: 'Moderate'}
        df_test['predicted_condition'] = [condition_map[p] for p in predictions]
        df_test['prediction_confidence'] = np.max(probabilities, axis=1)
        
        # Save results
        df_test.to_csv('engine_predictions.csv', index=False)
        print("✓ Predictions saved to 'engine_predictions.csv'")
        
        # Summary
        print(f"\nPrediction Summary:")
        print(df_test['predicted_condition'].value_counts())
        print(f"Average confidence: {df_test['prediction_confidence'].mean():.3f}")
        
    except FileNotFoundError:
        print("✗ Test data file 'engines_dataset-test.csv' not found")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_usage()
    
    # Run batch prediction example
    batch_prediction_example()
    
    print("\n=== USAGE NOTES ===")
    print("1. Ensure optimized model files are present (run ultimate_model.py first)")
    print("2. Input sensor data as dictionary with exact sensor names")
    print("3. Model returns prediction (0/1/2) and probabilities for each class")
    print("4. Use prediction confidence to determine if human review is needed")
    print("5. For production use, consider implementing monitoring and retraining pipeline")