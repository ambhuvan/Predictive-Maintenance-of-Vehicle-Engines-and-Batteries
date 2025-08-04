import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use ImbPipeline for SMOTE in pipeline
import joblib # For saving/loading models
import time
import os

# --- Configuration ---
# IMPORTANT: Make sure this DATA_PATH points to the correct CSV file
# Based on your previous output, it should be 'engines_dataset-train_multi_class.csv'
DATA_PATH = 'engines_dataset-train_multi_class.csv' # CONFIRM THIS IS YOUR FILE NAME
TARGET_COLUMN = 'engineCondition'
RANDOM_STATE = 42 # for reproducibility

# --- Data Loading and Initial Preparation ---
def load_data(path):
    df = pd.read_csv(path)
    # The 'engineCondition' column is already numerical (0, 1, 2)
    # So, no mapping is needed here. Removed the .map() call.

    # Handle NaN values in the target column (still good to keep this general check)
    if df['engineCondition'].isnull().any():
        nan_count = df['engineCondition'].isnull().sum()
        original_rows = len(df)
        df.dropna(subset=['engineCondition'], inplace=True) # Drops rows where 'engineCondition' is NaN
        print(f"Warning: {nan_count} rows had NaN in 'engineCondition' and were dropped.")
        print(f"Remaining rows: {len(df)} out of {original_rows}")

    return df

def prepare_data(df, target_column):
    # Check if 'timestamp' column exists before dropping it
    columns_to_drop = [target_column]
    if 'timestamp' in df.columns:
        columns_to_drop.append('timestamp')
    elif 'Timestamp' in df.columns: # Sometimes it's capitalized
        columns_to_drop.append('Timestamp')
    # Add any other columns you explicitly know are NOT features and should be dropped
    # For example, if you have an 'id' column:
    # if 'id' in df.columns:
    #     columns_to_drop.append('id')

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]
    return X, y

# --- Main Execution ---
def main():
    print("Loading and preparing the dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X, y = prepare_data(df, TARGET_COLUMN)

    # Split data into training and testing sets BEFORE SMOTE for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    # Display original target distribution
    print("\nTarget distribution (Original):")
    print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    # --- Apply SMOTE for initial model comparison (on training data only) ---
    print("\nApplying SMOTE to the training data for initial comparison...")
    scaler_smote_initial = StandardScaler()
    X_train_scaled_initial = scaler_smote_initial.fit_transform(X_train)
    smote_initial = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled_initial, y_train_resampled_initial = smote_initial.fit_resample(X_train_scaled_initial, y_train)

    print("Target distribution (After SMOTE on Training Data for initial comparison):")
    print(y_train_resampled_initial.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    # --- Initial Model Comparison ---
    print("\nModel Comparison (on manually SMOTE-resampled training data):")
    models = {
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(random_state=RANDOM_STATE), # Corrected 'RANDM_STATE' to 'RANDOM_STATE' here
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Neural Network': MLPClassifier(random_state=RANDOM_STATE, max_iter=500) # Increased max_iter
    }

    results = []
    for name, model in models.items():
        start_time = time.time()
        # Scale test data for initial evaluation
        X_test_scaled_initial = scaler_smote_initial.transform(X_test)

        # Train on SMOTE-resampled training data
        model.fit(X_train_resampled_initial, y_train_resampled_initial)
        train_time = time.time() - start_time

        # Predict on original (unbalanced) scaled test data
        start_pred_time = time.time()
        y_pred = model.predict(X_test_scaled_initial)
        pred_time = time.time() - start_pred_time

        accuracy = accuracy_score(y_test, y_pred)
        # Using f1_weighted for cross_val_score in initial comparison
        # since it's a good general metric
        cv_f1 = np.mean(cross_val_score(model, X_train_resampled_initial, y_train_resampled_initial, cv=5, scoring='f1_weighted', n_jobs=-1))

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Cross-Val (F1)': cv_f1,
            'Train Time': train_time,
            'Pred Time': pred_time
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    best_initial_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\nBest model (based on Accuracy, trained on manually SMOTE data): {best_initial_model['Model']} with accuracy: {best_initial_model['Accuracy']:.4f}")

    print("\nDetailed Performance by Class (Initial Models trained on manually SMOTE data):\n")
    # Generate and print classification reports for initial models
    for name, model in models.items():
        print(f"{name} Performance:")
        X_test_scaled_initial = scaler_smote_initial.transform(X_test)
        y_pred = model.predict(X_test_scaled_initial)
        # Map numerical labels back for better readability in reports if needed
        # For now, keeping as 0, 1, 2. If you want 'Bad', 'Good', 'Moderate',
        # you'll need to create a mapping for the report.
        report = classification_report(y_test, y_pred, target_names=['Bad', 'Good', 'Moderate'], zero_division=0)
        print(report)
        print("-" * 50)


    # --- Plot Confusion Matrices for Initial Models ---
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(models.items()):
        plt.subplot(2, 3, i + 1)
        X_test_scaled_initial = scaler_smote_initial.transform(X_test)
        y_pred = model.predict(X_test_scaled_initial)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bad', 'Good', 'Moderate'],
                    yticklabels=['Bad', 'Good', 'Moderate'])
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('model_confusion_matrices.png')
    plt.close()
    print("Confusion matrices saved as 'model_confusion_matrices.png'")

    # --- Plot Feature Importance for Ensemble Models ---
    plt.figure(figsize=(10, 6))
    # Random Forest Feature Importance
    if 'Random Forest' in models and hasattr(models['Random Forest'], 'feature_importances_'):
        rf_model = models['Random Forest']
        importances_rf = rf_model.feature_importances_
        indices_rf = np.argsort(importances_rf)[::-1]
        plt.bar(range(X_train.shape[1]), importances_rf[indices_rf], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices_rf], rotation=90)
        plt.title("Random Forest Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig('feature_importance_rf.png') # Changed filename to specify RF
        plt.close()
        print("Feature importance plot saved as 'feature_importance_rf.png'")

    # Gradient Boosting Feature Importance
    if 'Gradient Boosting' in models and hasattr(models['Gradient Boosting'], 'feature_importances_'):
        gb_model = models['Gradient Boosting']
        importances_gb = gb_model.feature_importances_
        indices_gb = np.argsort(importances_gb)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(X_train.shape[1]), importances_gb[indices_gb], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices_gb], rotation=90)
        plt.title("Gradient Boosting Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig('feature_importance_gb.png') # Changed filename to specify GB
        plt.close()
        print("Feature importance plot saved as 'feature_importance_gb.png'")


    print("\n================================================================================")
    print("--- Initiating Hyperparameter Tuning for Top Models (with SMOTE and Scaling in Pipeline) ---")
    print("================================================================================")

    # Define the preprocessing steps within a pipeline
    # SMOTE should be applied ONLY to the training data in each CV fold
    # This is correctly handled by ImbPipeline
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE))
    ]

    # --- Tuning Gradient Boosting Classifier ---
    print("\n--- Tuning Gradient Boosting Classifier (this might take a while)...")
    pipeline_gb = ImbPipeline(pipeline_steps + [('model', GradientBoostingClassifier(random_state=RANDOM_STATE))])

    param_grid_gb = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__max_depth': [3, 4, 5]
    }

    # IMPORTANT CHANGE: Scoring changed to 'f1_macro'
    grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search_gb.fit(X_train, y_train) # Use original X_train, y_train here!

    print("\nBest parameters for Gradient Boosting:", grid_search_gb.best_params_)
    print("Best cross-validation F1-score (macro):", grid_search_gb.best_score_)

    best_gb_model = grid_search_gb.best_estimator_
    y_pred_gb_tuned = best_gb_model.predict(X_test)
    accuracy_gb_tuned = accuracy_score(y_test, y_pred_gb_tuned)
    print(f"\nGradient Boosting (Tuned) Test Accuracy: {accuracy_gb_tuned:.4f}")
    print("Classification Report (Tuned Gradient Boosting):")
    print(classification_report(y_test, y_pred_gb_tuned, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))


    # --- Tuning Random Forest Classifier ---
    print("\n--- Tuning Random Forest Classifier (this might take a while)...")
    pipeline_rf = ImbPipeline(pipeline_steps + [('model', RandomForestClassifier(random_state=RANDOM_STATE))])

    param_grid_rf = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, None], # None means unlimited depth
        'model__min_samples_split': [2, 5]
    }

    # IMPORTANT CHANGE: Scoring changed to 'f1_macro'
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train) # Use original X_train, y_train here!

    print("\nBest parameters for Random Forest:", grid_search_rf.best_params_)
    print("Best cross-validation F1-score (macro):", grid_search_rf.best_score_)

    best_rf_model = grid_search_rf.best_estimator_
    y_pred_rf_tuned = best_rf_model.predict(X_test)
    accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
    print(f"\nRandom Forest (Tuned) Test Accuracy: {accuracy_rf_tuned:.4f}")
    print("Classification Report (Tuned Random Forest):")
    print(classification_report(y_test, y_pred_rf_tuned, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))

    # --- Select and Save the Overall Best Tuned Model ---
    # Compare best models based on f1_macro score from GridSearchCV
    if grid_search_gb.best_score_ > grid_search_rf.best_score_:
        overall_best_model = best_gb_model
        overall_best_accuracy = accuracy_gb_tuned
        model_name = "Gradient Boosting Tuned"
    else:
        overall_best_model = best_rf_model
        overall_best_accuracy = accuracy_rf_tuned
        model_name = "Random Forest Tuned"

    print(f"\nOverall Best Tuned Model: {model_name} with Test Accuracy: {overall_best_accuracy:.4f}")

    # Ensure the directory for saving exists
    if not os.path.exists('models'):
        os.makedirs('models')

    model_filename = 'models/best_tuned_engine_classifier_with_smote_pipeline.joblib'
    joblib.dump(overall_best_model, model_filename)
    print(f"Best tuned model (with SMOTE and Scaler in pipeline) saved as '{model_filename}'")


    # --- Generate Confusion Matrices for Tuned Models ---
    print("\nGenerating Confusion Matrices for Tuned Models...")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    cm_gb_tuned = confusion_matrix(y_test, y_pred_gb_tuned)
    sns.heatmap(cm_gb_tuned, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good', 'Moderate'],
                yticklabels=['Bad', 'Good', 'Moderate'])
    plt.title('Tuned Gradient Boosting Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)
    sns.heatmap(cm_rf_tuned, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good', 'Moderate'],
                yticklabels=['Bad', 'Good', 'Moderate'])
    plt.title('Tuned Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.savefig('tuned_model_confusion_matrices.png') # Changed filename to specify tuned
    plt.close()
    print("Confusion matrices saved as 'tuned_model_confusion_matrices.png'")

    # You might want to plot feature importances for the *tuned* best models as well.
    # This requires accessing the 'model' step of the best_estimator_ pipeline.
    # Example for Random Forest:
    # if hasattr(overall_best_model.named_steps['model'], 'feature_importances_'):
    #     tuned_rf_importances = overall_best_model.named_steps['model'].feature_importances_
    #     # ... (plotting logic similar to above)


if __name__ == "__main__":
    main()