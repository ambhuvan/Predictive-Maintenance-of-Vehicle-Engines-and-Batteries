import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

class OptimizedEnginePredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None
        
    def load_data(self, data_path="engines_dataset-train_multi_class.csv"):
        """Load and prepare the dataset"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Check and map column names if needed
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
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df['engineCondition'].value_counts(normalize=True).round(3)}")
        
        return df
    
    def engineer_features(self, df):
        """Create advanced features from the raw sensor data"""
        print("Engineering features...")
        
        # Create a copy to avoid modifying original
        df_eng = df.copy()
        
        # Basic engineered features
        df_eng['pressure_ratio'] = df_eng['lubOilPressure'] / (df_eng['coolantPressure'] + 1e-8)
        df_eng['temp_difference'] = df_eng['lubOilTemp'] - df_eng['coolantTemp']
        df_eng['fuel_efficiency'] = df_eng['fuelPressure'] / (df_eng['engineRpm'] + 1)
        df_eng['thermal_load'] = df_eng['lubOilTemp'] * df_eng['coolantTemp']
        df_eng['pressure_sum'] = df_eng['lubOilPressure'] + df_eng['coolantPressure']
        df_eng['temp_avg'] = (df_eng['lubOilTemp'] + df_eng['coolantTemp']) / 2
        
        # Advanced ratio features
        df_eng['lub_fuel_ratio'] = df_eng['lubOilPressure'] / (df_eng['fuelPressure'] + 1e-8)
        df_eng['rpm_pressure_ratio'] = df_eng['engineRpm'] / (df_eng['lubOilPressure'] + 1)
        df_eng['cooling_efficiency'] = df_eng['coolantPressure'] / (df_eng['coolantTemp'] + 1)
        
        # Viscosity calculation (from existing code)
        df_eng['viscosity'] = df_eng['lubOilTemp'].apply(lambda temp: 0.7 * np.exp(1500 / (temp + 273)))
        
        # Statistical features based on all numeric columns
        numeric_cols = ['engineRpm', 'lubOilPressure', 'fuelPressure', 'coolantPressure', 'lubOilTemp', 'coolantTemp']
        
        # Z-scores for outlier detection
        for col in numeric_cols:
            df_eng[f'{col}_zscore'] = np.abs((df_eng[col] - df_eng[col].mean()) / df_eng[col].std())
        
        # Feature interactions using polynomial features (degree 2, interaction only)
        from itertools import combinations
        for col1, col2 in combinations(numeric_cols, 2):
            df_eng[f'{col1}_{col2}_interaction'] = df_eng[col1] * df_eng[col2]
        
        print(f"Feature engineering complete. New shape: {df_eng.shape}")
        return df_eng
    
    def prepare_data(self, df, target_column='engineCondition'):
        """Prepare features and target for modeling"""
        # Remove target and any non-feature columns
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def handle_class_imbalance(self, X_train, y_train, strategy='smote_tomek'):
        """Apply various class balancing techniques"""
        print(f"Applying class balancing strategy: {strategy}")
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=self.random_state)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            return X_train, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"Resampled data shape: {X_resampled.shape}")
        print(f"Resampled target distribution:\n{pd.Series(y_resampled).value_counts(normalize=True).round(3)}")
        
        return X_resampled, y_resampled
    
    def feature_selection(self, X_train, y_train, method='rfe', n_features=20):
        """Select best features"""
        print(f"Selecting top {n_features} features using {method}...")
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        else:
            return X_train, None
        
        X_selected = selector.fit_transform(X_train, y_train)
        
        if hasattr(selector, 'get_support'):
            selected_features = [self.feature_names[i] for i, selected in enumerate(selector.get_support()) if selected]
            print(f"Selected features: {selected_features}")
        
        return X_selected, selector
    
    def train_advanced_models(self, X_train, y_train, X_val, y_val):
        """Train multiple advanced models"""
        print("Training advanced models...")
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            objective='multi:softprob'
        )
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     verbose=False)
        
        # LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            objective='multiclass',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        # CatBoost
        print("Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=False
        )
        cb_model.fit(X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False)
        
        # Enhanced Random Forest
        print("Training Enhanced Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Enhanced Gradient Boosting
        print("Training Enhanced Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        gb_model.fit(X_train, y_train)
        
        self.models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model, 
            'CatBoost': cb_model,
            'RandomForest': rf_model,
            'GradientBoosting': gb_model
        }
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("Evaluating models...")
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1_Macro': f1_macro,
                'F1_Weighted': f1_weighted
            })
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Bad', 'Good', 'Moderate']))
        
        results_df = pd.DataFrame(results)
        print(f"\nModel Comparison:")
        print(results_df.sort_values('Accuracy', ascending=False))
        
        return results_df
    
    def create_ensemble(self, X_train, y_train):
        """Create ensemble model from best performers"""
        print("Creating ensemble model...")
        
        # Select top 3 models based on performance
        estimators = [
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('cb', self.models['CatBoost'])
        ]
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        self.ensemble_model.fit(X_train, y_train)
        print("Ensemble model created successfully!")
        
        return self.ensemble_model
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance from tree-based models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        tree_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
        
        for idx, model_name in enumerate(tree_models):
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:top_n]
                    
                    axes[idx].barh(range(top_n), importances[indices][::-1])
                    if self.feature_names:
                        feature_labels = [self.feature_names[i] for i in indices][::-1]
                        axes[idx].set_yticks(range(top_n))
                        axes[idx].set_yticklabels(feature_labels)
                    axes[idx].set_title(f'{model_name} Feature Importance')
                    axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('optimized_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plots saved as 'optimized_feature_importance.png'")
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models"""
        n_models = len(self.models) + (1 if self.ensemble_model else 0)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot individual models
        for name, model in self.models.items():
            row, col = plot_idx // cols, plot_idx % cols
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Bad', 'Good', 'Moderate'],
                       yticklabels=['Bad', 'Good', 'Moderate'],
                       ax=axes[row, col])
            axes[row, col].set_title(f'{name}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')
            plot_idx += 1
        
        # Plot ensemble if available
        if self.ensemble_model and plot_idx < rows * cols:
            row, col = plot_idx // cols, plot_idx % cols
            y_pred = self.ensemble_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=['Bad', 'Good', 'Moderate'],
                       yticklabels=['Bad', 'Good', 'Moderate'],
                       ax=axes[row, col])
            axes[row, col].set_title('Ensemble Model')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('optimized_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices saved as 'optimized_confusion_matrices.png'")
    
    def save_models(self, prefix='optimized'):
        """Save all trained models"""
        print("Saving models...")
        
        # Save individual models
        for name, model in self.models.items():
            filename = f'{prefix}_{name.lower()}_model.joblib'
            joblib.dump(model, filename)
            print(f"Saved {name} model as {filename}")
        
        # Save ensemble model
        if self.ensemble_model:
            filename = f'{prefix}_ensemble_model.joblib'
            joblib.dump(self.ensemble_model, filename)
            print(f"Saved ensemble model as {filename}")
        
        # Save preprocessors
        if self.scaler:
            joblib.dump(self.scaler, f'{prefix}_scaler.joblib')
        if self.feature_selector:
            joblib.dump(self.feature_selector, f'{prefix}_feature_selector.joblib')
        
        print("All models saved successfully!")
    
    def run_optimization(self, data_path="engines_dataset-train_multi_class.csv"):
        """Run the complete optimization pipeline"""
        print("Starting model optimization pipeline...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Engineer features
        df_eng = self.engineer_features(df)
        
        # Prepare data
        X, y = self.prepare_data(df_eng)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Further split train into train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Scale features
        print("Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected, self.feature_selector = self.feature_selection(
            X_train_scaled, y_train_split, method='rfe', n_features=25
        )
        X_val_selected = self.feature_selector.transform(X_val_scaled) if self.feature_selector else X_val_scaled
        X_test_selected = self.feature_selector.transform(X_test_scaled) if self.feature_selector else X_test_scaled
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train_selected, y_train_split, strategy='smote_tomek'
        )
        
        # Train models
        self.train_advanced_models(X_train_balanced, y_train_balanced, X_val_selected, y_val)
        
        # Evaluate models
        results_df = self.evaluate_models(X_test_selected, y_test)
        
        # Create ensemble
        self.create_ensemble(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test_selected)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='macro')
            
            print(f"\nEnsemble Model Results:")
            print(f"Accuracy: {ensemble_accuracy:.4f}")
            print(f"F1-Macro: {ensemble_f1:.4f}")
            print(classification_report(y_test, y_pred_ensemble, target_names=['Bad', 'Good', 'Moderate']))
        
        # Generate visualizations
        self.plot_feature_importance()
        self.plot_confusion_matrices(X_test_selected, y_test)
        
        # Save models
        self.save_models()
        
        print("\nOptimization pipeline completed successfully!")
        return results_df, ensemble_accuracy if self.ensemble_model else None

if __name__ == "__main__":
    # Run the optimization
    optimizer = OptimizedEnginePredictor(random_state=42)
    results, ensemble_accuracy = optimizer.run_optimization()
    
    print(f"\nFinal Results Summary:")
    print(f"Best individual model accuracy: {results['Accuracy'].max():.4f}")
    if ensemble_accuracy:
        print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
        print(f"Improvement over baseline (~65%): {ensemble_accuracy - 0.65:.4f}")