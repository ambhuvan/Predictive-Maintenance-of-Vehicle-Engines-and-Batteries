import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None
        
    def load_data(self, data_path="engines_dataset-train_multi_class.csv"):
        """Load and prepare data"""
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Map column names if needed
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
        print(f"Class distribution:\n{df['engineCondition'].value_counts(normalize=True).round(4)}")
        
        return df
    
    def create_optimized_features(self, df):
        """Create optimized feature set based on analysis"""
        print("Creating optimized features...")
        
        df_opt = df.copy()
        
        # Remove highly correlated viscosity feature if it exists
        if 'viscosity' in df_opt.columns:
            df_opt = df_opt.drop('viscosity', axis=1)
        
        # Focus on most discriminative features: engineRpm, coolantTemp, fuelPressure
        
        # Engine performance ratios (engineRpm is most discriminative)
        df_opt['rpm_fuel_ratio'] = df_opt['engineRpm'] / (df_opt['fuelPressure'] + 1)
        df_opt['rpm_temp_ratio'] = df_opt['engineRpm'] / (df_opt['lubOilTemp'] + 1)
        df_opt['rpm_normalized'] = (df_opt['engineRpm'] - df_opt['engineRpm'].mean()) / df_opt['engineRpm'].std()
        
        # Thermal features (coolantTemp is 2nd most discriminative) 
        df_opt['temp_difference'] = df_opt['coolantTemp'] - df_opt['lubOilTemp']
        df_opt['temp_ratio'] = df_opt['coolantTemp'] / (df_opt['lubOilTemp'] + 1)
        df_opt['coolant_normalized'] = (df_opt['coolantTemp'] - df_opt['coolantTemp'].mean()) / df_opt['coolantTemp'].std()
        
        # Fuel system features (fuelPressure is 3rd most discriminative)
        df_opt['fuel_efficiency'] = df_opt['fuelPressure'] / df_opt['engineRpm']
        df_opt['fuel_normalized'] = (df_opt['fuelPressure'] - df_opt['fuelPressure'].mean()) / df_opt['fuelPressure'].std()
        
        # Pressure interactions
        df_opt['pressure_ratio'] = df_opt['lubOilPressure'] / (df_opt['coolantPressure'] + 1e-8)
        df_opt['total_pressure'] = df_opt['lubOilPressure'] + df_opt['coolantPressure'] + df_opt['fuelPressure']
        
        # Combined efficiency metrics
        df_opt['overall_efficiency'] = (df_opt['fuelPressure'] * df_opt['lubOilPressure']) / (df_opt['engineRpm'] + 1)
        df_opt['thermal_load'] = df_opt['lubOilTemp'] * df_opt['coolantTemp']
        
        # Squared and interaction terms for top features
        df_opt['engineRpm_squared'] = df_opt['engineRpm'] ** 2
        df_opt['coolantTemp_squared'] = df_opt['coolantTemp'] ** 2
        df_opt['rpm_coolant_interaction'] = df_opt['engineRpm'] * df_opt['coolantTemp']
        
        print(f"Optimized feature set created. Shape: {df_opt.shape}")
        return df_opt
    
    def prepare_data(self, df, target_col='engineCondition'):
        """Prepare features and target"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.feature_names = X.columns.tolist()
        return X, y
    
    def preprocess_features(self, X_train, X_test, y_train):
        """Preprocess features with scaling and selection"""
        print("Preprocessing features...")
        
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection - keep more features but remove least important
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_mask = self.feature_selector.get_support()
            selected_features = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
            print(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        
        return X_train_selected, X_test_selected
    
    def balance_classes(self, X_train, y_train):
        """Balance classes using BorderlineSMOTE"""
        print("Balancing classes...")
        
        # Use BorderlineSMOTE for better boundary detection
        smote = BorderlineSMOTE(random_state=self.random_state, kind='borderline-1')
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Balanced distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def train_models(self, X_train, y_train):
        """Train optimized models with class weights"""
        print("Training optimized models...")
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # 1. Optimized XGBoost
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
        
        # Set sample weights for XGBoost
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # 2. Optimized LightGBM  
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_state,
            objective='multiclass',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # 3. Optimized CatBoost
        print("Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            class_weights=list(class_weights),
            random_state=self.random_state,
            verbose=False
        )
        cb_model.fit(X_train, y_train)
        
        # 4. Optimized Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # 5. Optimized Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Apply sample weights to GB
        gb_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'CatBoost': cb_model,
            'RandomForest': rf_model,
            'GradientBoosting': gb_model
        }
        
        return self.models
    
    def create_ensemble(self, X_train, y_train):
        """Create voting ensemble of best models"""
        print("Creating ensemble model...")
        
        # Use top 3 models for ensemble
        estimators = [
            ('lgb', self.models['LightGBM']),
            ('cb', self.models['CatBoost']),
            ('xgb', self.models['XGBoost'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        self.ensemble_model.fit(X_train, y_train)
        print("Ensemble model created!")
        
        return self.ensemble_model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("Evaluating models...")
        results = []
        
        # Evaluate individual models
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
            print(f"Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            f1_macro_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
            f1_weighted_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            results.append({
                'Model': 'Ensemble',
                'Accuracy': accuracy_ensemble,
                'F1_Macro': f1_macro_ensemble,
                'F1_Weighted': f1_weighted_ensemble
            })
            
            print(f"\nEnsemble Results:")
            print(f"Accuracy: {accuracy_ensemble:.4f}, F1-Macro: {f1_macro_ensemble:.4f}")
            print(classification_report(y_test, y_pred_ensemble, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))
        
        results_df = pd.DataFrame(results)
        print(f"\nModel Comparison:")
        print(results_df.sort_values('Accuracy', ascending=False))
        
        return results_df
    
    def plot_results(self, X_test, y_test):
        """Plot confusion matrices and feature importance"""
        
        # Confusion matrices
        n_models = len(self.models) + (1 if self.ensemble_model else 0)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Individual models
        for name, model in self.models.items():
            if plot_idx >= rows * cols:
                break
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
        
        # Ensemble
        if self.ensemble_model and plot_idx < rows * cols:
            row, col = plot_idx // cols, plot_idx % cols
            y_pred = self.ensemble_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=['Bad', 'Good', 'Moderate'],
                       yticklabels=['Bad', 'Good', 'Moderate'],
                       ax=axes[row, col])
            axes[row, col].set_title('Ensemble')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('final_optimized_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices saved as 'final_optimized_confusion_matrices.png'")
    
    def save_models(self):
        """Save all models"""
        print("Saving models...")
        
        for name, model in self.models.items():
            filename = f'final_optimized_{name.lower()}_model.joblib'
            joblib.dump(model, filename)
            print(f"Saved {name}")
        
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, 'final_optimized_ensemble_model.joblib')
            print("Saved ensemble model")
        
        # Save preprocessors
        if self.scaler:
            joblib.dump(self.scaler, 'final_optimized_scaler.joblib')
        if self.feature_selector:
            joblib.dump(self.feature_selector, 'final_optimized_feature_selector.joblib')
        
        print("All models saved!")
    
    def cross_validate_best(self, X, y):
        """Cross-validate the best models"""
        print("Cross-validating best models...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Test LightGBM and CatBoost (typically best performers)
        models_to_test = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                class_weight='balanced',
                random_state=self.random_state,
                verbose=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        }
        
        for name, model in models_to_test.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
            print(f"{name} CV F1-Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def run_final_optimization(self, data_path="engines_dataset-train_multi_class.csv"):
        """Run the complete final optimization pipeline"""
        print("=== FINAL OPTIMIZATION PIPELINE ===")
        
        # Load data
        df = self.load_data(data_path)
        
        # Create optimized features
        df_opt = self.create_optimized_features(df)
        
        # Prepare data
        X, y = self.prepare_data(df_opt)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Preprocess features
        X_train_processed, X_test_processed = self.preprocess_features(X_train, X_test, y_train)
        
        # Balance classes
        X_train_balanced, y_train_balanced = self.balance_classes(X_train_processed, y_train)
        
        # Train models
        self.train_models(X_train_balanced, y_train_balanced)
        
        # Create ensemble
        self.create_ensemble(X_train_balanced, y_train_balanced)
        
        # Evaluate models
        results_df = self.evaluate_models(X_test_processed, y_test)
        
        # Cross-validation
        X_full_processed = self.feature_selector.transform(self.scaler.transform(X))
        self.cross_validate_best(X_full_processed, y)
        
        # Plot results
        self.plot_results(X_test_processed, y_test)
        
        # Save models
        self.save_models()
        
        # Final summary
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f"\n=== FINAL OPTIMIZATION RESULTS ===")
        print(f"Best Model: {best_model['Model']}")
        print(f"Best Accuracy: {best_model['Accuracy']:.4f}")
        print(f"Best F1-Macro: {best_model['F1_Macro']:.4f}")
        print(f"Improvement over baseline (64.9%): {(best_model['Accuracy'] - 0.649) * 100:.2f} percentage points")
        
        return results_df

if __name__ == "__main__":
    # Run final optimization
    optimizer = FinalOptimizedPredictor(random_state=42)
    results = optimizer.run_final_optimization()
    
    print(f"\nFinal optimization completed successfully!")