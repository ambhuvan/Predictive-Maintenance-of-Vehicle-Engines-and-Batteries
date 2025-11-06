import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

class UltraOptimizedEnginePredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_selector = None
        self.base_models = {}
        self.stacked_model = None
        self.feature_names = None
        
    def load_and_analyze_data(self, data_path="engines_dataset-train_multi_class.csv"):
        """Load data with focused analysis"""
        print("Loading and analyzing data...")
        df = pd.read_csv(data_path)
        
        # Map column names
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
    
    def engineer_discriminative_features(self, df):
        """Focus on most discriminative features based on analysis"""
        print("Engineering discriminative features...")
        
        df_eng = df.copy()
        
        # Remove highly correlated viscosity (99.9% corr with lubOilTemp)
        if 'viscosity' in df_eng.columns:
            df_eng = df_eng.drop('viscosity', axis=1)
        
        # Focus on top discriminative features: engineRpm, coolantTemp, fuelPressure
        # Create targeted feature interactions
        
        # Engine performance features (engineRpm is most discriminative)
        df_eng['rpm_efficiency'] = df_eng['engineRpm'] / (df_eng['fuelPressure'] + 1)
        df_eng['rpm_thermal_load'] = df_eng['engineRpm'] * df_eng['lubOilTemp']
        df_eng['rpm_cooling_ratio'] = df_eng['engineRpm'] / (df_eng['coolantTemp'] + 1)
        
        # Thermal management features (coolantTemp is 2nd most discriminative)
        df_eng['thermal_stress'] = df_eng['coolantTemp'] - df_eng['lubOilTemp']
        df_eng['thermal_efficiency'] = df_eng['coolantPressure'] / (df_eng['coolantTemp'] + 1)
        df_eng['temp_pressure_interaction'] = df_eng['coolantTemp'] * df_eng['coolantPressure']
        
        # Fuel system features (fuelPressure is 3rd most discriminative) 
        df_eng['fuel_efficiency_ratio'] = df_eng['fuelPressure'] / df_eng['engineRpm']
        df_eng['fuel_thermal_interaction'] = df_eng['fuelPressure'] * df_eng['lubOilTemp']
        
        # Pressure ratios
        df_eng['pressure_balance'] = df_eng['lubOilPressure'] / (df_eng['coolantPressure'] + 1e-8)
        df_eng['fuel_lub_pressure_ratio'] = df_eng['fuelPressure'] / (df_eng['lubOilPressure'] + 1e-8)
        
        # Targeted outlier indicators based on analysis
        for col in ['engineRpm', 'coolantTemp', 'fuelPressure']:
            Q1 = df_eng[col].quantile(0.25)
            Q3 = df_eng[col].quantile(0.75)
            IQR = Q3 - Q1
            df_eng[f'{col}_outlier_score'] = np.abs(df_eng[col] - df_eng[col].median()) / (IQR + 1e-8)
        
        # Non-linear transformations for skewed distributions
        df_eng['engineRpm_log'] = np.log1p(df_eng['engineRpm'])
        df_eng['fuelPressure_sqrt'] = np.sqrt(df_eng['fuelPressure'])
        
        print(f"Feature engineering complete. Shape: {df_eng.shape}")
        return df_eng
    
    def advanced_preprocessing(self, X_train, X_test, y_train):
        """Advanced preprocessing focusing on discriminative power"""
        print("Applying advanced preprocessing...")
        
        # Use QuantileTransformer for better handling of outliers and skewed distributions
        self.preprocessor = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Feature selection based on discriminative power
        print("Selecting most discriminative features...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=15)  # Focus on top features
        
        X_train_selected = self.feature_selector.fit_transform(X_train_processed, y_train)
        X_test_selected = self.feature_selector.transform(X_test_processed)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_mask = self.feature_selector.get_support()
            self.selected_features = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
            print(f"Selected features: {self.selected_features}")
        
        return X_train_selected, X_test_selected
    
    def intelligent_sampling(self, X_train, y_train):
        """Intelligent sampling strategy for extreme class imbalance"""
        print("Applying intelligent sampling strategy...")
        
        # Use BorderlineSMOTE for better boundary handling of imbalanced classes
        sampler = BorderlineSMOTE(
            random_state=self.random_state,
            kind='borderline-1'
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        print(f"Resampled shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def train_optimized_models(self, X_train, y_train, X_val, y_val):
        """Train highly optimized models with class weights"""
        print("Training optimized models...")
        
        # Calculate class weights for cost-sensitive learning
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # 1. Optimized XGBoost with class weights
        print("Training optimized XGBoost...")
        scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]  # For binary-like weighting
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric='mlogloss',
            objective='multi:softprob',
            early_stopping_rounds=50,
            tree_method='hist'
        )
        
        # Set sample weights for XGBoost
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        xgb_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 2. Optimized LightGBM with class weights
        print("Training optimized LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            objective='multiclass',
            class_weight='balanced',
            verbose=-1
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 3. Optimized CatBoost with class weights
        print("Training optimized CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            class_weights=list(class_weights),
            random_state=self.random_state,
            verbose=False,
            early_stopping_rounds=50
        )
        
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # 4. Highly tuned Random Forest
        print("Training optimized Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='log2',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        self.base_models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'CatBoost': cb_model,
            'RandomForest': rf_model
        }
        
        return self.base_models
    
    def create_stacked_ensemble(self, X_train, y_train, X_val, y_val):
        """Create advanced stacked ensemble"""
        print("Creating stacked ensemble...")
        
        # Create base estimators for stacking
        base_estimators = [
            ('xgb', self.base_models['XGBoost']),
            ('lgb', self.base_models['LightGBM']),
            ('cb', self.base_models['CatBoost']),
            ('rf', self.base_models['RandomForest'])
        ]
        
        # Meta-learner with class weights
        meta_learner = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Create stacking classifier
        self.stacked_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Fit stacked model
        self.stacked_model.fit(X_train, y_train)
        
        print("Stacked ensemble created successfully!")
        return self.stacked_model
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("Evaluating models...")
        results = []
        
        # Evaluate base models
        for name, model in self.base_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate per-class F1 scores
            f1_per_class = f1_score(y_test, y_pred, average=None)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1_Macro': f1_macro,
                'F1_Weighted': f1_weighted,
                'F1_Class_0': f1_per_class[0],
                'F1_Class_1': f1_per_class[1],
                'F1_Class_2': f1_per_class[2] if len(f1_per_class) > 2 else 0
            })
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))
        
        # Evaluate stacked ensemble
        if self.stacked_model:
            y_pred_stacked = self.stacked_model.predict(X_test)
            accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
            f1_macro_stacked = f1_score(y_test, y_pred_stacked, average='macro')
            f1_weighted_stacked = f1_score(y_test, y_pred_stacked, average='weighted')
            
            f1_per_class_stacked = f1_score(y_test, y_pred_stacked, average=None)
            
            results.append({
                'Model': 'Stacked_Ensemble',
                'Accuracy': accuracy_stacked,
                'F1_Macro': f1_macro_stacked,
                'F1_Weighted': f1_weighted_stacked,
                'F1_Class_0': f1_per_class_stacked[0],
                'F1_Class_1': f1_per_class_stacked[1],
                'F1_Class_2': f1_per_class_stacked[2] if len(f1_per_class_stacked) > 2 else 0
            })
            
            print(f"\nStacked Ensemble Results:")
            print(f"Accuracy: {accuracy_stacked:.4f}, F1-Macro: {f1_macro_stacked:.4f}")
            print(classification_report(y_test, y_pred_stacked, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))
        
        results_df = pd.DataFrame(results)
        print(f"\nFinal Model Comparison:")
        print(results_df.sort_values('F1_Macro', ascending=False))
        
        return results_df
    
    def cross_validate_best_model(self, X, y):
        """Cross-validate the best performing model"""
        print("Performing cross-validation on best models...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        models_to_cv = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=self.random_state,
                verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        }
        
        cv_results = {}
        for name, model in models_to_cv.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name} CV F1-Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_optimized_models(self):
        """Save all optimized models"""
        print("Saving optimized models...")
        
        # Save base models
        for name, model in self.base_models.items():
            filename = f'ultra_optimized_{name.lower()}_model.joblib'
            joblib.dump(model, filename)
            print(f"Saved {name} as {filename}")
        
        # Save stacked ensemble
        if self.stacked_model:
            joblib.dump(self.stacked_model, 'ultra_optimized_stacked_ensemble.joblib')
            print("Saved stacked ensemble")
        
        # Save preprocessors
        if self.preprocessor:
            joblib.dump(self.preprocessor, 'ultra_optimized_preprocessor.joblib')
        if self.feature_selector:
            joblib.dump(self.feature_selector, 'ultra_optimized_feature_selector.joblib')
        
        print("All optimized models saved!")
    
    def run_ultra_optimization(self, data_path="engines_dataset-train_multi_class.csv"):
        """Run the complete ultra-optimization pipeline"""
        print("=== STARTING ULTRA-OPTIMIZATION PIPELINE ===")
        
        # Load and analyze data
        df = self.load_and_analyze_data(data_path)
        
        # Engineer discriminative features
        df_eng = self.engineer_discriminative_features(df)
        
        # Prepare data
        X = df_eng.drop('engineCondition', axis=1)
        y = df_eng['engineCondition']
        self.feature_names = X.columns.tolist()
        
        # Split data strategically
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Further split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Advanced preprocessing
        X_train_processed, X_test_processed = self.advanced_preprocessing(X_train_split, X_test, y_train_split)
        X_val_processed = self.feature_selector.transform(self.preprocessor.transform(X_val))
        
        # Intelligent sampling
        X_train_sampled, y_train_sampled = self.intelligent_sampling(X_train_processed, y_train_split)
        
        # Train optimized models
        self.train_optimized_models(X_train_sampled, y_train_sampled, X_val_processed, y_val)
        
        # Create stacked ensemble
        self.create_stacked_ensemble(X_train_sampled, y_train_sampled, X_val_processed, y_val)
        
        # Comprehensive evaluation
        results_df = self.evaluate_models(X_test_processed, y_test)
        
        # Cross-validation
        X_full_processed = self.feature_selector.transform(self.preprocessor.transform(X))
        cv_results = self.cross_validate_best_model(X_full_processed, y)
        
        # Save models
        self.save_optimized_models()
        
        # Final summary
        best_model = results_df.loc[results_df['F1_Macro'].idxmax()]
        print(f"\n=== ULTRA-OPTIMIZATION RESULTS ===")
        print(f"Best Model: {best_model['Model']}")
        print(f"Best Accuracy: {best_model['Accuracy']:.4f}")
        print(f"Best F1-Macro: {best_model['F1_Macro']:.4f}")
        print(f"Improvement over baseline (65%): {(best_model['Accuracy'] - 0.65) * 100:.2f} percentage points")
        
        return results_df, cv_results

if __name__ == "__main__":
    # Run ultra-optimization
    optimizer = UltraOptimizedEnginePredictor(random_state=42)
    results, cv_results = optimizer.run_ultra_optimization()
    
    print(f"\nUltra-optimization completed successfully!")