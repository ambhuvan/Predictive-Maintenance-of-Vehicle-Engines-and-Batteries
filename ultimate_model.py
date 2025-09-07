import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours  
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class UltimateEngineOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        
    def analyze_class_separability(self, df):
        """Deep analysis of class separability"""
        print("=== ANALYZING CLASS SEPARABILITY ===")
        
        numeric_cols = ['engineRpm', 'lubOilPressure', 'fuelPressure', 'coolantPressure', 'lubOilTemp', 'coolantTemp']
        
        separability_scores = {}
        
        for col in numeric_cols:
            # Calculate between-class variance vs within-class variance
            overall_var = df[col].var()
            
            class_means = []
            class_vars = []
            
            for cls in [0, 1, 2]:
                class_data = df[df['engineCondition'] == cls][col]
                class_means.append(class_data.mean())
                class_vars.append(class_data.var())
            
            between_class_var = np.var(class_means)
            within_class_var = np.mean(class_vars)
            
            separability = between_class_var / (within_class_var + 1e-10)
            separability_scores[col] = separability
            
            print(f"{col}:")
            print(f"  Between-class variance: {between_class_var:.6f}")
            print(f"  Within-class variance: {within_class_var:.6f}")
            print(f"  Separability ratio: {separability:.6f}")
            
            # Show class statistics
            for cls in [0, 1, 2]:
                class_data = df[df['engineCondition'] == cls][col]
                print(f"  Class {cls}: mean={class_data.mean():.3f}, std={class_data.std():.3f}, count={len(class_data)}")
        
        # Sort features by separability
        sorted_features = sorted(separability_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nFeatures ranked by separability:")
        for feature, score in sorted_features:
            print(f"{feature}: {score:.6f}")
        
        return separability_scores, sorted_features
    
    def create_advanced_features(self, df, top_features):
        """Create features based on separability analysis"""
        print("Creating advanced features based on separability analysis...")
        
        df_advanced = df.copy()
        
        # Remove viscosity if present (99.9% correlated with lubOilTemp)
        if 'viscosity' in df_advanced.columns:
            df_advanced = df_advanced.drop('viscosity', axis=1)
        
        # Focus on top 3 most separable features
        top_3_features = [f[0] for f in top_features[:3]]
        print(f"Top 3 separable features: {top_3_features}")
        
        # Advanced transformations for top features
        for feature in top_3_features:
            # Non-linear transformations
            df_advanced[f'{feature}_log'] = np.log1p(df_advanced[feature])
            df_advanced[f'{feature}_sqrt'] = np.sqrt(df_advanced[feature])
            df_advanced[f'{feature}_square'] = df_advanced[feature] ** 2
            
            # Binning based on quartiles
            quartiles = df_advanced[feature].quantile([0.25, 0.5, 0.75])
            df_advanced[f'{feature}_bin'] = pd.cut(df_advanced[feature], 
                                                  bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                                                  labels=[0, 1, 2, 3])
        
        # Create ratios between top separable features
        for i, feat1 in enumerate(top_3_features):
            for feat2 in top_3_features[i+1:]:
                df_advanced[f'{feat1}_{feat2}_ratio'] = df_advanced[feat1] / (df_advanced[feat2] + 1e-8)
                df_advanced[f'{feat1}_{feat2}_diff'] = df_advanced[feat1] - df_advanced[feat2]
                df_advanced[f'{feat1}_{feat2}_product'] = df_advanced[feat1] * df_advanced[feat2]
        
        # Distance from class centroids
        for cls in [0, 1, 2]:
            class_data = df_advanced[df_advanced['engineCondition'] == cls]
            for feature in top_3_features:
                class_mean = class_data[feature].mean()
                df_advanced[f'dist_to_class_{cls}_{feature}'] = np.abs(df_advanced[feature] - class_mean)
        
        print(f"Advanced feature engineering complete. Shape: {df_advanced.shape}")
        return df_advanced
    
    def smart_class_balancing(self, X_train, y_train):
        """Smart class balancing strategy"""
        print("Applying smart class balancing...")
        
        # Use SMOTETomek which combines SMOTE oversampling with Tomek link cleaning
        # This should produce cleaner decision boundaries
        sampler = SMOTETomek(random_state=self.random_state)
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Balanced distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def train_ultimate_model(self, X_train, y_train, X_val, y_val):
        """Train and tune the ultimate model"""
        print("Training and tuning ultimate model...")
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Define models to test with extensive hyperparameter grids
        models = {
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, class_weight='balanced'),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(random_state=self.random_state, verbose=False),
                'params': {
                    'iterations': [200, 300, 500],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [8, 12, 16],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [8, 12, 16],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2']
                }
            }
        }
        
        best_score = 0
        best_model_name = None
        best_model = None
        
        # Grid search for each model
        for name, model_config in models.items():
            print(f"Tuning {name}...")
            
            # Reduce parameter grid for faster execution
            reduced_params = {}
            for param, values in model_config['params'].items():
                reduced_params[param] = values[:2] if len(values) > 2 else values
            
            grid_search = GridSearchCV(
                model_config['model'],
                reduced_params,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            
            # Handle sample weights for XGBoost
            if name == 'XGBoost':
                sample_weights = np.array([class_weight_dict[y] for y in y_train])
                grid_search.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                grid_search.fit(X_train, y_train)
            
            print(f"{name} best score: {grid_search.best_score_:.4f}")
            print(f"{name} best params: {grid_search.best_params_}")
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model_name = name
                best_model = grid_search.best_estimator_
        
        print(f"\nBest model: {best_model_name} with F1-macro: {best_score:.4f}")
        self.best_model = best_model
        
        return best_model, best_model_name, best_score
    
    def evaluate_ultimate_model(self, X_test, y_test):
        """Evaluate the ultimate model"""
        print("Evaluating ultimate model...")
        
        if self.best_model is None:
            print("No model trained yet!")
            return None
        
        y_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Ultimate Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        print(f"Improvement over baseline (64.9%): {(accuracy - 0.649) * 100:.2f} percentage points")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad', 'Good', 'Moderate'], zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Bad', 'Good', 'Moderate'],
                   yticklabels=['Bad', 'Good', 'Moderate'])
        plt.title('Ultimate Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('ultimate_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved as 'ultimate_model_confusion_matrix.png'")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def save_ultimate_model(self):
        """Save the ultimate model"""
        if self.best_model:
            joblib.dump(self.best_model, 'ultimate_optimized_model.joblib')
            if self.preprocessor:
                joblib.dump(self.preprocessor, 'ultimate_preprocessor.joblib')
            print("Ultimate model saved!")
    
    def run_ultimate_optimization(self, data_path="engines_dataset-train_multi_class.csv"):
        """Run the complete ultimate optimization"""
        print("=== ULTIMATE ENGINE CONDITION OPTIMIZATION ===")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Map columns if needed
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
        
        # Analyze class separability
        separability_scores, sorted_features = self.analyze_class_separability(df)
        
        # Create advanced features
        df_advanced = self.create_advanced_features(df, sorted_features)
        
        # Prepare data
        X = df_advanced.drop('engineCondition', axis=1)
        y = df_advanced['engineCondition']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Advanced preprocessing
        self.preprocessor = PowerTransformer(method='yeo-johnson')
        X_train_processed = self.preprocessor.fit_transform(X_train_split)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Smart class balancing
        X_train_balanced, y_train_balanced = self.smart_class_balancing(X_train_processed, y_train_split)
        
        # Train ultimate model
        best_model, best_model_name, best_score = self.train_ultimate_model(
            X_train_balanced, y_train_balanced, X_val_processed, y_val
        )
        
        # Evaluate on test set
        results = self.evaluate_ultimate_model(X_test_processed, y_test)
        
        # Save model
        self.save_ultimate_model()
        
        print(f"\n=== ULTIMATE OPTIMIZATION COMPLETE ===")
        print(f"Best Model: {best_model_name}")
        print(f"Cross-validation F1-Macro: {best_score:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Macro: {results['f1_macro']:.4f}")
        
        return results, best_model_name, best_score

if __name__ == "__main__":
    optimizer = UltimateEngineOptimizer(random_state=42)
    results, best_model_name, cv_score = optimizer.run_ultimate_optimization()
    
    print("Ultimate optimization completed!")