import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def deep_data_analysis():
    """Perform deep analysis to understand data patterns and optimization opportunities"""
    
    # Load data
    df = pd.read_csv("engines_dataset-train_multi_class.csv")
    
    print("=== DEEP DATA ANALYSIS FOR OPTIMIZATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Basic statistics
    print(f"\nTarget distribution:")
    target_dist = df['engineCondition'].value_counts(normalize=True).sort_index()
    print(target_dist)
    
    print(f"\nClass imbalance ratio (worst:best): {target_dist.max() / target_dist.min():.2f}")
    
    # Detailed feature analysis
    numeric_cols = ['engineRpm', 'lubOilPressure', 'fuelPressure', 'coolantPressure', 'lubOilTemp', 'coolantTemp', 'viscosity']
    
    print(f"\n=== FEATURE ANALYSIS BY CLASS ===")
    for col in numeric_cols:
        print(f"\n{col}:")
        class_stats = df.groupby('engineCondition')[col].agg(['mean', 'std', 'min', 'max'])
        print(class_stats)
        
        # Calculate separability (coefficient of variation between classes)
        class_means = df.groupby('engineCondition')[col].mean()
        cv = class_means.std() / class_means.mean()
        print(f"Inter-class coefficient of variation: {cv:.4f}")
    
    # Correlation analysis
    print(f"\n=== CORRELATION ANALYSIS ===")
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print("Highly correlated feature pairs (|r| > 0.7):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
    # Outlier analysis
    print(f"\n=== OUTLIER ANALYSIS ===")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = len(outliers) / len(df) * 100
        
        print(f"{col}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        
        # Check outliers by class
        outlier_by_class = outliers['engineCondition'].value_counts(normalize=True)
        print(f"  Outlier distribution: {outlier_by_class.to_dict()}")
    
    # Feature separability analysis
    print(f"\n=== FEATURE SEPARABILITY ANALYSIS ===")
    
    # Calculate Fisher's Linear Discriminant ratio for each feature
    def fisher_score(feature, target):
        classes = target.unique()
        overall_mean = feature.mean()
        
        numerator = 0
        denominator = 0
        
        for cls in classes:
            class_data = feature[target == cls]
            class_mean = class_data.mean()
            class_var = class_data.var()
            class_count = len(class_data)
            
            numerator += class_count * (class_mean - overall_mean) ** 2
            denominator += class_count * class_var
        
        return numerator / (denominator + 1e-10)
    
    fisher_scores = {}
    for col in numeric_cols:
        score = fisher_score(df[col], df['engineCondition'])
        fisher_scores[col] = score
    
    # Sort by discriminative power
    sorted_features = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
    print("Features ranked by discriminative power (Fisher score):")
    for feature, score in sorted_features:
        print(f"{feature}: {score:.4f}")
    
    # Class overlap analysis
    print(f"\n=== CLASS OVERLAP ANALYSIS ===")
    
    # Scale the data for analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Clustering analysis to understand natural groupings
    print(f"\n=== CLUSTERING ANALYSIS ===")
    
    silhouette_scores = []
    for n_clusters in range(2, 8):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))
        print(f"K={n_clusters}, Silhouette score: {silhouette_avg:.4f}")
    
    best_k = max(silhouette_scores, key=lambda x: x[1])
    print(f"Best number of clusters: {best_k[0]} (silhouette: {best_k[1]:.4f})")
    
    # Analyze cluster-class relationship
    kmeans_best = KMeans(n_clusters=best_k[0], random_state=42)
    cluster_labels = kmeans_best.fit_predict(X_scaled)
    
    df_cluster = df.copy()
    df_cluster['cluster'] = cluster_labels
    cluster_class_relation = pd.crosstab(df_cluster['cluster'], df_cluster['engineCondition'], normalize='index')
    print(f"\nCluster-Class relationship (normalized by cluster):")
    print(cluster_class_relation)
    
    # Generate visualizations
    create_analysis_plots(df, numeric_cols, X_pca)
    
    return df, fisher_scores, sorted_features

def create_analysis_plots(df, numeric_cols, X_pca):
    """Create comprehensive analysis plots"""
    
    # 1. Feature distributions by class
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(numeric_cols):
        plt.subplot(3, 3, i+1)
        for cls in sorted(df['engineCondition'].unique()):
            data = df[df['engineCondition'] == cls][col]
            plt.hist(data, alpha=0.6, label=f'Class {cls}', bins=30)
        plt.title(f'{col} Distribution by Class')
        plt.legend()
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA visualization
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for i, cls in enumerate(sorted(df['engineCondition'].unique())):
        mask = df['engineCondition'] == cls
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=f'Class {cls}', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization of Classes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_class_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plots for outlier visualization
    plt.figure(figsize=(20, 12))
    for i, col in enumerate(numeric_cols):
        plt.subplot(3, 3, i+1)
        df.boxplot(column=col, by='engineCondition', ax=plt.gca())
        plt.title(f'{col} by Engine Condition')
        plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig('boxplots_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis plots saved:")
    print("- feature_distributions_by_class.png")
    print("- pca_class_visualization.png") 
    print("- correlation_heatmap.png")
    print("- boxplots_by_class.png")

def suggest_optimizations(fisher_scores, sorted_features):
    """Suggest optimization strategies based on analysis"""
    
    print(f"\n=== OPTIMIZATION SUGGESTIONS ===")
    
    # 1. Feature engineering suggestions
    print("1. FEATURE ENGINEERING:")
    print("   - Focus on top discriminative features:", [f[0] for f in sorted_features[:3]])
    print("   - Create ratios and interactions between top features")
    print("   - Consider log transformation for skewed features")
    print("   - Apply polynomial features selectively to avoid overfitting")
    
    # 2. Data preprocessing suggestions
    print("\n2. DATA PREPROCESSING:")
    print("   - Use robust scaling instead of standard scaling (outlier-resistant)")
    print("   - Consider outlier removal or capping (but preserve class balance)")
    print("   - Apply feature selection based on Fisher scores")
    
    # 3. Model strategy suggestions
    print("\n3. MODEL STRATEGY:")
    print("   - Focus on tree-based models (handle non-linearity well)")
    print("   - Use cost-sensitive learning for class imbalance")
    print("   - Consider advanced ensemble methods (stacking)")
    print("   - Apply different strategies for minority class (Class 2)")
    
    # 4. Sampling strategy suggestions
    print("\n4. SAMPLING STRATEGY:")
    print("   - Try different SMOTE variants (BorderlineSMOTE, ADASYN)")
    print("   - Consider class weights instead of oversampling")
    print("   - Use stratified sampling to preserve class distribution")
    
    # 5. Validation strategy
    print("\n5. VALIDATION STRATEGY:")
    print("   - Use stratified cross-validation")
    print("   - Focus on F1-macro score (better for imbalanced data)")
    print("   - Consider per-class metrics for minority class performance")

if __name__ == "__main__":
    df, fisher_scores, sorted_features = deep_data_analysis()
    suggest_optimizations(fisher_scores, sorted_features)