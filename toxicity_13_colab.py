# %% [markdown]
# Toxicity Dataset : https://archive.ics.uci.edu/dataset/728/toxicity-2
# 
# The dataset includes 171 molecules designed for functional domains of a core clock protein, CRY1, responsible for generating circadian rhythm. 56 of the molecules are toxic and the rest are non-toxic. 
# 
# The data consists a complete set of 1203 molecular descriptors and needs feature selection before classification since some of the features are redundant. 
# 
# Introductory Paper:
# Structure-based design and classifications of small molecules regulating the circadian rhythm period
# By Seref Gul, F. Rahim, Safak Isin, Fatma Yilmaz, Nuri Ozturk, M. Turkay, I. Kavakli. 2021
# https://www.semanticscholar.org/paper/Structure-based-design-and-classifications-of-small-Gul-Rahim/5944836c47bc7d1a2b0464a9a1db94d4bc7f28ce

# %% [markdown]
# # Imports

# %%
pip install ucimlrepo

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import networkx as nx

from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_validate, GridSearchCV, learning_curve, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, auc, roc_curve,
    classification_report, matthews_corrcoef,
    average_precision_score, balanced_accuracy_score, make_scorer
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn import clone
from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBClassifier

# For SMOTE (install: pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# For statistical testing
from scipy.stats import ttest_rel, wilcoxon

# %%
# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# %%
# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# # Data loading and preprocessing

# %%
data = fetch_ucirepo(id=728)
X = data.data.features
y = data.data.targets

# Binary encoding: NonToxic=1, Toxic=0
y_binary = (y['Class'] == 'NonToxic').astype(int)

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Feature matrix shape: {X.shape}")
print(f"Number of molecules (n): {X.shape[0]}")
print(f"Number of descriptors (p): {X.shape[1]}")
print(f"Dimensionality ratio (p/n): {X.shape[1]/X.shape[0]:.2f}")
print(f"\nClass distribution:")
print(y['Class'].value_counts())
print(f"\nClass balance:")
for cls, count in y['Class'].value_counts().items():
    pct = count / len(y) * 100
    print(f"  {cls}: {count} ({pct:.2f}%)")

# %%
# Check for missing values
print(f"\nMissing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y_binary.isnull().sum()}")

# %%
original_features = ['MDEC-23', 'MATS2v', 'ATSC8s', 'VE3_Dt', 'CrippenMR', 
                    'SpMax7_Bhe', 'SpMin1_Bhs', 'C1SP2', 'GATS8e', 'GATS8s', 
                    'SpMax5_Bhv', 'VE3_Dzi', 'VPC-4']

# %% [markdown]
# # Feature Correlation Analysis

# %%
def analyze_multicollinearity_advanced(X, corr_threshold=0.9, vif_threshold=10.0):
    """
    Comprehensive multicollinearity analysis including Pairwise, VIF, and Eigenvalue diagnostics.
    """
    print(f"{'='*80}")
    print(f"ADVANCED MULTICOLLINEARITY DIAGNOSTICS")
    print(f"{'='*80}\n")
    
    n_features = X.shape[1]
    
    # ---------------------------------------------------------
    # 1. Eigenvalue Analysis (Condition Number)
    # ---------------------------------------------------------
    # A condition number > 30 indicates moderate to severe multicollinearity
    # A very high number confirms "mathematically degenerate" matrix
    corr_matrix = X.corr()
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    condition_number = np.sqrt(eigenvalues.max() / (eigenvalues.min() + 1e-10)) # Avoid div by 0
    
    print(f"1. GLOBAL STABILITY ANALYSIS")
    print(f"   - Condition Number: {condition_number:.2e}")
    print(f"   - Effective Rank (Eigenvalues > 1e-5): {np.sum(eigenvalues > 1e-5)} / {n_features}")
    if condition_number > 100:
        print("   -> DIAGNOSIS: Catastrophic Multicollinearity (Ill-conditioned matrix)")
    
    # ---------------------------------------------------------
    # 2. Pairwise Correlation & Grouping (Graph Theory)
    # ---------------------------------------------------------
    # Instead of just listing pairs, we group them into "communities" of redundant features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = upper.stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > corr_threshold]
    
    # Create a graph of correlations
    G = nx.Graph()
    for (f1, f2), val in high_corr_pairs.items():
        G.add_edge(f1, f2, weight=val)
    
    components = list(nx.connected_components(G))
    
    print(f"\n2. PAIRWISE REDUNDANCY (r > {corr_threshold})")
    print(f"   - Total Correlated Pairs: {len(high_corr_pairs)}")
    print(f"   - Redundant Feature Groups (Clusters): {len(components)}")
    
    # Show largest cluster example
    if components:
        largest_cluster = max(components, key=len)
        print(f"   - Example Largest Cluster (Size {len(largest_cluster)}): {list(largest_cluster)[:5]}...")

    # ---------------------------------------------------------
    # 3. Variance Inflation Factor (VIF)
    # ---------------------------------------------------------
    # Note: This can be slow for very large N. We use a progress update.
    print(f"\n3. VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("   Calculating VIF for all features (this may take a moment)...")
    
    # Handle infinite VIF by catching errors or checking perfect correlation
    vif_data = []
    
    # We use the correlation matrix inverse diagonal for efficiency if N is large
    # VIF_i = 1 / (1 - R_i^2) = Diagonal of Inverse Correlation Matrix
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        vif_values = np.diag(inv_corr)
        vif_series = pd.Series(vif_values, index=corr_matrix.index)
    except np.linalg.LinAlgError:
        print("   ! Matrix is singular. Using pseudo-inverse (approximate VIF).")
        inv_corr = np.linalg.pinv(corr_matrix.values)
        vif_series = pd.Series(np.diag(inv_corr), index=corr_matrix.index)

    # Statistics
    high_vif_count = (vif_series > vif_threshold).sum()
    pct_high_vif = (high_vif_count / n_features) * 100
    
    print(f"   - Severity: {pct_high_vif:.1f}% of features ({high_vif_count}/{n_features}) have VIF > {vif_threshold}")
    print(f"   - Mean VIF: {vif_series.mean():.2f}")
    print(f"   - Max VIF:  {vif_series.max():.2f} (Feature: {vif_series.idxmax()})")

    # ---------------------------------------------------------
    # 4. Visualization (Scree Plot + Histogram)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of Correlations
    all_corrs = upper.values.flatten()
    all_corrs = all_corrs[~np.isnan(all_corrs)]
    ax1.hist(all_corrs, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(corr_threshold, color='red', linestyle='--', label=f'Threshold (+/-) {corr_threshold}')
    ax1.axvline(-corr_threshold, color='red', linestyle='--')
    ax1.set_title('Pairwise Correlation Distribution')
    ax1.set_xlabel('Absolute Correlation')
    ax1.legend()
    
    # Scree Plot (Log Eigenvalues)
    ax2.plot(np.arange(len(eigenvalues)), np.log10(eigenvalues[::-1]), 'b-', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('Eigenspectrum (Log Scale)')
    ax2.set_xlabel('Principal Component Index')
    ax2.set_ylabel('Log10(Eigenvalue)')
    ax2.text(0.5, 0.5, 'Steep drop = High Redundancy', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'high_corr_pairs': high_corr_pairs,
        'vif_data': vif_series,
        'condition_number': condition_number
    }

stats = analyze_multicollinearity_advanced(X)

# %% [markdown]
# # Train-Test Split with Stratification

# %%
# Shuffle dataset
np.random.seed(RANDOM_STATE)
shuffle_idx = np.random.permutation(len(X))
X_shuffled = X.iloc[shuffle_idx].reset_index(drop=True)
y_shuffled = y_binary.iloc[shuffle_idx].reset_index(drop=True)

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_shuffled
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n{'='*80}")
print("TRAIN-TEST SPLIT")
print("="*80)
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"\nTraining class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nTest class distribution:")
print(pd.Series(y_test).value_counts())

feature_names = X.columns.tolist()

# %%
# Compute class weights for weighted models
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\n{'='*80}")
print("CLASS WEIGHTS")
print("="*80)
print(f"Class 0 (Toxic): {class_weight_dict[0]:.3f}")
print(f"Class 1 (NonToxic): {class_weight_dict[1]:.3f}")

# %% [markdown]
# # SMOTE for Class Imbalance

# %%
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n{'='*80}")
print("SMOTE RESAMPLING")
print("="*80)
print(f"Original training size: {X_train_scaled.shape}")
print(f"SMOTE training size: {X_train_smote.shape}")
print(f"\nOriginal class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nSMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts())

# %% [markdown]
# # Model Definitions and Hyperparameter Grids

# %%
models = {
    # ===== PENALIZED REGRESSION MODELS =====
    # No penalty
    'LR_No_Penalty': LogisticRegression(penalty=None, max_iter=5000, 
                                        solver='lbfgs', random_state=RANDOM_STATE),
    
    # Ridge (L2) - Multiple C values
    'LR_Ridge_C0.01': LogisticRegression(penalty='l2', C=0.01, max_iter=5000,
                                         solver='lbfgs', random_state=RANDOM_STATE),
    'LR_Ridge_C0.1': LogisticRegression(penalty='l2', C=0.1, max_iter=5000,
                                        solver='lbfgs', random_state=RANDOM_STATE),
    'LR_Ridge_C1': LogisticRegression(penalty='l2', C=1.0, max_iter=5000,
                                      solver='lbfgs', random_state=RANDOM_STATE),
    'LR_Ridge_C10': LogisticRegression(penalty='l2', C=10.0, max_iter=5000,
                                       solver='lbfgs', random_state=RANDOM_STATE),
    'LR_Ridge_C100': LogisticRegression(penalty='l2', C=100.0, max_iter=5000,
                                        solver='lbfgs', random_state=RANDOM_STATE),
    
    # Lasso (L1) - Multiple C values
    'LR_Lasso_C0.001': LogisticRegression(penalty='l1', C=0.001, max_iter=5000,
                                          solver='saga', random_state=RANDOM_STATE),
    'LR_Lasso_C0.01': LogisticRegression(penalty='l1', C=0.01, max_iter=5000,
                                         solver='saga', random_state=RANDOM_STATE),
    'LR_Lasso_C0.1': LogisticRegression(penalty='l1', C=0.1, max_iter=5000,
                                        solver='saga', random_state=RANDOM_STATE),
    'LR_Lasso_C1': LogisticRegression(penalty='l1', C=1.0, max_iter=5000,
                                      solver='saga', random_state=RANDOM_STATE),
    'LR_Lasso_C10': LogisticRegression(penalty='l1', C=10.0, max_iter=5000,
                                       solver='saga', random_state=RANDOM_STATE),
    
    # Elastic Net - Multiple configurations
    'LR_ElasticNet_L1_0.3_C0.1': LogisticRegression(penalty='elasticnet', solver='saga',
                                                     l1_ratio=0.3, C=0.1, max_iter=5000,
                                                     random_state=RANDOM_STATE),
    'LR_ElasticNet_L1_0.5_C0.1': LogisticRegression(penalty='elasticnet', solver='saga',
                                                     l1_ratio=0.5, C=0.1, max_iter=5000,
                                                     random_state=RANDOM_STATE),
    'LR_ElasticNet_L1_0.5_C1': LogisticRegression(penalty='elasticnet', solver='saga',
                                                   l1_ratio=0.5, C=1.0, max_iter=5000,
                                                   random_state=RANDOM_STATE),
    'LR_ElasticNet_L1_0.7_C1': LogisticRegression(penalty='elasticnet', solver='saga',
                                                   l1_ratio=0.7, C=1.0, max_iter=5000,
                                                   random_state=RANDOM_STATE),
    
    # Weighted versions of best regularized models
    'LR_Lasso_C0.1_Weighted': LogisticRegression(penalty='l1', C=0.1, max_iter=5000,
                                                 solver='saga', class_weight='balanced',
                                                 random_state=RANDOM_STATE),
    'LR_Ridge_C0.1_Weighted': LogisticRegression(penalty='l2', C=0.1, max_iter=5000,
                                                 solver='lbfgs', class_weight='balanced',
                                                 random_state=RANDOM_STATE),
    
    # Other linear classifiers
    'Ridge_Classifier': RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
    'SGD_Classifier': SGDClassifier(loss='log_loss', max_iter=5000, 
                                   random_state=RANDOM_STATE),

    # ===== DISCRIMINANT ANALYSIS =====
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),

    # ===== NAIVE BAYES =====
    'Naive_Bayes': GaussianNB(),

    # ===== TREE-BASED MODELS =====
    'Decision_Tree_D5': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    'Decision_Tree_D10': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Decision_Tree_D20': DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE),
    'Decision_Tree_Weighted': DecisionTreeClassifier(max_depth=10, 
                                                     class_weight='balanced',
                                                     random_state=RANDOM_STATE),

    # ===== ENSEMBLE METHODS =====
    # Random Forest
    'Random_Forest_N50': RandomForestClassifier(n_estimators=50, max_depth=10, 
                                                random_state=RANDOM_STATE),
    'Random_Forest_N100': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                 random_state=RANDOM_STATE),
    'Random_Forest_N100_Weighted': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                          class_weight='balanced',
                                                          random_state=RANDOM_STATE),
    'Extra_Trees_N100': ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                            random_state=RANDOM_STATE),

    # Boosting
    'AdaBoost_N50': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE,
                                       algorithm='SAMME'),
    'AdaBoost_N100': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                        algorithm='SAMME'),
    'GradientBoosting_N50': GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                       random_state=RANDOM_STATE),
    'GradientBoosting_N100': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                        random_state=RANDOM_STATE),
    'XGBoost_D3_N50': XGBClassifier(max_depth=3, n_estimators=50,
                                   eval_metric='logloss', use_label_encoder=False,
                                   random_state=RANDOM_STATE),
    'XGBoost_D3_N100': XGBClassifier(max_depth=3, n_estimators=100,
                                    eval_metric='logloss', use_label_encoder=False,
                                    random_state=RANDOM_STATE),

    # ===== SUPPORT VECTOR MACHINES =====
    'SVM_Linear': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
    'SVM_Linear_Weighted': SVC(kernel='linear', probability=True, 
                              class_weight='balanced', random_state=RANDOM_STATE),
    'SVM_RBF_C1': SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE),
    'SVM_RBF_C10': SVC(kernel='rbf', C=10.0, probability=True, random_state=RANDOM_STATE),
    'SVM_Poly_D2': SVC(kernel='poly', degree=2, probability=True, random_state=RANDOM_STATE),
    'SVM_Poly_D3': SVC(kernel='poly', degree=3, probability=True, random_state=RANDOM_STATE),

    # ===== K-NEAREST NEIGHBORS =====
    'KNN_K3': KNeighborsClassifier(n_neighbors=3),
    'KNN_K5': KNeighborsClassifier(n_neighbors=5),
    'KNN_K7': KNeighborsClassifier(n_neighbors=7),
    'KNN_K10': KNeighborsClassifier(n_neighbors=10),

    # ===== NEURAL NETWORKS =====
    'NN_Small': MLPClassifier(hidden_layer_sizes=(25,), max_iter=2000,
                             early_stopping=True, solver='lbfgs', 
                             random_state=RANDOM_STATE),
    'NN_Medium': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=2000,
                              early_stopping=True, solver='lbfgs',
                              random_state=RANDOM_STATE),
    'NN_Large': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000,
                             early_stopping=True, solver='lbfgs',
                             random_state=RANDOM_STATE),
    'NN_Adam': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=2000,
                            early_stopping=True, solver='adam',
                            random_state=RANDOM_STATE),
}

print(f"\nTotal models defined: {len(models)}")

# %% [markdown]
# # Train and Evaluate Models

# %%
def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, 
                                model_name="Model"):
    """
    Comprehensive evaluation including:
    - Standard metrics (accuracy, precision, recall, F1, AUC)
    - Matthews Correlation Coefficient (MCC)
    - Confusion matrix
    - Probability calibration check
    """
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probability scores
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_train_proba = model.decision_function(X_train)
        y_test_proba = model.decision_function(X_test)
    else:
        y_train_proba = y_train_pred.astype(float)
        y_test_proba = y_test_pred.astype(float)
    
    # Calculate metrics
    results = {
        'model': model_name,
        'train_acc': accuracy_score(y_train, y_train_pred),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_test_pred),
        'pr_auc': average_precision_score(y_test, y_test_proba),
        'bacc': balanced_accuracy_score(y_test, y_test_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity, Sensitivity, NPV
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Check for majority class prediction
    unique_preds = np.unique(y_test_pred)
    results['predicts_one_class'] = len(unique_preds) == 1
    
    return results, cm

# %% [markdown]
# ## STANDARD TRAIN-TEST SPLIT

# %%
print("\n" + "="*80)
print("TRAINING MODELS - STANDARD TRAIN-TEST SPLIT")
print("="*80)

results_standard = []
confusion_matrices = {}

for name, model in models.items():
    print(f"Training {name}...", end=" ")
    try:
        metrics, cm = evaluate_model_comprehensive(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        results_standard.append(metrics)
        confusion_matrices[name] = cm
        
        # Flag models predicting only one class
        flag = " [⚠ ONE CLASS]" if metrics['predicts_one_class'] else ""
        print(f"mcc: {metrics['mcc']:.4f}, pr_auc: {metrics['pr_auc']:.4f}, bacc: {metrics['bacc']:.4f}{flag}")
    except Exception as e:
        print(f"FAILED: {e}")

results_standard_df = pd.DataFrame(results_standard).sort_values('mcc', ascending=False).reset_index(drop=True)

# %%
results_standard_df

# %% [markdown]
# ## With SMOTE Resampling

# %%
print("\n" + "="*80)
print("TRAINING MODELS - WITH SMOTE RESAMPLING")
print("="*80)

results_smote = []

for name, model in models.items():
    # Skip if model doesn't work well with SMOTE
    if 'LDA' in name or 'QDA' in name:
        continue
        
    print(f"Training {name} with SMOTE...", end=" ")
    try:
        metrics, cm = evaluate_model_comprehensive(
            model, X_train_smote, X_test_scaled, y_train_smote, y_test, 
            f"{name}_SMOTE"
        )
        results_smote.append(metrics)
        
        flag = " [⚠ ONE CLASS]" if metrics['predicts_one_class'] else ""
        print(f"mcc: {metrics['mcc']:.4f}, pr_auc: {metrics['pr_auc']:.4f}, bacc: {metrics['bacc']:.4f}{flag}")
    except Exception as e:
        print(f"FAILED: {e}")

results_smote_df = pd.DataFrame(results_smote).sort_values('mcc', ascending=False).reset_index(drop=True)

# %%
results_smote_df

# %%
results_combined_df = pd.concat([results_standard_df, results_smote_df], 
                                ignore_index=True)
results_combined_df = results_combined_df.sort_values('mcc', 
                                                       ascending=False).reset_index(drop=True)

# %% [markdown]
# ## Model Grouping for Analysis

# %%
def assign_model_group(name):
    """Assign models to categories for grouped analysis"""
    if 'SMOTE' in name:
        return 'SMOTE Variants'
    elif any(k in name for k in ['LR_', 'Ridge_Classifier', 'SGD_Classifier']):
        return 'Linear / Penalized'
    elif 'Decision_Tree' in name and not any(x in name for x in ['Random', 'Extra']):
        return 'Single Tree'
    elif any(k in name for k in ['Random_Forest', 'Extra_Trees']):
        return 'Random Forest / ExtraTrees'
    elif any(k in name for k in ['AdaBoost', 'GradientBoosting', 'XGBoost']):
        return 'Boosting Ensembles'
    elif 'SVM_' in name:
        return 'SVM'
    elif 'KNN_' in name:
        return 'KNN'
    elif 'NN_' in name:
        return 'Neural Networks'
    elif name in ['LDA', 'QDA']:
        return 'Discriminant Analysis'
    elif name == 'Naive_Bayes':
        return 'Naive Bayes'
    else:
        return 'Other'

results_combined_df['Group'] = results_combined_df['model'].apply(assign_model_group)


# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Performance Comparison

# %%
def plot_single_metric(results_df, n=15, metric='test_auc', title=None, figsize=(12, 10)):    
    # Validate metric exists
    available_metrics = results_df.columns.tolist()
    if metric not in available_metrics:
        raise ValueError(f"metric '{metric}' not found. Available: {available_metrics}")
    
    # Sort by metric and get top n
    top_n = results_df.nlargest(n, metric).copy()
    top_n = top_n.iloc[::-1]  # Reverse for better display
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine colors based on one-class prediction flag if available
    if 'predicts_one_class' in top_n.columns:
        colors = ['#2bbbdf' if not row['predicts_one_class'] else '#ff6b6b' 
                  for _, row in top_n.iterrows()]
    else:
        colors = '#2bbbdf'
    
    # ===== PLOT METRIC =====
    ax.barh(range(len(top_n)), top_n[metric], color=colors, alpha=0.8, 
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['model'], fontsize=10)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, weight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} (Top {n} Models)', 
                 fontsize=14, weight='bold')
    
    # Set x-limits based on metric range
    if metric in ['test_acc', 'test_auc', 'precision', 'recall', 'f1', 'pr_auc', 
                  'bacc', 'sensitivity', 'specificity', 'npv', 'train_acc', 'train_auc']:
        ax.set_xlim(0, 1.0)
    
    ax.grid(axis='x', alpha=0.3)
    
    # ===== LEGEND =====
    if 'predicts_one_class' in top_n.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2bbbdf', alpha=0.8, label='Valid predictions'),
            Patch(facecolor='#ff6b6b', alpha=0.8, label='One-class predictions')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                  framealpha=0.95, edgecolor='black')
    
    # ===== TITLE =====
    if title is None:
        title = f'Top {n} Models by {metric.upper()}'
    
    fig.suptitle(title, fontsize=16, weight='bold', y=0.995)
    
    plt.tight_layout()
    
    return fig


# %%
# Example 1: Test AUC
fig1 = plot_single_metric(results_combined_df, n=15, metric='test_auc',
                          title='Top 15 Models: Test AUC')
plt.savefig('fig_top15_test_auc.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 2: MCC
fig2 = plot_single_metric(results_combined_df, n=15, metric='mcc',
                          title='Top 15 Models: Matthews Correlation Coefficient')
plt.savefig('fig_top15_mcc.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 3: F1 Score
fig3 = plot_single_metric(results_combined_df, n=15, metric='f1',
                          title='Top 15 Models: F1 Score')
plt.savefig('fig_top15_f1.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 4: PR-AUC
fig4 = plot_single_metric(results_combined_df, n=15, metric='pr_auc',
                          title='Top 15 Models: Precision-Recall AUC')
plt.savefig('fig_top15_pr_auc.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 5: Balanced Accuracy
fig5 = plot_single_metric(results_combined_df, n=20, metric='bacc',
                          title='Top 20 Models: Balanced Accuracy')
plt.savefig('fig_top20_bacc.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 6: Sensitivity
fig6 = plot_single_metric(results_combined_df, n=15, metric='sensitivity',
                          title='Top 15 Models: Sensitivity (True Positive Rate)')
plt.savefig('fig_top15_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
def plot_train_test_scatter_enhanced(results_df):
    """Enhanced train vs test scatter with overfitting zones and improved visual design"""
    df_plot = results_df[~results_df['model'].str.contains('SMOTE')].copy()
    
    # Calculate overfitting gap for analysis
    df_plot['overfitting_gap'] = df_plot['train_acc'] - df_plot['test_acc']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by group with improved palette
    groups = df_plot['Group'].unique()
    palette = sns.color_palette('husl', n_colors=len(groups))  # Better color distinction
    color_map = dict(zip(groups, palette))
    
    # Plot each group with enhanced styling
    for group in groups:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['train_acc'], group_data['test_acc'],
                  label=group, s=150, alpha=0.75,  # Larger markers
                  color=color_map[group], 
                  edgecolors='white', linewidth=1.5,  # White borders for better visibility
                  zorder=3)  # Ensure points are on top
    
    # Reference lines with improved styling
    ax.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, linewidth=2.5, 
            label='Perfect Generalization', zorder=2)
    
    # Multiple overfitting zones for better interpretation
    ax.fill_between([0.5, 1], [0.45, 0.95], [0.5, 1], 
                    alpha=0.15, color='red', label='Severe Overfitting', zorder=1)
    ax.fill_between([0.5, 1], [0.475, 0.975], [0.5, 1], 
                    alpha=0.08, color='orange', label='Mild Overfitting', zorder=1)
    
    # Underfitting zone (below diagonal)
    ax.fill_between([0.5, 1], [0.5, 1], [0.55, 1.05], 
                    alpha=0.08, color='blue', label='Underfitting', zorder=1)
    
    # Annotate top 3 overfitters
    top_overfitters = df_plot.nlargest(3, 'overfitting_gap')
    for idx, row in top_overfitters.iterrows():
        ax.annotate(row['model'].split('_')[0],  # Shortened label
                   xy=(row['train_acc'], row['test_acc']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.4),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                 alpha=0.6, lw=1))
    
    # Styling improvements
    ax.set_xlabel('Training Accuracy', fontsize=14, weight='semibold')
    ax.set_ylabel('Test Accuracy', fontsize=14, weight='semibold')
    ax.set_title('Model Generalization: Train vs Test Accuracy Analysis', 
                fontsize=16, weight='bold', pad=20)
    
    # Improved legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Set limits with padding
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    
    # Enhanced grid
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)  # Grid behind all elements
    
    # Add minor ticks for better readability
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.025))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    
    # Background color
    ax.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    plt.savefig('fig_train_test_scatter.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Enhanced overfitting summary
    print("\n" + "="*90)
    print(" "*30 + "OVERFITTING ANALYSIS")
    print("="*90)
    
    print(f"\nDataset Statistics:")
    print(f"  Total models analyzed: {len(df_plot)}")
    print(f"  Mean overfitting gap: {df_plot['overfitting_gap'].mean():.4f}")
    print(f"  Median overfitting gap: {df_plot['overfitting_gap'].median():.4f}")
    print(f"  Std overfitting gap: {df_plot['overfitting_gap'].std():.4f}")
    
    print("\n" + "-"*90)
    print("Top 5 Worst Overfitters (High Train-Test Gap):")
    print("-"*90)
    worst = df_plot.nlargest(5, 'overfitting_gap')[
        ['model', 'train_acc', 'test_acc', 'overfitting_gap']
    ]
    print(worst.to_string(index=False))
    
    print("\n" + "-"*90)
    print("Top 5 Best Generalizers (Low Train-Test Gap):")
    print("-"*90)
    best = df_plot.nsmallest(5, 'overfitting_gap')[
        ['model', 'train_acc', 'test_acc', 'overfitting_gap']
    ]
    print(best.to_string(index=False))
    
    # Additional insights
    print("\n" + "-"*90)
    print("Performance by Group:")
    print("-"*90)
    group_stats = df_plot.groupby('Group').agg({
        'overfitting_gap': ['mean', 'std'],
        'test_acc': ['mean', 'max']
    }).round(4)
    print(group_stats)
    print("="*90 + "\n")


plot_train_test_scatter_enhanced(results_combined_df)


# %%
def plot_accuracy_vs_auc_threshold_improved(results_df):
    df_plot = results_df[~results_df['model'].str.contains('SMOTE')].copy()
    df_plot['gap'] = df_plot['test_acc'] - df_plot['test_auc']
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Add subtle reference grid FIRST (behind everything)
    for val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ax.axhline(val, color='gray', alpha=0.08, linewidth=0.7, zorder=1)
        ax.axvline(val, color='gray', alpha=0.08, linewidth=0.7, zorder=1)
    
    # Zones with reduced opacity
    ax.fill_between([0.4, 1.0], [0.4, 1.0], [0.35, 0.95],
                    alpha=0.03, color='red', label='Deceptive Zone', zorder=2)
    ax.fill_between([0.4, 1.0], [0.4, 1.0], [0.45, 1.05],
                    alpha=0.03, color='green', label='Robust Zone', zorder=2)
    
    # Add multiple threshold lines for severity levels
    ax.plot([0.4, 1.0], [0.4, 1.0], 'r--', alpha=0.6, linewidth=2.5,
            label='Balance Line (AUC = Acc)', zorder=3)
    ax.plot([0.4, 0.98], [0.42, 1.0], 'orange', alpha=0.3, linewidth=1.5,
            linestyle=':', label='Warning Zone (Gap = 0.02)', zorder=3)
    
    # Scatter with gap-based coloring (alternative approach)
    scatter = ax.scatter(df_plot['test_acc'], df_plot['test_auc'],
                        c=df_plot['gap'], s=180, alpha=0.75,
                        cmap='RdYlGn_r', edgecolors='white', linewidth=1.5,
                        vmin=-0.05, vmax=0.15, zorder=4)
    
    # Colorbar for gap values
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Accuracy - AUC Gap', fontsize=12, weight='bold')
    
    # Annotate top deceptive models
    deceptive_top = df_plot.nlargest(5, 'gap')
    for _, row in deceptive_top.iterrows():
        ax.annotate(row['model'], 
                   (row['test_acc'], row['test_auc']),
                   xytext=(10, -10), textcoords='offset points',
                   fontsize=8, alpha=0.8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', lw=0.8), zorder=5)
    
    # Annotate top robust models
    robust_top = df_plot.nsmallest(3, 'gap')
    for _, row in robust_top.iterrows():
        ax.annotate(row['model'],
                   (row['test_acc'], row['test_auc']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, alpha=0.8, color='darkgreen', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', lw=0.8, color='green'), zorder=5)
    
    # Quadrant labels with context
    ax.text(0.55, 0.92, 'Ideal Zone\n(High Acc, High AUC)',
           fontsize=11, weight='bold', color='darkgreen', alpha=0.6,
           ha='center', bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.15))
    ax.text(0.75, 0.55, 'Deceptive Models\n(High Acc, Low AUC)\nClass Imbalance Risk',
           fontsize=11, weight='bold', color='darkred', alpha=0.6,
           ha='center', bbox=dict(boxstyle='round', fc='salmon', alpha=0.15))
    
    ax.set_xlabel('Test Accuracy', fontsize=15, weight='bold')
    ax.set_ylabel('Test AUC-ROC', fontsize=15, weight='bold')
    ax.set_title('Model Performance Diagnostic: Accuracy vs. AUC\nDetecting Class Imbalance & Deceptive Metrics',
                fontsize=16, weight='bold', pad=20)
    
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.35, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('fig_accuracy_vs_auc_diagnostic.png', dpi=400, bbox_inches='tight')
    plt.show()
    
    # Enhanced analysis with severity classification
    df_plot['severity'] = pd.cut(df_plot['gap'], 
                                  bins=[-np.inf, 0, 0.02, 0.05, np.inf],
                                  labels=['Robust', 'Acceptable', 'Warning', 'Critical'])
    
    print("\n" + "="*90)
    print("MODEL PERFORMANCE DIAGNOSTIC SUMMARY")
    print("="*90)
    print(f"\nSeverity Distribution:")
    print(df_plot['severity'].value_counts().to_string())
    print(f"\nCritical Models (Gap > 0.05): {len(df_plot[df_plot['severity'] == 'Critical'])}")
    print(f"Warning Models (0.02 < Gap ≤ 0.05): {len(df_plot[df_plot['severity'] == 'Warning'])}")
    
    return df_plot

plot_accuracy_vs_auc_threshold_improved(results_combined_df)

# %% [markdown]
# ### Model Family Performance Comparison

# %%
def plot_grouped_performance(results_df):
    """Boxplot by model group"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Filter out SMOTE variants for clearer comparison
    df_no_smote = results_df[~results_df['model'].str.contains('SMOTE')]
    
    sns.boxplot(data=df_no_smote, x='Group', y='test_acc', 
               palette='Set3', ax=axes[0])
    axes[0].set_xlabel('Model Group', fontsize=11)
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0].set_title('Test Accuracy by Model Group', fontsize=13, weight='bold')
    axes[0].set_ylim(0.4, 0.75)
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df_no_smote, x='Group', y='test_auc', 
               palette='Set3', ax=axes[1])
    axes[1].set_xlabel('Model Group', fontsize=11)
    axes[1].set_ylabel('Test AUC', fontsize=11)
    axes[1].set_title('Test AUC by Model Group', fontsize=13, weight='bold')
    axes[1].set_ylim(0.4, 0.75)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('fig_group_performance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_grouped_performance(results_combined_df)


# %% [markdown]
# ### SMOTE vs No-SMOTE

# %%
def quantify_smote_effect(models_dict, X_train, X_train_smote, X_test, 
                          y_train, y_train_smote, y_test):
    """
    Systematic comparison of SMOTE vs standard training efficacy across multiple metrics.
    """
    smote_comparison = []
    
    for model_name, model in models_dict.items():
        if any(x in model_name for x in ['LDA', 'QDA', 'SMOTE']): continue
        
        # Helper to train and evaluate
        def evaluate_model(X, y):
            m = clone(model)
            m.fit(X, y)
            y_pred = m.predict(X_test)
            y_proba = m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') else y_pred
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            return {
                'Acc': accuracy_score(y_test, y_pred),
                'AUC': roc_auc_score(y_test, y_proba),
                'MCC': matthews_corrcoef(y_test, y_pred),
                'PR_AUC': auc(recall, precision)
            }

        # Train both versions
        metrics_std = evaluate_model(X_train, y_train)
        metrics_smote = evaluate_model(X_train_smote, y_train_smote)
        
        # Calculate deltas and compile results
        result_entry = {
            'Model': model_name, 
            'Group': assign_model_group(model_name)
        }
        
        for metric in ['Acc', 'AUC', 'MCC', 'PR_AUC']:
            delta = metrics_smote[metric] - metrics_std[metric]
            pct_change = (delta / metrics_std[metric] * 100) if metrics_std[metric] != 0 else 0
            
            result_entry.update({
                f'Standard_{metric}': metrics_std[metric],
                f'SMOTE_{metric}': metrics_smote[metric],
                f'Delta_{metric}': delta,
                f'Pct_Change_{metric}': pct_change
            })
            
        smote_comparison.append(result_entry)
    
    smote_df = pd.DataFrame(smote_comparison)
    
    # Visualization: 2x2 Grid (Top Benefit/Harm for MCC, Group Effects for MCC & AUC)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plotting helper
    def plot_bar(ax, data, val_col, label_col, title, color_logic=None):
        colors = color_logic if color_logic else ('green' if 'Benefit' in title else 'red')
        ax.barh(range(len(data)), data[val_col], color=colors, alpha=0.7)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data[label_col].index if label_col is None else data[label_col], fontsize=9)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    # 1. Top 10 Benefited (MCC)
    top_benefit = smote_df.nlargest(10, 'Delta_MCC')
    plot_bar(axes[0, 0], top_benefit, 'Delta_MCC', 'Model', 'Top 10: SMOTE Benefited Most (MCC)')

    # 2. Top 10 Harmed (MCC)
    top_harm = smote_df.nsmallest(10, 'Delta_MCC')
    plot_bar(axes[0, 1], top_harm, 'Delta_MCC', 'Model', 'Top 10: SMOTE Harmed Most (MCC)')

    # 3 & 4. Group Effects (MCC and AUC)
    for i, metric in enumerate(['MCC', 'AUC']):
        group_fx = smote_df.groupby('Group')[f'Delta_{metric}'].mean().sort_values()
        colors = ['green' if x > 0 else 'red' for x in group_fx]
        plot_bar(axes[1, i], group_fx, None, None, f'SMOTE Effect by Model Family ({metric})', colors)

    plt.tight_layout()
    plt.savefig('fig_smote_comprehensive_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary Report
    print(f"\n{'='*80}\nSMOTE IMPACT ANALYSIS\n{'='*80}")
    cols = ['Model', 'Delta_MCC', 'Pct_Change_MCC', 'Delta_PR_AUC', 'Delta_Acc']
    print("\nTop 5 Models Benefited (MCC):")
    print(smote_df.nlargest(5, 'Delta_MCC')[cols].to_string(index=False))
    print("\nTop 5 Models Harmed (MCC):")
    print(smote_df.nsmallest(5, 'Delta_MCC')[cols].to_string(index=False))
    
    print("\nGroup-Level Summary:")
    print(smote_df.groupby('Group')[['Delta_MCC', 'Delta_PR_AUC', 'Delta_Acc']].agg(['mean', 'std']).round(4))
    
    smote_df.to_csv('smote_impact_analysis.csv', index=False)
    print("\n✓ Analysis exported to 'smote_impact_analysis.csv'")
    
    return smote_df

# Execution
# smote_impact_df = quantify_smote_effect(models, X_train_scaled, X_train_smote, 
#                                         X_test_scaled, y_train, y_train_smote, y_test)


# %% [markdown]
# # Nested Cross-Validation

# %%
def nested_cross_validation(model, X, y, inner_cv=5, outer_cv=5, param_grid=None, 
                            scoring_metric='roc_auc', n_jobs=-1):
    """
    Improved nested CV with:
    - Configurable inner scoring metric (use MCC for imbalanced data)
    - 95% confidence intervals
    - Hyperparameter stability tracking
    - Statistical comparison to baseline
    - Proper handling of failed folds
    """
    from sklearn.model_selection import GridSearchCV
    from scipy import stats
    
    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_STATE)
    
    outer_scores = {
        'test_accuracy': [],
        'test_auc': [],
        'test_f1': [],
        'test_mcc': [],
        'test_pr_auc': [],
        'test_bacc': [],
        'test_sensitivity': [],
        'test_specificity': [],
        'test_precision': [],
        'best_params': [],
        'n_selected_features': []
    }
    
    fold_idx = 0
    for train_idx, test_idx in outer_skf.split(X, y):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        try:
            if param_grid:
                inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, 
                                           random_state=RANDOM_STATE + fold_idx)
                
                # Map scoring_metric to appropriate scorer
                if scoring_metric == 'mcc':
                    scorer = make_scorer(matthews_corrcoef)
                elif scoring_metric == 'roc_auc':
                    scorer = 'roc_auc'
                elif scoring_metric == 'f1':
                    scorer = 'f1'
                else:
                    scorer = scoring_metric
                
                grid_search = RandomizedSearchCV(
                    model, 
                    param_grid, 
                    n_iter=10, # Number of parameter settings sampled
                    cv=inner_skf, 
                    scoring=scorer,
                    n_jobs=n_jobs,
                    return_train_score=False
                )
                grid_search.fit(X_train_outer, y_train_outer)
                best_model = grid_search.best_estimator_
                outer_scores['best_params'].append(grid_search.best_params_)
            else:
                best_model = clone(model)
                best_model.fit(X_train_outer, y_train_outer)
                outer_scores['best_params'].append({})
            
            y_pred = best_model.predict(X_test_outer)
            
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test_outer)[:, 1]
            else:
                y_proba = best_model.decision_function(X_test_outer)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            
            tn, fp, fn, tp = confusion_matrix(y_test_outer, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            outer_scores['test_accuracy'].append(accuracy_score(y_test_outer, y_pred))
            outer_scores['test_auc'].append(roc_auc_score(y_test_outer, y_proba))
            outer_scores['test_f1'].append(f1_score(y_test_outer, y_pred))
            outer_scores['test_mcc'].append(matthews_corrcoef(y_test_outer, y_pred))
            outer_scores['test_pr_auc'].append(average_precision_score(y_test_outer, y_proba))
            outer_scores['test_bacc'].append(balanced_accuracy_score(y_test_outer, y_pred))
            outer_scores['test_sensitivity'].append(sensitivity)
            outer_scores['test_specificity'].append(specificity)
            outer_scores['test_precision'].append(precision)
            
            if hasattr(best_model, 'coef_'):
                n_selected = np.sum(np.abs(best_model.coef_[0]) > 1e-5)
                outer_scores['n_selected_features'].append(n_selected)
            else:
                outer_scores['n_selected_features'].append(X.shape[1])
                
        except Exception as e:
            print(f"    Fold {fold_idx} failed: {str(e)[:50]}")
            continue
        
        fold_idx += 1
    
    if len(outer_scores['test_mcc']) == 0:
        return None
    
    def calculate_ci(scores, confidence=0.95):
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)
        ci = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
        return mean, std_err, (mean - ci, mean + ci)
    
    mcc_mean, mcc_se, mcc_ci = calculate_ci(outer_scores['test_mcc'])
    pr_auc_mean, pr_auc_se, pr_auc_ci = calculate_ci(outer_scores['test_pr_auc'])
    bacc_mean, bacc_se, bacc_ci = calculate_ci(outer_scores['test_bacc'])
    
    from collections import Counter
    param_stability = {}
    if outer_scores['best_params']:
        for param_name in outer_scores['best_params'][0].keys():
            param_values = [p[param_name] for p in outer_scores['best_params']]
            most_common = Counter(param_values).most_common(1)[0]
            param_stability[param_name] = {
                'most_common': most_common[0],
                'frequency': most_common[1] / len(param_values)
            }
    
    return {
        'mean_accuracy': np.mean(outer_scores['test_accuracy']),
        'std_accuracy': np.std(outer_scores['test_accuracy']),
        'mean_auc': np.mean(outer_scores['test_auc']),
        'std_auc': np.std(outer_scores['test_auc']),
        'mean_f1': np.mean(outer_scores['test_f1']),
        'std_f1': np.std(outer_scores['test_f1']),
        'mean_mcc': mcc_mean,
        'std_mcc': np.std(outer_scores['test_mcc']),
        'se_mcc': mcc_se,
        'ci_mcc_lower': mcc_ci[0],
        'ci_mcc_upper': mcc_ci[1],
        'mean_pr_auc': pr_auc_mean,
        'std_pr_auc': np.std(outer_scores['test_pr_auc']),
        'se_pr_auc': pr_auc_se,
        'ci_pr_auc_lower': pr_auc_ci[0],
        'ci_pr_auc_upper': pr_auc_ci[1],
        'mean_bacc': bacc_mean,
        'std_bacc': np.std(outer_scores['test_bacc']),
        'se_bacc': bacc_se,
        'ci_bacc_lower': bacc_ci[0],
        'ci_bacc_upper': bacc_ci[1],
        'mean_sensitivity': np.mean(outer_scores['test_sensitivity']),
        'mean_specificity': np.mean(outer_scores['test_specificity']),
        'mean_precision': np.mean(outer_scores['test_precision']),
        'mean_n_features': np.mean(outer_scores['n_selected_features']),
        'std_n_features': np.std(outer_scores['n_selected_features']),
        'param_stability': param_stability,
        'all_params': outer_scores['best_params'],
        'n_folds_completed': len(outer_scores['test_mcc'])
    }




# %%
print("\n" + "="*80)
print("NESTED CV - COMPREHENSIVE MODEL COMPARISON")
print("="*80)

hyperparameter_grids = {
    # LINEAR MODELS (Interpretable)
    'Ridge': {
        'model': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, 
                                   random_state=RANDOM_STATE),
        'params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': [None, 'balanced']
        }
    },
    'Lasso': {
        'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, 
                                   random_state=RANDOM_STATE),
        'params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced']
        }
    },
    'ElasticNet': {
        'model': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, 
                                    tol=1e-3, warm_start=True, random_state=RANDOM_STATE),
        'params': {
            'C': [0.01, 0.1, 1.0], # can add 10.0
            'l1_ratio': [0.5, 0.7], # can add 0.3, 0.9
            'class_weight': [None, 'balanced']
        }
    },
    
    # NON-LINEAR MODELS (Predictive)
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    },
    
    'XGBoost': {
        'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, 
                                   eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0],
            'scale_pos_weight': [1, len(y_train[y_train==0])/len(y_train[y_train==1])]
        }
    },
    
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'params': {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    
    'QDA': {
        'model': QuadraticDiscriminantAnalysis(),
        'params': {
            'reg_param': [0.0, 0.1, 0.3, 0.5, 0.7]
        }
    },
    
    'SVM_RBF': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': [None, 'balanced']
        }
    },
    
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    },
    
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7, 1.0]
        }
    }
}


nested_cv_results = []

for model_name, config in hyperparameter_grids.items():
    print(f"\nNested CV: {model_name}...", end=" ")
    
    try:
        metrics = nested_cross_validation(
            config['model'], 
            X_train_scaled, 
            y_train, 
            inner_cv=5, 
            outer_cv=5, 
            param_grid=config['params'],
            n_jobs=-1
        )
        
        if metrics is None:
            print(f"✗ All folds failed")
            continue
            
        metrics['model'] = model_name
        metrics['model_type'] = 'Linear' if model_name in ['Ridge', 'Lasso', 'ElasticNet'] else 'Non-Linear'
        nested_cv_results.append(metrics)
        
        print(f"✓ MCC: {metrics['mean_mcc']:.3f} [{metrics['ci_mcc_lower']:.3f}, {metrics['ci_mcc_upper']:.3f}], "
              f"PR-AUC: {metrics['mean_pr_auc']:.3f}")
            
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")



# %%
if nested_cv_results:
    nested_cv_results_df = pd.DataFrame(nested_cv_results)
    
    print("\n" + "="*80)
    print("RESULTS BY MODEL TYPE")
    print("="*80)
    
    for model_type in ['Linear', 'Non-Linear']:
        type_df = nested_cv_results_df[nested_cv_results_df['model_type'] == model_type]
        if len(type_df) > 0:
            print(f"\n{model_type} Models:")
            print(f"  Best MCC: {type_df['mean_mcc'].max():.3f} ({type_df.loc[type_df['mean_mcc'].idxmax(), 'model']})")
            print(f"  Best PR-AUC: {type_df['mean_pr_auc'].max():.3f} ({type_df.loc[type_df['mean_pr_auc'].idxmax(), 'model']})")
    
    print("\n" + "="*80)
    print("TOP 10 MODELS (RANKED BY MCC)")
    print("="*80)
    
    top10 = nested_cv_results_df.nlargest(10, 'mean_mcc')[
        ['model', 'model_type', 'mean_mcc', 'ci_mcc_lower', 'ci_mcc_upper', 
         'mean_pr_auc', 'mean_bacc', 'mean_n_features']
    ]
    print(top10.to_string(index=False))


# %%
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON (NESTED CV)")
print("="*80)

comparison_df = nested_cv_results_df[[
    'model', 'model_type', 
    'mean_mcc', 'ci_mcc_lower', 'ci_mcc_upper',
    'mean_pr_auc', 'ci_pr_auc_lower', 'ci_pr_auc_upper',
    'mean_bacc', 'mean_sensitivity', 'mean_specificity',
    'mean_n_features', 'n_folds_completed'
]].copy()

comparison_df['mcc_significant'] = comparison_df['ci_mcc_lower'] > 0
comparison_df = comparison_df.sort_values('mean_mcc', ascending=False)

print(comparison_df.to_string(index=False))

best_overall = comparison_df.iloc[0]
best_linear = comparison_df[comparison_df['model_type'] == 'Linear'].iloc[0]

print(f"\n✓ Best Overall: {best_overall['model']} (MCC={best_overall['mean_mcc']:.3f})")
print(f"✓ Best Linear: {best_linear['model']} (MCC={best_linear['mean_mcc']:.3f})")
print(f"✓ Performance Gap: {(best_overall['mean_mcc'] - best_linear['mean_mcc']):.3f}")

# %% [markdown]
# ## Statistical Significance Testing

# %%
print("\n" + "="*80)
print("PAIRWISE STATISTICAL COMPARISONS (TOP 5 MODELS)")
print("="*80)

top5 = comparison_df.head(5)
significance_matrix = np.ones((len(top5), len(top5)))

for i in range(len(top5)):
    for j in range(i+1, len(top5)):
        model_i = top5.iloc[i]
        model_j = top5.iloc[j]
        
        ci_i = (model_i['ci_mcc_lower'], model_i['ci_mcc_upper'])
        ci_j = (model_j['ci_mcc_lower'], model_j['ci_mcc_upper'])
        
        ci_overlap = ci_i[0] <= ci_j[1] and ci_j[0] <= ci_i[1]
        significance_matrix[i, j] = 1 if ci_overlap else 0
        significance_matrix[j, i] = significance_matrix[i, j]
        
        print(f"\n{model_i['model']} vs {model_j['model']}:")
        print(f"  MCC: {model_i['mean_mcc']:.3f} vs {model_j['mean_mcc']:.3f}")
        print(f"  Difference: {abs(model_i['mean_mcc'] - model_j['mean_mcc']):.3f}")
        print(f"  95% CI Overlap: {'Yes (no sig diff)' if ci_overlap else 'No (significant)'}")


# %%
def assign_model_group(name):
    """Assign models to categories for grouped analysis"""
    if 'SMOTE' in name:
        return 'SMOTE Variants'
    elif any(k in name for k in ['LR_', 'Ridge_Classifier', 'SGD_Classifier']):
        return 'Linear / Penalized'
    elif 'Decision_Tree' in name and not any(x in name for x in ['Random', 'Extra']):
        return 'Single Tree'
    elif any(k in name for k in ['Random_Forest', 'Extra_Trees']):
        return 'Random Forest / ExtraTrees'
    elif any(k in name for k in ['AdaBoost', 'GradientBoosting', 'XGBoost']):
        return 'Boosting Ensembles'
    elif 'SVM_' in name:
        return 'SVM'
    elif 'KNN_' in name:
        return 'KNN'
    elif 'NN_' in name:
        return 'Neural Networks'
    elif name in ['LDA', 'QDA']:
        return 'Discriminant Analysis'
    elif name == 'Naive_Bayes':
        return 'Naive Bayes'
    else:
        return 'Other'

results_combined_df['Group'] = results_combined_df['model'].apply(assign_model_group)


# %% [markdown]
# ## VISUALIZATIONS

# %% [markdown]
# ### With Confidence Intervals

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

top_n = 10
top_models = comparison_df.head(top_n)

# MCC Plot
ax = axes[0]
y_pos = np.arange(len(top_models))
colors = ['#2E86AB' if t == 'Linear' else '#A23B72' for t in top_models['model_type']]

ax.barh(y_pos, top_models['mean_mcc'], color=colors, alpha=0.7)
ax.errorbar(top_models['mean_mcc'], y_pos, 
            xerr=[top_models['mean_mcc'] - top_models['ci_mcc_lower'],
                  top_models['ci_mcc_upper'] - top_models['mean_mcc']],
            fmt='none', color='black', capsize=5, linewidth=2)

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_models['model'])
ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=11)
ax.set_title('Top 10 Models by MCC (95% CI)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# PR-AUC Plot
ax = axes[1]
ax.barh(y_pos, top_models['mean_pr_auc'], color=colors, alpha=0.7)
ax.errorbar(top_models['mean_pr_auc'], y_pos,
            xerr=[top_models['mean_pr_auc'] - top_models['ci_pr_auc_lower'],
                  top_models['ci_pr_auc_upper'] - top_models['mean_pr_auc']],
            fmt='none', color='black', capsize=5, linewidth=2)

baseline_pr = sum(y_train) / len(y_train)
ax.axvline(baseline_pr, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
           label=f'Baseline ({baseline_pr:.2f})')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_models['model'])
ax.set_xlabel('Precision-Recall AUC', fontsize=11)
ax.set_title('Top 10 Models by PR-AUC (95% CI)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Balanced Accuracy
ax = axes[2]
ax.barh(y_pos, top_models['mean_bacc'], color=colors, alpha=0.7)
ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_models['model'])
ax.set_xlabel('Balanced Accuracy', fontsize=11)
ax.set_title('Top 10 Models by Balanced Accuracy', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2E86AB', alpha=0.7, label='Linear'),
                   Patch(facecolor='#A23B72', alpha=0.7, label='Non-Linear')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison_metrics.png")
plt.show()

# %% [markdown]
# ### Interpretability and Trade-offs

# %%
fig, ax = plt.subplots(figsize=(10, 7))

linear_models = comparison_df[comparison_df['model_type'] == 'Linear']
nonlinear_models = comparison_df[comparison_df['model_type'] == 'Non-Linear']

ax.scatter(linear_models['mean_n_features'], linear_models['mean_mcc'], 
           s=200, alpha=0.7, color='#2E86AB', marker='o', 
           edgecolors='black', linewidth=1.5, label='Linear')

ax.scatter(nonlinear_models['mean_n_features'], nonlinear_models['mean_mcc'], 
           s=200, alpha=0.7, color='#A23B72', marker='s', 
           edgecolors='black', linewidth=1.5, label='Non-Linear')

for idx, row in comparison_df.head(15).iterrows():
    ax.annotate(row['model'], 
                (row['mean_n_features'], row['mean_mcc']),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.8)

ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Number of Features Used', fontsize=12)
ax.set_ylabel('MCC (Predictive Performance)', fontsize=12)
ax.set_title('Interpretability-Performance Trade-off\n(Fewer Features = More Interpretable)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('interpretability_performance_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: interpretability_performance_tradeoff.png")
plt.show()

# %% [markdown]
# ### Sensitivity-Specificity Balanced

# %%
fig, ax = plt.subplots(figsize=(10, 8))

for idx, row in comparison_df.head(15).iterrows():
    color = '#2E86AB' if row['model_type'] == 'Linear' else '#A23B72'
    marker = 'o' if row['model_type'] == 'Linear' else 's'
    
    ax.scatter(row['mean_specificity'], row['mean_sensitivity'], 
               s=300, alpha=0.7, color=color, marker=marker,
               edgecolors='black', linewidth=1.5)
    
    ax.annotate(row['model'], 
                (row['mean_specificity'], row['mean_sensitivity']),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=9, alpha=0.8)

ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='Random Classifier')
ax.plot([0, 1], [1, 0], 'g--', linewidth=1, alpha=0.3)

ax.set_xlabel('Specificity (TNR)', fontsize=12)
ax.set_ylabel('Sensitivity (TPR/Recall)', fontsize=12)
ax.set_title('Sensitivity-Specificity Balance (Top 15 Models)', 
             fontsize=13, fontweight='bold')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', alpha=0.7, label='Linear'),
    Patch(facecolor='#A23B72', alpha=0.7, label='Non-Linear')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

plt.tight_layout()
plt.savefig('sensitivity_specificity_balance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sensitivity_specificity_balance.png")
plt.show()


# %% [markdown]
# ### Model Type Comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = [
    ('mean_mcc', 'MCC'),
    ('mean_pr_auc', 'PR-AUC'),
    ('mean_bacc', 'Balanced Accuracy')
]

for idx, (metric, label) in enumerate(metrics_to_plot):
    ax = axes[idx]
    
    data_to_plot = [
        comparison_df[comparison_df['model_type'] == 'Linear'][metric].values,
        comparison_df[comparison_df['model_type'] == 'Non-Linear'][metric].values
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Linear', 'Non-Linear'],
                    patch_artist=True, widths=0.6)
    
    colors = ['#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    if metric == 'mean_mcc':
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label} by Model Type', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    linear_median = np.median(data_to_plot[0])
    nonlinear_median = np.median(data_to_plot[1])
    
    _, p_value = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1], alternative='two-sided')
    
    y_max = max(np.max(data_to_plot[0]), np.max(data_to_plot[1]))
    ax.text(1.5, y_max * 0.95, f'p={p_value:.3f}', 
            ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('model_type_comparison_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_type_comparison_boxplot.png")
plt.show()

# %% [markdown]
# ### Model Family Performance Comparison

# %%
plot_grouped_performance(comparison_df)


# %% [markdown]
# ### SMOTE vs No-SMOTE

# %%
quantify_smote_effect(comparison_df)

# %% [markdown]
# ## Train Best Models with Stable Features (For Interpretability) 

# %%
print("\n" + "="*80)
print("STABILITY SELECTION: LINEAR vs NON-LINEAR COMPARISON")
print("="*80)

# 1. STABILITY SELECTION ON BEST LINEAR MODEL
best_linear_name = best_linear['model']  # e.g., ElasticNet
print(f"\n1. Linear Model: {best_linear_name}")

if 'Lasso' in best_linear_name:
    penalty = 'l1'
    l1_ratio = None
elif 'ElasticNet' in best_linear_name:
    penalty = 'elasticnet'
    l1_ratio = 0.5
else:
    penalty = 'l2'
    l1_ratio = None

def stability_selection_linear(X, y, penalty='l1', l1_ratio=0.5, n_bootstraps=100, 
                               threshold=0.7, C_range=None):
    """Bootstrap stability for linear models"""
    n_samples, n_features = X.shape
    if C_range is None:
        C_range = np.logspace(-3, 2, 20)
    
    selection_matrix = np.zeros((n_bootstraps, n_features))
    
    for b in range(n_bootstraps):
        sample_idx = resample(np.arange(n_samples), n_samples=n_samples//2, 
                              replace=False, random_state=b)
        X_boot = X[sample_idx]
        y_boot = y.iloc[sample_idx]
        
        for C in C_range:
            if penalty == 'l1':
                model = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                                          max_iter=10000, random_state=b)
            elif penalty == 'elasticnet':
                model = LogisticRegression(penalty='elasticnet', C=C, 
                                          l1_ratio=l1_ratio, solver='saga', 
                                          max_iter=10000, random_state=b)
            elif penalty == 'l2':
                model = LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                          max_iter=10000, random_state=b)
            
            model.fit(X_boot, y_boot)
            selected = np.abs(model.coef_[0]) > 1e-5
            selection_matrix[b, :] = np.maximum(selection_matrix[b, :], selected)
    
    selection_freq = selection_matrix.mean(axis=0)
    stable_features = np.where(selection_freq >= threshold)[0]
    
    return selection_freq, stable_features

selection_freq_linear, stable_features_linear = stability_selection_linear(
    X_train_scaled, y_train, 
    penalty=penalty, l1_ratio=l1_ratio,
    n_bootstraps=100, threshold=0.7
)

print(f"   Stable features (π≥0.7): {len(stable_features_linear)}")
print(f"   Mean selection frequency: {selection_freq_linear[stable_features_linear].mean():.3f}")


# 2. STABILITY SELECTION ON BEST NON-LINEAR MODEL
best_nonlinear_name = best_overall['model']  # e.g., XGBoost or KNN
print(f"\n2. Non-Linear Model: {best_nonlinear_name}")

def stability_selection_nonlinear(model_template, X, y, n_bootstraps=100, 
                                  threshold=0.7, n_top_features=50):
    """Bootstrap stability for tree-based/ensemble models using feature importance"""
    n_samples, n_features = X.shape
    selection_matrix = np.zeros((n_bootstraps, n_features))
    
    for b in range(n_bootstraps):
        # Bootstrap sample
        sample_idx = resample(np.arange(n_samples), n_samples=n_samples//2, 
                              replace=False, random_state=b)
        X_boot = X[sample_idx]
        y_boot = y.iloc[sample_idx]
        
        # Fit model and get feature importances
        model_boot = clone(model_template)
        model_boot.fit(X_boot, y_boot)
        
        if hasattr(model_boot, 'feature_importances_'):
            importances = model_boot.feature_importances_
        elif hasattr(model_boot, 'coef_'):
            importances = np.abs(model_boot.coef_[0])
        else:
            continue
        
        # Select top n_top_features by importance
        top_idx = np.argsort(importances)[-n_top_features:]
        selection_matrix[b, top_idx] = 1
    
    selection_freq = selection_matrix.mean(axis=0)
    stable_features = np.where(selection_freq >= threshold)[0]
    
    return selection_freq, stable_features

# Get the actual best non-linear model instance
if best_nonlinear_name in models:
    best_nonlinear_model = models[best_nonlinear_name]
else:
    # If it's from nested CV, instantiate it
    best_nonlinear_model = nested_cv_results_df[
        nested_cv_results_df['model'] == best_nonlinear_name
    ].iloc[0]
    # Use a default model of that type
    if 'XGBoost' in best_nonlinear_name:
        best_nonlinear_model = XGBClassifier(max_depth=3, n_estimators=100,
                                            eval_metric='logloss', random_state=RANDOM_STATE)
    elif 'KNN' in best_nonlinear_name:
        best_nonlinear_model = KNeighborsClassifier(n_neighbors=5)
    else:
        best_nonlinear_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

selection_freq_nonlinear, stable_features_nonlinear = stability_selection_nonlinear(
    best_nonlinear_model, X_train_scaled, y_train,
    n_bootstraps=100, threshold=0.7, n_top_features=50
)

print(f"   Stable features (π≥0.7): {len(stable_features_nonlinear)}")
print(f"   Mean selection frequency: {selection_freq_nonlinear[stable_features_nonlinear].mean():.3f}")


# 3. VISUALIZE BOTH
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Linear model stability
top_30_linear = np.argsort(selection_freq_linear)[-30:][::-1]
colors_linear = ['#27AE60' if selection_freq_linear[i] >= 0.7 else '#E74C3C' 
                 for i in top_30_linear]
axes[0].barh(range(len(top_30_linear)), selection_freq_linear[top_30_linear], 
            color=colors_linear, alpha=0.7, edgecolor='black')
axes[0].axvline(0.7, color='blue', linestyle='--', linewidth=2, label='Threshold (0.7)')
axes[0].set_xlabel('Selection Frequency', fontsize=11)
axes[0].set_title(f'{best_linear_name}\n(Linear Model)', fontsize=12, weight='bold')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Non-linear model stability
top_30_nonlinear = np.argsort(selection_freq_nonlinear)[-30:][::-1]
colors_nonlinear = ['#27AE60' if selection_freq_nonlinear[i] >= 0.7 else '#E74C3C' 
                    for i in top_30_nonlinear]
axes[1].barh(range(len(top_30_nonlinear)), selection_freq_nonlinear[top_30_nonlinear], 
            color=colors_nonlinear, alpha=0.7, edgecolor='black')
axes[1].axvline(0.7, color='blue', linestyle='--', linewidth=2, label='Threshold (0.7)')
axes[1].set_xlabel('Selection Frequency', fontsize=11)
axes[1].set_title(f'{best_nonlinear_name}\n(Non-Linear Model)', fontsize=12, weight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('stability_selection_linear_vs_nonlinear.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. COMPARE FEATURE AGREEMENT
common_stable = set(stable_features_linear) & set(stable_features_nonlinear)
only_linear = set(stable_features_linear) - set(stable_features_nonlinear)
only_nonlinear = set(stable_features_nonlinear) - set(stable_features_linear)

print(f"\n" + "="*80)
print("FEATURE SELECTION AGREEMENT")
print("="*80)
print(f"Common stable features: {len(common_stable)}")
print(f"Only in linear model: {len(only_linear)}")
print(f"Only in non-linear model: {len(only_nonlinear)}")
print(f"Agreement ratio: {len(common_stable) / max(len(stable_features_linear), len(stable_features_nonlinear)) * 100:.1f}%")

# %%
# Bootstrap stability analysis
from collections import Counter

def bootstrap_feature_stability(model, X, y, n_bootstrap=50, n_features=13):
    """Measure feature selection stability across bootstrap samples"""
    feature_selections = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]
        
        # Fit model
        model.fit(X_boot, y_boot)
        
        # Get top features
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue
            
        top_idx = np.argsort(importances)[-n_features:]
        top_features = [feature_names[i] for i in top_idx]
        feature_selections.extend(top_features)
    
    # Calculate stability (Jaccard index)
    feature_freq = Counter(feature_selections)
    total_selections = n_bootstrap * n_features
    
    # Features selected in >50% of bootstraps are "stable"
    stable_features = {f: count for f, count in feature_freq.items() 
                      if count >= n_bootstrap * 0.5}
    
    return feature_freq, stable_features

# Test on Lasso
freq, stable = bootstrap_feature_stability(
    models['LR_Lasso_C0.1'], X_train_scaled, y_train, 
    n_bootstrap=50, n_features=13
)

print(f"\nFeature Selection Stability Analysis (50 bootstrap iterations):")
print(f"Total unique features selected: {len(freq)}")
print(f"Stable features (selected in >50% of bootstraps): {len(stable)}")
print(f"\nTop 10 most stable features:")
for feat, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
    stability_pct = (count / 50) * 100
    print(f"  {feat}: {stability_pct:.1f}% selection rate")


# %%



