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
# # IMPORTS

# %%
pip install ucimlrepo

# %%
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_val_predict, learning_curve, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, SGDClassifier
import warnings
warnings.filterwarnings('ignore')

# %%
# If you use McNemar test (optional, comment out if not using)
# from mlxtend.evaluate import mcnemar_table, mcnemar

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
# # DATA LOAD AND PREPROCESSING

# %%
data = fetch_ucirepo(id=728)
X = data.data.features
y = data.data.targets

y_binary = (y['Class'] == 'NonToxic').astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution:\n{y['Class'].value_counts()}")
print(f"Class balance:\n{y['Class'].value_counts(normalize=True)}")

# %%
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X_shuffled = X.iloc[shuffle_idx].reset_index(drop=True)
y_shuffled = y_binary.iloc[shuffle_idx].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.2,
    random_state=42, stratify=y_shuffled
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

# %%
feature_names = X.columns.tolist()

# %% [markdown]
# # MODEL DEFINITIONS

# %%
models = {
    # Linear / Penalized
    'LR_No_Penalty': LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs'),
    'LR_Ridge_C1': LogisticRegression(penalty='l2', C=1.0, max_iter=2000, solver='lbfgs'),
    'LR_Ridge_C0.1': LogisticRegression(penalty='l2', C=0.1, max_iter=2000, solver='lbfgs'),
    'LR_Ridge_C10': LogisticRegression(penalty='l2', C=10.0, max_iter=2000, solver='lbfgs'),
    'LR_Lasso_C1': LogisticRegression(penalty='l1', C=1.0, max_iter=2000, solver='saga'),
    'LR_Lasso_C0.1': LogisticRegression(penalty='l1', C=0.1, max_iter=2000, solver='saga'),
    'LR_ElasticNet_L1_0.5': LogisticRegression(penalty='elasticnet', solver='saga',
                                               l1_ratio=0.5, C=1.0, max_iter=2000),
    'LR_ElasticNet_L1_0.7': LogisticRegression(penalty='elasticnet', solver='saga',
                                               l1_ratio=0.7, C=1.0, max_iter=2000),
    'Ridge_Classifier': RidgeClassifier(alpha=1.0),
    'SGD_Classifier': SGDClassifier(loss='log_loss', max_iter=2000, random_state=42),

    # Discriminant Analysis
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),

    # Naive Bayes
    'Naive_Bayes': GaussianNB(),

    # Tree-based
    'Decision_Tree_D5': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Decision_Tree_D10': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Decision_Tree_D20': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Decision_Tree_Unpruned': DecisionTreeClassifier(random_state=42),

    # Bagging
    'Random_Forest_N50': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    'Random_Forest_N100': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Random_Forest_N200': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Random_Forest_Deep': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
    'Extra_Trees_N100': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),

    # Boosting
    'AdaBoost_N50': AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME'),
    'AdaBoost_N100': AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
    'GradientBoosting_N50': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    'GradientBoosting_N100': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'XGBoost_D3_N50': XGBClassifier(max_depth=3, n_estimators=50, random_state=42,
                                    eval_metric='logloss', use_label_encoder=False),
    'XGBoost_D3_N100': XGBClassifier(max_depth=3, n_estimators=100, random_state=42,
                                     eval_metric='logloss', use_label_encoder=False),
    'XGBoost_D5_N100': XGBClassifier(max_depth=5, n_estimators=100, random_state=42,
                                     eval_metric='logloss', use_label_encoder=False),

    # SVM
    'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
    'SVM_RBF_C1': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'SVM_RBF_C10': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
    'SVM_Poly_D2': SVC(kernel='poly', degree=2, probability=True, random_state=42),
    'SVM_Poly_D3': SVC(kernel='poly', degree=3, probability=True, random_state=42),

    # KNN
    'KNN_K3': KNeighborsClassifier(n_neighbors=3),
    'KNN_K5': KNeighborsClassifier(n_neighbors=5),
    'KNN_K7': KNeighborsClassifier(n_neighbors=7),
    'KNN_K10': KNeighborsClassifier(n_neighbors=10),

    # Neural Networks
    'NN_Small': MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, random_state=42,
                              early_stopping=True, solver='lbfgs'),
    'NN_Medium': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42,
                               early_stopping=True, solver='lbfgs'),
    'NN_Large': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
                              early_stopping=True, solver='lbfgs'),
    'NN_Deep': MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42,
                             early_stopping=True, solver='lbfgs'),
    'NN_Adam': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42,
                             early_stopping=True, solver='adam'),
}

# %% [markdown]
# # BASIC SINGLE SPLIT EVALUATION

# %%
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = model.decision_function(X_train)
        y_test_proba = model.decision_function(X_test)

    return {
        'train_acc': accuracy_score(y_train, y_train_pred),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred)
    }

print("\n" + "="*80)
print("TRAINING MODELS (SINGLE SPLIT)")
print("="*80)

results = []
for name, model in models.items():
    print(f"Training {name}...")
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    metrics['model'] = name
    results.append(metrics)
    print(f"  Test Accuracy: {metrics['test_acc']:.4f}, Test AUC: {metrics['test_auc']:.4f}")


# %%
results_df = pd.DataFrame(results)
results_df = results_df[['model', 'train_acc', 'test_acc', 'train_auc',
                         'test_auc', 'precision', 'recall', 'f1']]
results_df = results_df.sort_values('test_acc', ascending=False).reset_index(drop=True)

print("\n=== MODEL COMPARISON (SINGLE SPLIT) ===")
print(results_df.to_string(index=False))

# %% [markdown]
# # CROSS-VALIDATION EVALUATION

# %%
def evaluate_model_cv(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision', n_jobs=-1)
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1)

    return {
        'acc_mean': acc_scores.mean(),
        'acc_std': acc_scores.std(),
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'precision_mean': precision_scores.mean(),
        'precision_std': precision_scores.std(),
        'recall_mean': recall_scores.mean(),
        'recall_std': recall_scores.std()
    }

print("\n" + "="*80)
print("CROSS-VALIDATION (5-FOLD STRATIFIED)")
print("="*80)

cv_results = []
for name, model in models.items():
    print(f"Evaluating {name}...")
    metrics = evaluate_model_cv(model, X_train_scaled, y_train, cv=5)
    metrics['model'] = name
    cv_results.append(metrics)
    print(f"  Acc: {metrics['acc_mean']:.4f} ± {metrics['acc_std']:.4f}")
    print(f"  AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")

# %%
cv_results_df = pd.DataFrame(cv_results)
cv_results_df = cv_results_df.sort_values('acc_mean', ascending=False).reset_index(drop=True)

print("\n=== CROSS-VALIDATION RESULTS ===")
print(cv_results_df[['model', 'acc_mean', 'acc_std',
                     'auc_mean', 'auc_std', 'f1_mean', 'f1_std']].to_string(index=False))

# %% [markdown]
# # MODEL GROUPING

# %%
def assign_model_group(name):
    categories = {
        'Linear / Penalized': ['LR_', 'Ridge_Classifier', 'SGD_Classifier'],
        'Single Tree': ['Decision_Tree'],
        'Random Forest / ExtraTrees': ['Random_Forest', 'Extra_Trees'],
        'Boosting Ensembles': ['AdaBoost', 'GradientBoosting', 'XGBoost'],
        'SVM': ['SVM_'],
        'KNN': ['KNN_'],
        'Neural Networks': ['NN_'],
        'Discriminant Analysis': ['LDA', 'QDA'],
        'Naive Bayes': ['Naive_Bayes']
    }
    for group, keywords in categories.items():
        if any(k in name for k in keywords):
            return group
    return 'Other'

results_df['Group'] = results_df['model'].apply(assign_model_group)
cv_results_df['Group'] = cv_results_df['model'].apply(assign_model_group)

# %% [markdown]
# # FEATURE IMPORTANCE AND COMPARISON

# %%
original_features = ['MDEC-23', 'MATS2v', 'ATSC8s', 'VE3_Dt', 'CrippenMR', 'SpMax7_Bhe',
                     'SpMin1_Bhs', 'C1SP2', 'GATS8e', 'GATS8s', 'SpMax5_Bhv',
                     'VE3_Dzi', 'VPC-4']

# %%
def get_feature_importance(model, X_train, X_test, y_train, y_test):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        method = "Built-in"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        method = "Coefficients"
    else:
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=42, n_jobs=-1
        )
        importances = perm_importance.importances_mean
        method = "Permutation"
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    return importance_df, method

print("\n" + "="*80)
print("FEATURE IMPORTANCE COMPARISON")
print("="*80)
print(f"Original study features ({len(original_features)}):")
print(original_features)

# %%
feature_comparison = {}
for name, model in models.items():
    print(f"\n### {name} ###")
    # Ensure model is fitted
    model.fit(X_train_scaled, y_train)
    importance_df, method = get_feature_importance(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    top_13 = importance_df.head(13)
    top_13_features = top_13['feature'].tolist()
    overlap = set(top_13_features) & set(original_features)
    overlap_count = len(overlap)
    overlap_pct = (overlap_count / len(original_features)) * 100

    print(f"Method: {method}")
    print(f"Overlap with original: {overlap_count}/{len(original_features)} ({overlap_pct:.1f}%)")
    if overlap:
        print(f"Matching: {sorted(overlap)}")

    feature_comparison[name] = {
        'top_13': top_13_features,
        'overlap_count': overlap_count,
        'overlap_features': sorted(overlap),
        'method': method
    }

# %%
summary_df = pd.DataFrame({
    'Model': list(feature_comparison.keys()),
    'Overlap Count': [v['overlap_count'] for v in feature_comparison.values()],
    'Overlap %': [(v['overlap_count'] / len(original_features)) * 100
                  for v in feature_comparison.values()],
    'Method': [v['method'] for v in feature_comparison.values()]
}).sort_values('Overlap Count', ascending=False)

print("\n=== SUMMARY: OVERLAP WITH ORIGINAL STUDY ===")
print(summary_df.to_string(index=False))

# %% [markdown]
# # FEATURE CONSISTENCY

# %%
all_top_features = []
for comp in feature_comparison.values():
    all_top_features.extend(comp['top_13'])

feature_counts = pd.Series(all_top_features).value_counts()
frequent_features = feature_counts[feature_counts >= 3]

print("\n" + "="*80)
print("FEATURES SELECTED BY 3+ MODELS")
print("="*80)
if len(frequent_features) > 0:
    for feat, count in frequent_features.items():
        in_original = "✓" if feat in original_features else " "
        print(f"[{in_original}] {feat}: {count}/{len(models)} models")
else:
    print("No features selected by ≥3 models.")

comparison_results = []
for model_name, comp in feature_comparison.items():
    for i, feat in enumerate(comp['top_13'], 1):
        comparison_results.append({
            'model': model_name,
            'rank': i,
            'feature': feat,
            'in_original_study': feat in original_features
        })
comparison_df = pd.DataFrame(comparison_results)

# %% [markdown]
# # VISUALIZATIONS

# %% [markdown]
# ## PERFORMANCE

# %%
def plot_top_models(results_df, n=15):
    top_n = results_df.head(n).copy()
    plot_data = top_n.melt(
        id_vars=['model', 'Group'],
        value_vars=['test_acc', 'test_auc'],
        var_name='Metric',
        value_name='Score'
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(
        data=plot_data, y='model', x='Score', hue='Metric',
        dodge=True, ax=ax,
        palette={'test_acc': "#2bbbdf", 'test_auc': "#f03f3f"}
    )
    ax.set_title(f'Top {n} Models – Test Accuracy & AUC', fontsize=16, weight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Model')
    ax.set_xlim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Test Accuracy', 'Test AUC'],
              title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_top_models(results_df, n=15)

# %%
def plot_cv_performance(results_df, n=15):
    # Sort for first chart by test_acc, second by test_auc
    top_n_acc = results_df.sort_values('test_acc', ascending=False).head(n).copy()
    top_n_auc = results_df.sort_values('test_auc', ascending=False).head(n).copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Reverse order so top models are at the top
    top_n_acc = top_n_acc.iloc[::-1]
    top_n_auc = top_n_auc.iloc[::-1]

    ax1.barh(range(len(top_n_acc)), top_n_acc['test_acc'],
             color='#2bbbdf', alpha=0.7)
    ax1.set_yticks(range(len(top_n_acc)))
    ax1.set_yticklabels(top_n_acc['model'])
    ax1.set_xlabel('Test Accuracy')
    ax1.set_title('Test Accuracy (Sorted by Accuracy)')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)

    ax2.barh(range(len(top_n_auc)), top_n_auc['test_auc'],
             color='#f03f3f', alpha=0.7)
    ax2.set_yticks(range(len(top_n_auc)))
    ax2.set_yticklabels(top_n_auc['model'])
    ax2.set_xlabel('Test AUC')
    ax2.set_title('Test AUC (Sorted by AUC)')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_cv_performance(results_df, n=15)

# %%
def plot_grouped_boxplot(results_df, metric='test_acc'):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=results_df, x='Group', y=metric, palette='Set3', ax=ax)
    # ax.set_title(f'Model Performance by Group – {metric}', fontsize=16, weight='bold')
    ax.set_xlabel('Model Group')
    ax.set_ylabel(metric)
    ax.set_ylim(0.4, 0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_grouped_boxplot(results_df, metric='test_acc')

# %%
def plot_accuracy_vs_auc(results_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='test_acc', y='test_auc',
                    hue='Group', s=100, palette='tab10')
    plt.title('Test Accuracy vs Test AUC', weight='bold')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Test AUC')
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    plt.legend(title='Model Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_accuracy_vs_auc(results_df)

# %% [markdown]
# ## FEATURE CONSISTENCY

# %%
def plot_feature_consistency(feature_comparison, original_features, top_n=20):
    all_top_features = []
    for comp in feature_comparison.values():
        all_top_features.extend(comp['top_13'])
    feat_counts = pd.Series(all_top_features).value_counts().head(top_n)
    feat_df = feat_counts.reset_index()
    feat_df.columns = ['Feature', 'Frequency']
    feat_df['In_Original_Study'] = feat_df['Feature'].apply(
        lambda x: 'Yes' if x in original_features else 'No'
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Feature', data=feat_df,
                hue='In_Original_Study', dodge=False,
                palette={'Yes': 'green', 'No': 'gray'})
    plt.title(f'Top {top_n} Features Selected Across Models', fontsize=15)
    plt.xlabel('Number of Models')
    plt.legend(title='In Original Study?')
    plt.tight_layout()
    plt.show()

plot_feature_consistency(feature_comparison, original_features, top_n=20)

# %%
def plot_feature_overlap(summary_df):
    overlap_df = summary_df.sort_values('Overlap Count', ascending=True)
    plt.figure(figsize=(10, 12))
    sns.barplot(data=overlap_df, y='Model', x='Overlap Count', palette='Reds_d')
    plt.title('Overlap of Top-13 Features with Original Study', weight='bold')
    plt.xlabel('Matching Features (out of 13)')
    for i, row in enumerate(overlap_df.itertuples()):
        plt.text(row._2 + 0.05, i, f'{row._2}/13', va='center')
    plt.tight_layout()
    plt.show()

plot_feature_overlap(summary_df)

# %% [markdown]
# ## CROSS-VALIDATION STABILITY PLOT

# %%
def plot_cv_performance_with_errors(cv_results_df, n=15):
    # Sort for first chart by acc_mean, second by auc_mean
    top_n_acc = cv_results_df.sort_values('acc_mean', ascending=False).head(n).copy()
    top_n_auc = cv_results_df.sort_values('auc_mean', ascending=False).head(n).copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    ax1.barh(range(len(top_n_acc)), top_n_acc['acc_mean'],
             xerr=top_n_acc['acc_std'], capsize=5, color='#2bbbdf', alpha=0.7)
    ax1.set_yticks(range(len(top_n_acc)))
    ax1.set_yticklabels(top_n_acc['model'])
    ax1.set_xlabel('Accuracy (Mean ± Std)')
    ax1.set_title('CV Accuracy Stability (Sorted by Accuracy)')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)

    ax2.barh(range(len(top_n_auc)), top_n_auc['auc_mean'],
             xerr=top_n_auc['auc_std'], capsize=5, color='#f03f3f', alpha=0.7)
    ax2.set_yticks(range(len(top_n_auc)))
    ax2.set_yticklabels(top_n_auc['model'])
    ax2.set_xlabel('AUC (Mean ± Std)')
    ax2.set_title('CV AUC Stability (Sorted by AUC)')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_cv_performance_with_errors(cv_results_df, n=15)

# %% [markdown]
# ## OVERFITTING SCATTER (TRAIN VS TEST)

# %%
def plot_train_vs_test_scatter(results_df):
    plt.figure(figsize=(10, 8))
    for group in results_df['Group'].unique():
        group_data = results_df[results_df['Group'] == group]
        plt.scatter(group_data['train_acc'], group_data['test_acc'],
                    label=group, s=100, alpha=0.7)
    plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='Perfect Generalization')
    plt.xlabel('Training Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0.5, 1.05)
    plt.ylim(0.5, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    results_df['overfitting_gap'] = results_df['train_acc'] - results_df['test_acc']
    worst_overfitters = results_df.nlargest(5, 'overfitting_gap')[['model', 'overfitting_gap']]
    print("\nModels with highest overfitting:")
    print(worst_overfitters.to_string(index=False))
    print("\n" + "="*80)
    best_generalizers = results_df.nsmallest(5, 'overfitting_gap')[['model', 'overfitting_gap']]
    print("\nModels with lowest overfitting:")
    print(best_generalizers.to_string(index=False))

plot_train_vs_test_scatter(results_df)

# %% [markdown]
# ## LEARNING CURVES FOR TOP MODELS

# %%
def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=-1):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, shuffle=True,
        random_state=42, scoring='roc_auc'
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(f'Learning Curve - {title}', fontsize=14, weight='bold')
    plt.xlabel("Training Examples")
    plt.ylabel("AUC")
    plt.grid(alpha=0.3)
    plt.fill_between(train_sizes_abs, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes_abs, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training")
    plt.plot(train_sizes_abs, test_mean, 'o-', color="g", label="CV")
    plt.legend(loc="best")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    final_gap = train_mean[-1] - test_mean[-1]
    print(f"{title}: Overfitting gap = {final_gap:.4f}")

print("\n" + "="*80)
print("LEARNING CURVES - TOP 5 MODELS")
print("="*80)
top_5_names = cv_results_df.head(5)['model'].tolist()
for model_name in top_5_names:
    plot_learning_curve(models[model_name], model_name, X_train_scaled, y_train, cv=5)

# %% [markdown]
# ## PRECISION-RECALL CURVES (TOP MODELS)

# %%
def plot_precision_recall_curves(models_dict, model_names, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(10, 8))
    for name in model_names:
        model = models_dict[name]
        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (AUC={pr_auc:.3f})', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Top Models', fontsize=14, weight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

top_5_for_pr = cv_results_df.head(5)['model'].tolist()
plot_precision_recall_curves(models, top_5_for_pr,
                             X_train_scaled, X_test_scaled, y_train, y_test)

# %%


# %%


# %%



