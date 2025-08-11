# =============================================================================
# TASK 4: MODÈLES DE PRÉDICTION DU CHURN
# Objectif: Développer et évaluer des modèles ML pour prédire le churn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("TASK 4: MODÈLES DE PRÉDICTION DU CHURN")
print("="*60)

# 1. CHARGEMENT DES DONNÉES
print("\n1. CHARGEMENT DES DONNÉES")
print("-" * 40)

train_data = pd.read_csv('train_data_task1.csv')
test_data = pd.read_csv('test_data_task1.csv')

X_train = train_data.drop('Churn', axis=1)
y_train = train_data['Churn']
X_test = test_data.drop('Churn', axis=1)
y_test = test_data['Churn']

print(f"✓ Données d'entraînement: {train_data.shape}")
print(f"✓ Données de test: {test_data.shape}")

# 2. PRÉPARATION ET STANDARDISATION
print("\n2. PRÉPARATION DES FEATURES")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("✓ Features standardisées")

# 3. FEATURE SELECTION
print("\n3. SÉLECTION DES FEATURES")
print("-" * 40)

k_best = SelectKBest(score_func=f_classif, k=10)
X_train_selected = k_best.fit_transform(X_train_scaled_df, y_train)
X_test_selected = k_best.transform(X_test_scaled_df)

selected_features = X_train.columns[k_best.get_support()]
print(f"Top {len(selected_features)} features sélectionnées :")
for feat, score in zip(selected_features, k_best.scores_[k_best.get_support()]):
    print(f"- {feat}: {score:.2f}")

# 4. MODÈLES À TESTER
print("\n4. DÉFINITION DES MODÈLES")
print("-" * 40)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

print(f"Modèles à évaluer : {list(models.keys())}")

# 5. ENTRAÎNEMENT ET ÉVALUATION
print("\n5. ENTRAÎNEMENT ET OPTIMISATION")
print("-" * 40)

best_models = {}
model_results = {}

for name, m in models.items():
    print(f"\n--- {name} ---")
    gs = GridSearchCV(m['model'], m['params'], cv=5, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_selected, y_train)
    best_models[name] = gs.best_estimator_

    y_pred = gs.best_estimator_.predict(X_test_selected)
    y_proba = gs.best_estimator_.predict_proba(X_test_selected)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    model_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_proba
    }

    print(f"Meilleurs paramètres : {gs.best_params_}")
    print(f"ROC-AUC (CV) : {gs.best_score_:.4f}")
    print(f"Test - Acc: {accuracy:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# 6. COMPARAISON VISUELLE
print("\n6. COMPARAISON DES MODÈLES")
print("-" * 40)

results_df = pd.DataFrame(model_results).T[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
best_model_name = results_df['roc_auc'].idxmax()
best_model = best_models[best_model_name]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Bar chart
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
width = 0.15
x = np.arange(len(models))

for i, metric in enumerate(metrics):
    scores = [results_df.loc[m, metric] for m in models]
    axes[0,0].bar(x + i*width, scores, width, label=metric.title())

axes[0,0].set_xticks(x + width*2)
axes[0,0].set_xticklabels(models.keys(), rotation=45)
axes[0,0].set_title('Comparaison des métriques')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, model_results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title(f'Matrice de Confusion - {best_model_name}')
axes[0,1].set_xlabel("Prédit")
axes[0,1].set_ylabel("Réel")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model_results[best_model_name]['y_pred_proba'])
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-AUC = {results_df.loc[best_model_name, "roc_auc"]:.2f}')
axes[1,0].plot([0, 1], [0, 1], linestyle='--', color='navy')
axes[1,0].set_title(f'Courbe ROC - {best_model_name}')
axes[1,0].set_xlabel("Taux de faux positifs")
axes[1,0].set_ylabel("Taux de vrais positifs")
axes[1,0].legend(loc='lower right')
axes[1,0].grid(True, alpha=0.3)

# Tableau de synthèse
axes[1,1].axis('off')
table = axes[1,1].table(cellText=results_df.round(3).values,
                        rowLabels=results_df.index,
                        colLabels=results_df.columns,
                        loc='center')
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(12)
axes[1,1].set_title("Tableau des scores", pad=20)

plt.tight_layout()
plt.savefig("model_comparison_task4.png", dpi=300, bbox_inches='tight')
plt.show()

# 7. SAUVEGARDE DU MEILLEUR MODÈLE
joblib.dump(best_model, 'best_churn_model_task4.pkl')
print(f"\n✓ Meilleur modèle sauvegardé sous: best_churn_model_task4.pkl")

# 8. RÉSUMÉ FINAL
print("\n" + "="*60)
print("RÉSUMÉ FINAL - TASK 4")
print("="*60)

print(f"🏆 Meilleur modèle : {best_model_name}")
for metric in metrics:
    print(f"- {metric.capitalize()}: {results_df.loc[best_model_name, metric]:.4f}")

print("\n✓ Task 4 terminée - Modèle prêt pour déploiement")
print("✓ Visualisation : model_comparison_task4.png")
# =============================================================================