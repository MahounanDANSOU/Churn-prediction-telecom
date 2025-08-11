# =============================================================================
# TASK 2: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration graphique
sns.set_palette("husl")
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("TASK 2: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
print("="*60)

# -------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# -------------------------------------------------------------------------
print("\n1. CHARGEMENT DES DONNÉES")
df = pd.read_csv('data_processed_task1.csv')
print(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Reconstitution label Churn
df['Churn_label'] = df['Churn'].map({0: 'No', 1: 'Yes'})

# -------------------------------------------------------------------------
# 2. TAUX DE CHURN GLOBAL
# -------------------------------------------------------------------------
print("\n2. ANALYSE DU TAUX DE CHURN GLOBAL")
churn_counts = df['Churn'].value_counts()
churn_rate = churn_counts[1] / len(df) * 100

print(f"- Clients restés (No Churn): {churn_counts[0]} ({100 - churn_rate:.2f}%)")
print(f"- Clients partis (Churn): {churn_counts[1]} ({churn_rate:.2f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#2ecc71', '#e74c3c']
churn_labels = ['No Churn', 'Churn']

ax1.bar(churn_labels, churn_counts, color=colors)
ax1.set_title('Nombre de clients par statut de churn')
ax1.set_ylabel('Nombre de clients')
for i, v in enumerate(churn_counts):
    ax1.text(i, v + 50, f"{v} ({v/len(df)*100:.1f}%)", ha='center')

ax2.pie(churn_counts, labels=churn_labels, colors=colors, autopct='%1.1f%%', explode=(0, 0.1))
ax2.set_title('Répartition en pourcentage')

plt.tight_layout()
plt.savefig('churn_global_analysis.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 3. ANALYSE DÉMOGRAPHIQUE
# -------------------------------------------------------------------------
print("\n3. ANALYSE DÉMOGRAPHIQUE")
demographic_vars = ['gender', 'Partner', 'Dependents']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, var in enumerate(demographic_vars):
    if var in df.columns:
        churn_by_cat = pd.crosstab(df[var], df['Churn_label'], normalize='index') * 100
        churn_by_cat.plot(kind='bar', ax=axes[i], color=colors)
        axes[i].set_title(f'Taux de churn par {var}')
        axes[i].set_ylabel('Pourcentage (%)')
        axes[i].legend(['No Churn', 'Churn'])
        for label in churn_by_cat.index:
            print(f"- {var} = {label} → Churn: {churn_by_cat.loc[label, 'Yes']:.2f}%")

plt.tight_layout()
plt.savefig('demographic_churn_analysis.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 4. ANALYSE DE L'ANCIENNETÉ (TENURE)
# -------------------------------------------------------------------------
print("\n4. ANALYSE DE L'ANCIENNETÉ (TENURE)")
tenure_stats = df.groupby('Churn_label')['tenure'].describe()
print(tenure_stats)

no_churn_tenure = df[df['Churn'] == 0]['tenure']
churn_tenure = df[df['Churn'] == 1]['tenure']
t_stat, p_value = stats.ttest_ind(no_churn_tenure, churn_tenure)
print(f"\nTest t-Student : t = {t_stat:.2f}, p = {p_value:.4f} → {'significatif' if p_value < 0.05 else 'non significatif'}")

# Créer groupes robustes
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, df['tenure'].max()],
                            labels=['0-12 mois', '13-24 mois', '25-48 mois', '48+ mois'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution tenure
sns.histplot(data=df, x='tenure', hue='Churn_label', bins=30, ax=axes[0, 0], palette=colors)
axes[0, 0].set_title("Distribution de l'ancienneté")

# Boxplot
sns.boxplot(data=df, x='Churn_label', y='tenure', palette=colors, ax=axes[0, 1])
axes[0, 1].set_title('Boxplot de tenure par churn')

# Par tranches
tenure_churn = pd.crosstab(df['tenure_group'], df['Churn_label'], normalize='index') * 100
tenure_churn.plot(kind='bar', ax=axes[1, 0], color=colors)
axes[1, 0].set_title('Taux de churn par tranche de tenure')

# Ligne de tendance
tenure_churn_rate = df.groupby('tenure')['Churn'].mean() * 100
axes[1, 1].plot(tenure_churn_rate.index, tenure_churn_rate.values, color='#e74c3c', linewidth=2)
axes[1, 1].set_title('Taux de churn par mois de tenure')
axes[1, 1].set_xlabel('Tenure (mois)')
axes[1, 1].set_ylabel('Churn (%)')

plt.tight_layout()
plt.savefig('tenure_churn_analysis.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 5. CONTRATS
# -------------------------------------------------------------------------
print("\n5. ANALYSE DES TYPES DE CONTRATS")
if 'Contract_encoded' in df.columns:
    contract_map = df.groupby('Contract')['Contract_encoded'].first().to_dict()
    inv_map = {v: k for k, v in contract_map.items()}
    df['Contract_original'] = df['Contract_encoded'].map(inv_map)
    
    churn_contract = pd.crosstab(df['Contract_original'], df['Churn_label'], normalize='index') * 100
    for c in churn_contract.index:
        print(f"- {c}: {churn_contract.loc[c, 'Yes']:.2f}% churn")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    churn_contract.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Taux de churn par contrat')
    dist = df['Contract_original'].value_counts()
    ax2.pie(dist, labels=dist.index, autopct='%1.1f%%')
    ax2.set_title('Répartition des types de contrat')
    plt.tight_layout()
    plt.savefig('contract_churn_analysis.png', dpi=300)
    plt.show()

# -------------------------------------------------------------------------
# 6. MÉTHODES DE PAIEMENT
# -------------------------------------------------------------------------
print("\n6. MÉTHODES DE PAIEMENT")
if 'PaymentMethod_encoded' in df.columns:
    pay_map = df.groupby('PaymentMethod')['PaymentMethod_encoded'].first().to_dict()
    inv_map = {v: k for k, v in pay_map.items()}
    df['PaymentMethod_original'] = df['PaymentMethod_encoded'].map(inv_map)

    churn_pay = pd.crosstab(df['PaymentMethod_original'], df['Churn_label'], normalize='index') * 100
    for m in churn_pay.index:
        print(f"- {m}: {churn_pay.loc[m, 'Yes']:.2f}% churn")

    churn_pay.plot(kind='bar', color=colors)
    plt.title('Taux de churn par méthode de paiement')
    plt.ylabel('%')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('payment_churn_analysis.png', dpi=300)
    plt.show()

# -------------------------------------------------------------------------
# 7. CHARGES MENSUELLES
# -------------------------------------------------------------------------
print("\n7. ANALYSE DES CHARGES MENSUELLES")
monthly_stats = df.groupby('Churn_label')['MonthlyCharges'].describe()
print(monthly_stats)

no_churn_mc = df[df['Churn'] == 0]['MonthlyCharges']
churn_mc = df[df['Churn'] == 1]['MonthlyCharges']
_, pval = stats.ttest_ind(no_churn_mc, churn_mc)
print(f"\nT-test: p = {pval:.4f} → {'significatif' if pval < 0.05 else 'non significatif'}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x='Churn_label', y='MonthlyCharges', palette=colors, ax=axes[0])
axes[0].set_title('Boxplot des charges mensuelles')
sns.histplot(data=df, x='MonthlyCharges', hue='Churn_label', bins=30, palette=colors, ax=axes[1])
axes[1].set_title('Distribution des charges mensuelles')
plt.tight_layout()
plt.savefig('monthly_charges_analysis.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 8. MATRICE DE CORRÉLATION
# -------------------------------------------------------------------------
print("\n8. MATRICE DE CORRÉLATION")
corr_matrix = df.select_dtypes(include=[np.number]).corr()
churn_corr = corr_matrix['Churn'].sort_values(key=abs, ascending=False)
print("Corrélations > 0.1 avec churn :")
print(churn_corr[abs(churn_corr) > 0.1])

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', fmt='.2f', square=True)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 9. RÉSUMÉ
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("RÉSUMÉ DES INSIGHTS - TASK 2")
print("="*60)
print(f"✓ Taux de churn global: {churn_rate:.2f}%")
print(f"✓ Ancienneté moyenne - No Churn: {no_churn_tenure.mean():.1f} mois")
print(f"✓ Ancienneté moyenne - Churn: {churn_tenure.mean():.1f} mois")
print(f"✓ Charges mensuelles moyennes - No Churn: ${no_churn_mc.mean():.2f}")
print(f"✓ Charges mensuelles moyennes - Churn: ${churn_mc.mean():.2f}")

print("\nINSIGHTS CLÉS:")
print("- Churn élevé chez les nouveaux clients (< 12 mois)")
print("- Contrat month-to-month = risque de churn plus élevé")
print("- Méthode Electronic check liée à un churn plus fort")
print("- Charges mensuelles élevées → churn élevé")
print("- Les clients sans partenaire ni dépendant partent plus")

print("\n✓ Task 2 terminée — Les données sont prêtes pour la segmentation ou la modélisation.")
# =============================================================================