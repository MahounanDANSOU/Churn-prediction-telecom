# =============================================================================
# TASK 3: SEGMENTATION CLIENTÈLE
# Objectif: Segmenter les clients et analyser les risques de churn par segment
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("TASK 3: SEGMENTATION CLIENTÈLE")
print("="*60)

# 1. CHARGEMENT DES DONNÉES
print("\n1. CHARGEMENT DES DONNÉES")
print("-" * 40)

df = pd.read_csv('data_processed_task1.csv')
print(f"✓ Données chargées: {df.shape[0]} clients, {df.shape[1]} variables")

# Variables pour la segmentation
segmentation_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(f"Variables de segmentation: {segmentation_vars}")

# 2. PRÉPARATION DES DONNÉES POUR LA SEGMENTATION
print("\n2. PRÉPARATION DONNÉES SEGMENTATION")
print("-" * 40)

# Extraire les variables de segmentation
X_segment = df[segmentation_vars].copy()

# Vérifier les valeurs manquantes
print("Valeurs manquantes:")
print(X_segment.isnull().sum())

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(X_segment.describe())

# Standardisation des variables (important pour K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_segment)
X_scaled_df = pd.DataFrame(X_scaled, columns=segmentation_vars)

print("✓ Variables standardisées pour le clustering")

# 3. DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS
print("\n3. DÉTERMINATION NOMBRE OPTIMAL DE CLUSTERS")
print("-" * 40)

# Méthode du coude (Elbow Method)
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualisation de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title('Méthode du Coude pour K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de Clusters (k)')
plt.ylabel('Inertie (Within-cluster sum of squares)')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# Marquer le coude optimal (généralement k=4 pour ce type de données)
optimal_k = 4
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
            alpha=0.8, label=f'K optimal = {optimal_k}')
plt.legend()
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Nombre optimal de clusters sélectionné: {optimal_k}")

# 4. APPLICATION DU CLUSTERING K-MEANS
print("\n4. APPLICATION DU CLUSTERING")
print("-" * 40)

# Appliquer K-means avec le nombre optimal de clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Ajouter les clusters au dataframe
df['Segment'] = clusters
df['Segment_Label'] = df['Segment'].map({
    0: 'Nouveaux Clients', 
    1: 'Clients Moyens', 
    2: 'Clients Premium',
    3: 'Clients Loyaux'
})

print("Distribution des segments:")
segment_counts = df['Segment_Label'].value_counts()
for segment, count in segment_counts.items():
    print(f"- {segment}: {count} clients ({count/len(df)*100:.1f}%)")

# 5. ANALYSE DES SEGMENTS
print("\n5. ANALYSE DES SEGMENTS")
print("-" * 40)

# Caractéristiques moyennes par segment
segment_analysis = df.groupby('Segment_Label')[segmentation_vars].mean().round(2)
print("Caractéristiques moyennes par segment:")
print(segment_analysis)

# Visualisation des segments
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot: Tenure vs MonthlyCharges
scatter = axes[0,0].scatter(df['tenure'], df['MonthlyCharges'], 
                           c=df['Segment'], cmap='viridis', alpha=0.6)
axes[0,0].set_title('Segments: Ancienneté vs Charges Mensuelles')
axes[0,0].set_xlabel('Ancienneté (mois)')
axes[0,0].set_ylabel('Charges Mensuelles ($)')
plt.colorbar(scatter, ax=axes[0,0])

# Box plot par segment pour MonthlyCharges
df.boxplot(column='MonthlyCharges', by='Segment_Label', ax=axes[0,1])
axes[0,1].set_title('Charges Mensuelles par Segment')
axes[0,1].set_ylabel('Charges Mensuelles ($)')

# Box plot par segment pour Tenure
df.boxplot(column='tenure', by='Segment_Label', ax=axes[1,0])
axes[1,0].set_title('Ancienneté par Segment')
axes[1,0].set_ylabel('Ancienneté (mois)')

# Distribution des segments
segment_counts.plot(kind='bar', ax=axes[1,1], color='skyblue')
axes[1,1].set_title('Distribution des Segments')
axes[1,1].set_ylabel('Nombre de Clients')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('customer_segments_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. ANALYSE DU CHURN PAR SEGMENT
print("\n6. ANALYSE DU CHURN PAR SEGMENT")
print("-" * 40)

# Taux de churn par segment
churn_by_segment = df.groupby('Segment_Label')['Churn'].agg(['count', 'sum', 'mean']).round(4)
churn_by_segment.columns = ['Total_Clients', 'Churn_Count', 'Churn_Rate']
churn_by_segment['Churn_Rate_Pct'] = churn_by_segment['Churn_Rate'] * 100

print("Analyse du churn par segment:")
for segment in churn_by_segment.index:
    total = churn_by_segment.loc[segment, 'Total_Clients']
    churned = churn_by_segment.loc[segment, 'Churn_Count']
    rate = churn_by_segment.loc[segment, 'Churn_Rate_Pct']
    print(f"- {segment}: {churned}/{total} clients ({rate:.2f}%)")

# Visualisation du churn par segment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Taux de churn par segment
churn_by_segment['Churn_Rate_Pct'].plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Taux de Churn par Segment Client')
ax1.set_ylabel('Taux de Churn (%)')
ax1.tick_params(axis='x', rotation=45)

# Heatmap du churn par segment
churn_heatmap = pd.crosstab(df['Segment_Label'], df['Churn'])
sns.heatmap(churn_heatmap, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax2)
ax2.set_title('Matrice Segment vs Churn')
ax2.set_ylabel('Segment')
ax2.set_xlabel('Churn (0=No, 1=Yes)')

plt.tight_layout()
plt.savefig('churn_by_segment.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. IDENTIFICATION DES CLIENTS À RISQUE ÉLEVÉ
print("\n7. IDENTIFICATION CLIENTS À RISQUE ÉLEVÉ")
print("-" * 40)

# Définir les critères de risque élevé
high_risk_criteria = (
    (df['Churn'] == 0) &  # Clients actuels (pas encore partis)
    (
        (df['Segment_Label'].isin(['Nouveaux Clients'])) |  # Segments à risque
        (df['tenure'] < 12) |  # Faible ancienneté
        (df['Contract_encoded'] == 0) |  # Contrat month-to-month (si encodé comme 0)
        (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75))  # Charges élevées
    )
)

high_risk_customers = df[high_risk_criteria].copy()
print(f"✓ Clients à risque élevé identifiés: {len(high_risk_customers)}")

# Caractéristiques des clients à risque
print("\nCaractéristiques des clients à risque élevé:")
risk_profile = high_risk_customers[segmentation_vars + ['Segment_Label']].describe()
print(risk_profile)

# Distribution des clients à risque par segment
risk_by_segment = high_risk_customers['Segment_Label'].value_counts()
print(f"\nRépartition des clients à risque par segment:")
for segment, count in risk_by_segment.items():
    total_in_segment = df[df['Segment_Label'] == segment].shape[0]
    risk_pct = count / total_in_segment * 100
    print(f"- {segment}: {count} clients ({risk_pct:.1f}% du segment)")

# 8. IDENTIFICATION DES CLIENTS HIGH-VALUE À RISQUE
print("\n8. CLIENTS HIGH-VALUE À RISQUE")
print("-" * 40)

# Définir les clients high-value (charges élevées + longue ancienneté OU charges très élevées)
high_value_criteria = (
    (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8)) &
    (
        (df['tenure'] > 24) |  # Clients loyaux avec charges élevées
        (df['TotalCharges'] > df['TotalCharges'].quantile(0.8))  # Ou revenus totaux élevés
    )
)

# Clients high-value à risque (critères de valeur + critères de risque)
high_value_at_risk = df[
    high_value_criteria & 
    (df['Churn'] == 0) &  # Clients actuels
    (
        (df['Contract_encoded'] == 0) |  # Contrat flexible
        (df['PaymentMethod_encoded'] == 2)  # Electronic check (si plus risqué)
    )
].copy()

print(f"✓ Clients high-value à risque: {len(high_value_at_risk)}")
if len(high_value_at_risk) > 0:
    avg_value = high_value_at_risk['MonthlyCharges'].mean()
    total_risk_revenue = high_value_at_risk['MonthlyCharges'].sum()
    print(f"- Charges mensuelles moyennes: ${avg_value:.2f}")
    print(f"- Revenus mensuels totaux à risque: ${total_risk_revenue:.2f}")

# 9. VISUALISATION DES SEGMENTS ET RISQUES
print("\n9. VISUALISATION SEGMENTS ET RISQUES")
print("-" * 40)

# Analyse PCA pour visualisation 2D des segments
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Segment'] = df['Segment_Label']
pca_df['Churn'] = df['Churn']
pca_df['Risk_Level'] = 'Standard'
pca_df.loc[high_risk_customers.index, 'Risk_Level'] = 'High Risk'
if len(high_value_at_risk) > 0:
    pca_df.loc[high_value_at_risk.index, 'Risk_Level'] = 'High Value at Risk'

# Visualisation PCA avec segments et risques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Segments dans l'espace PCA
scatter1 = ax1.scatter(pca_df['PC1'], pca_df['PC2'], c=df['Segment'], 
                      cmap='viridis', alpha=0.6)
ax1.set_title('Segments Clients dans l\'Espace PCA')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.colorbar(scatter1, ax=ax1)

# Niveaux de risque dans l'espace PCA
risk_colors = {'Standard': 'blue', 'High Risk': 'orange', 'High Value at Risk': 'red'}
for risk_level, color in risk_colors.items():
    mask = pca_df['Risk_Level'] == risk_level
    ax2.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
               c=color, alpha=0.6, label=risk_level)

ax2.set_title('Niveaux de Risque dans l\'Espace PCA')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax2.legend()

plt.tight_layout()
plt.savefig('pca_segments_risks.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. SAUVEGARDE DES RÉSULTATS
print("\n10. SAUVEGARDE DES RÉSULTATS")
print("-" * 40)

try:
    # Sauvegarder le dataset avec les segments
    df.to_csv('data_with_segments_task3.csv', index=False)
    
    # Sauvegarder les clients à risque
    high_risk_customers.to_csv('high_risk_customers_task3.csv', index=False)
    
    if len(high_value_at_risk) > 0:
        high_value_at_risk.to_csv('high_value_at_risk_customers_task3.csv', index=False)
    
    # Sauvegarder l'analyse des segments
    segment_analysis.to_csv('segment_analysis_task3.csv')
    churn_by_segment.to_csv('churn_by_segment_task3.csv')
    
    print("✓ Fichiers sauvegardés:")
    print("  - data_with_segments_task3.csv")
    print("  - high_risk_customers_task3.csv")
    print("  - high_value_at_risk_customers_task3.csv")
    print("  - segment_analysis_task3.csv")
    print("  - churn_by_segment_task3.csv")
    
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde: {e}")

# RÉSUMÉ FINAL
print("\n" + "="*60)
print("RÉSUMÉ TASK 3 - SEGMENTATION CLIENTÈLE")
print("="*60)

print(f"✓ {optimal_k} segments clients identifiés")
print(f"✓ {len(high_risk_customers)} clients à risque élevé détectés")
if len(high_value_at_risk) > 0:
    print(f"✓ {len(high_value_at_risk)} clients high-value à risque identifiés")

print("\nSegments avec taux de churn le plus élevé:")
top_churn_segments = churn_by_segment.sort_values('Churn_Rate_Pct', ascending=False)
for i, (segment, row) in enumerate(top_churn_segments.head(2).iterrows()):
    print(f"{i+1}. {segment}: {row['Churn_Rate_Pct']:.2f}%")

print("\nFichiers générés:")
print("- elbow_method.png")
print("- customer_segments_analysis.png")
print("- churn_by_segment.png") 
print("- pca_segments_risks.png")

print("\n✓ Task 3 terminée - Segments prêts pour la modélisation (Task 4)")