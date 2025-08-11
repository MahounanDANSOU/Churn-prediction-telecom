# =============================================================================
# TASK 1: PRÉPARATION DES DONNÉES - CUSTOMER CHURN ANALYSIS
# Objectif: Charger, nettoyer et préparer les données pour l'analyse
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TASK 1: PRÉPARATION DES DONNÉES")
print("="*60)

# 1. CHARGEMENT DES DONNÉES
print("\n1. CHARGEMENT DES DONNÉES")
print("-" * 40)

# Charger le dataset (format CSV)
df = pd.read_csv('Telco_Customer_Churn_Dataset.csv')  # 🔁 Mets bien ce nom dans ton dossier
print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())

# 2. NETTOYAGE DE BASE
print("\n2. NETTOYAGE DE LA VARIABLE CIBLE")
print("-" * 40)

# Nettoyer les espaces dans la colonne 'Churn'
df['Churn'] = df['Churn'].astype(str).str.strip()

# Garder uniquement les valeurs 'Yes' et 'No'
df = df[df['Churn'].isin(['Yes', 'No'])]
print("✓ Valeurs de 'Churn' filtrées")

# Nettoyage de la colonne TotalCharges
print("\n3. NETTOYAGE DE TotalCharges")
print("-" * 40)

# Remplacer les chaînes vides par NaN et convertir en float
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remplacer les NaN par MonthlyCharges * tenure
df['TotalCharges'] = df.apply(
    lambda row: row['MonthlyCharges'] * row['tenure'] if pd.isna(row['TotalCharges']) else row['TotalCharges'],
    axis=1
)
print("✓ TotalCharges converti en float et complété si manquant")

# 4. ENCODAGE DES VARIABLES
print("\n4. ENCODAGE DES VARIABLES")
print("-" * 40)

df_processed = df.copy()
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')
categorical_cols.remove('Churn')

# Encodage binaire (Yes/No)
binary_cols = [col for col in categorical_cols if df_processed[col].nunique() == 2 and set(df_processed[col].unique()) <= {'Yes', 'No'}]
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
    print(f"✓ {col} encodé (Yes=1 / No=0)")

# Encodage label pour les colonnes à +2 modalités
multi_class_cols = [col for col in categorical_cols if col not in binary_cols]
label_encoders = {}
for col in multi_class_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"✓ {col} encodé avec LabelEncoder")

# Encodage de la cible
df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

# 5. DIVISION DES DONNÉES
print("\n5. DIVISION DU DATASET")
print("-" * 40)

# Définir X et y
columns_to_exclude = ['customerID', 'Churn'] + multi_class_cols
X = df_processed.drop(columns=columns_to_exclude, axis=1)
y = df_processed['Churn']

print(f"Dimensions - X: {X.shape}, y: {y.shape}")
print(f"Valeurs NaN dans X : {X.isnull().sum().sum()}, y : {y.isnull().sum()}")

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Données divisées : {X_train.shape[0]} train / {X_test.shape[0]} test")

# 6. SAUVEGARDE
print("\n6. SAUVEGARDE")
print("-" * 40)

df_processed.to_csv('data_processed_task1.csv', index=False)
pd.concat([X_train, y_train], axis=1).to_csv('train_data_task1.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('test_data_task1.csv', index=False)
print("✓ Données sauvegardées :")
print("  - data_processed_task1.csv")
print("  - train_data_task1.csv")
print("  - test_data_task1.csv")

# RÉSUMÉ
print("\n" + "="*60)
print("RÉSUMÉ TASK 1 - PRÉPARATION DES DONNÉES")
print("="*60)
print(f"✓ Données originales : {df.shape[0]} lignes")
print(f"✓ Variables binaires encodées : {len(binary_cols)}")
print(f"✓ Variables multi-classes encodées : {len(multi_class_cols)}")
print(f"✓ Taux de churn : {y.mean()*100:.2f}%")
