# ML Cancer Risk Project

## Description
Ce projet vise à prédire le diagnostic du cancer de la thyroïde à partir de données médicales en utilisant plusieurs modèles de Machine Learning. L'objectif est d'optimiser la classification entre cas bénins et malins en évaluant différentes techniques de modélisation et de prétraitement des données.

## Données utilisées
Le dataset contient des informations sur plusieurs patients avec les variables suivantes :
- **Données démographiques** : Age, Gender, Country, Ethnicity
- **Facteurs de risque** (binaire) : Family_History, Radiation_Exposure, Iodine_Deficiency, Smoking, Obesity, Diabetes
- **Données biologiques** : TSH_Level, T3_Level, T4_Level, Nodule_Size
- **Risque évalué** : Thyroid_Cancer_Risk (Low, Medium, High)
- **Diagnostic cible** : Diagnosis (Benign/Malignant)

## Prétraitement des données
- **Encodage des variables** :
  - Gender → OneHotEncoder (Gender_Male, Gender_Female)
  - Thyroid_Cancer_Risk → OrdinalEncoder (Low = 0, Medium = 1, High = 2)
  - Variables binaires → Mapping (Yes = 1, No = 0)
- **Gestion du déséquilibre des classes** :
  - SMOTE a été utilisé pour certaines approches afin de rééquilibrer les classes et améliorer la robustesse des modèles.

## Modèles évalués
Plusieurs modèles ont été testés et comparés :
- **Régression Logistique**
- **Arbre de décision**
- **Random Forest**
- **LightGBM**
- **SVM**

L'évaluation a été réalisée via la métrique **AUC** et **Accuracy**, avec des comparaisons basées sur la validation croisée.

## Résultats
Les modèles ont été comparés selon leurs performances en termes d'AUC et d'Accuracy. Voici un aperçu des résultats obtenus :

| Modèle | Accuracy | AUC |
|--------|----------|-----|
| Random Forest SMOTE | 0.826 | 0.695 |
| LGBM SMOTE | 0.783 | 0.681 |
| Decision Tree SMOTE | 0.822 | 0.691 |
| Decision Tree | 0.826 | 0.695 |
| Logistic Regression | 0.826 | 0.696 |

## Utilisation
### Installation
```bash
# Cloner le repo
git clone https://github.com/Death4Sword/ML_Cancer_Risk_Project.git
cd ML_Cancer_Risk_Project

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'application Streamlit
```bash
streamlit run app.py
```
L'interface permet d'entrer des valeurs pour prédire si un patient présente un cancer bénin ou malin.