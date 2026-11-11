# Détection de fraude dans les transactions Mobile Money

> Projet de Machine Learning de bout en bout : de l'analyse exploratoire à une démo interactive Streamlit, avec une démarche à la fois académique (hypothèses, tests statistiques, protocole expérimental rigoureux) et professionnelle (code modulaire, testé, reproductible).

🚧 **Projet en cours de réalisation**

## Contexte

Le Mobile Money est un moyen de paiement majeur, notamment en Afrique. Sa croissance s'accompagne de fraudes (prises de contrôle de comptes, transferts suivis de retraits immédiats…). L'objectif est de **détecter automatiquement les transactions frauduleuses** tout en limitant les fausses alertes qui dégradent l'expérience client.

## Données

- **Source** : [Financial Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)
- Données synthétiques générées par simulation (type PaySim) à partir de transactions Mobile Money réelles.
- ~6,3 millions de transactions, 11 colonnes, classe cible `isFraud` fortement déséquilibrée.

### Récupérer les données

Les données ne sont pas versionnées (≈ 470 Mo). Téléchargez le CSV depuis Kaggle et placez-le ici :

```
data/raw/Synthetic_Financial_datasets_log.csv
```

## Structure du projet

```
projet_detection_fraude/
├── data/
│   ├── raw/            # Données brutes (non versionnées)
│   └── processed/      # Données transformées (non versionnées)
├── notebooks/          # Notebooks d'analyse numérotés
├── src/                # Code source réutilisable
├── models/             # Modèles entraînés (non versionnés)
├── reports/
│   └── figures/        # Graphiques générés
├── app/                # Démo Streamlit
├── tests/              # Tests unitaires
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/cheikhahmedtidiane/projet_detection_fraude.git
cd projet_detection_fraude
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
```

Lancer les tests :

```bash
pytest
```

## Entraîner les modèles

Une seule commande reproduit l'entraînement des modèles retenus (données → découpage temporel → entraînement → seuil optimal sur la validation → sauvegarde), en ~30 secondes :

```bash
python -m src.train
```

Options utiles :

```bash
python -m src.train --models lightgbm_transaction   # un seul modèle
python -m src.train --cost-fp 2000                  # autre coût de fausse alerte pour le seuil
```

Chaque modèle est sauvegardé dans `models/` avec un fichier de métadonnées JSON (réglages, seuil, métriques de validation, versions des bibliothèques).

## Lancer la démo

Après l'entraînement des modèles :

```bash
streamlit run app/streamlit_app.py
```

La démo comporte trois pages :
- **Analyser une transaction** : score de fraude, décision au seuil retenu et explication SHAP variable par variable ;
- **Simuler une période** : rejouer une journée réelle de validation ou de test et voir l'effet du coût d'une fausse alerte sur les alertes et le coût total ;
- **À propos du modèle** : performances sur le test, ce que le modèle a appris, et limites liées au simulateur.

## Feuille de route

- [x] Initialisation du projet
- [x] Cadrage du problème et hypothèses → [reports/01_cadrage.md](reports/01_cadrage.md)
- [x] Chargement et qualité des données → [notebooks/01_chargement_qualite.ipynb](notebooks/01_chargement_qualite.ipynb)
- [x] Analyse exploratoire et tests statistiques → [notebooks/02_eda_hypotheses.ipynb](notebooks/02_eda_hypotheses.ipynb)
- [x] Protocole expérimental (split temporel) → [notebooks/03_protocole_experimental.ipynb](notebooks/03_protocole_experimental.ipynb)
- [x] Feature engineering → [notebooks/04_feature_engineering.ipynb](notebooks/04_feature_engineering.ipynb)
- [x] Modèles de référence → [notebooks/05_baselines.ipynb](notebooks/05_baselines.ipynb)
- [x] Modélisation avancée et gestion du déséquilibre → [notebooks/06_modelisation_avancee.ipynb](notebooks/06_modelisation_avancee.ipynb)
- [x] Évaluation finale et optimisation du seuil → [notebooks/07_evaluation_finale.ipynb](notebooks/07_evaluation_finale.ipynb)
- [x] Explicabilité (SHAP) → [notebooks/08_explicabilite.ipynb](notebooks/08_explicabilite.ipynb)
- [x] Pipeline d'entraînement → [src/train.py](src/train.py)
- [x] Démo Streamlit → [app/streamlit_app.py](app/streamlit_app.py)
- [ ] Rapport final

## Auteur

**Cheikh Ahmed Tidiane** — [GitHub](https://github.com/cheikhahmedtidiane)
