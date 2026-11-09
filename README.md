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

## Feuille de route

- [x] Initialisation du projet
- [x] Cadrage du problème et hypothèses → [reports/01_cadrage.md](reports/01_cadrage.md)
- [ ] Chargement et qualité des données
- [ ] Analyse exploratoire et tests statistiques
- [ ] Protocole expérimental (split temporel)
- [ ] Feature engineering
- [ ] Modèles de référence
- [ ] Modélisation avancée et gestion du déséquilibre
- [ ] Évaluation finale et optimisation du seuil
- [ ] Explicabilité (SHAP)
- [ ] Pipeline d'entraînement
- [ ] Démo Streamlit
- [ ] Rapport final

## Auteur

**Cheikh Ahmed Tidiane** — [GitHub](https://github.com/cheikhahmedtidiane)
