# 01 — Cadrage du problème

> Ce document fixe le cadre du projet **avant toute modélisation** : problématique, objectifs, hypothèses de recherche, métriques d'évaluation et périmètre. Les choix faits ici guident toutes les étapes suivantes.

---

## 1. Contexte

Le Mobile Money (Orange Money, Wave, M-Pesa…) est devenu un moyen de paiement central, en particulier en Afrique subsaharienne où il compense un faible taux de bancarisation. Ce volume croissant de transactions attire la fraude : prise de contrôle de comptes (vol de code, SIM swap, ingénierie sociale), puis **transfert des fonds vers un compte complice suivi d'un retrait immédiat**.

Pour un opérateur, la fraude a un double coût :
- **coût direct** : montants détournés, remboursements, litiges ;
- **coût indirect** : perte de confiance des clients, risque réglementaire.

À l'inverse, bloquer à tort une transaction légitime (fausse alerte) dégrade l'expérience client et mobilise des équipes d'investigation. **Tout l'enjeu est de trouver le bon compromis entre ces deux erreurs.**

## 2. Problématique

> **Peut-on, à partir des seules caractéristiques d'une transaction Mobile Money (type, montant, soldes, moment), détecter automatiquement les transactions frauduleuses de manière plus efficace que la règle métier existante, tout en maîtrisant le volume de fausses alertes ?**

Formulation ML : **classification binaire supervisée** sur la cible `isFraud` (1 = fraude, 0 = légitime), dans un contexte de **déséquilibre extrême des classes**.

## 3. Données

- **Source** : [Financial Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)
- **Nature** : données **synthétiques** générées par le simulateur PaySim (Lopez-Rojas et al., 2016), calibré sur un mois de logs réels d'un service Mobile Money africain.

| Colonne | Description |
|---|---|
| `step` | Unité de temps : 1 step = 1 heure (744 steps ≈ 31 jours) |
| `type` | Type de transaction : `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | Montant de la transaction |
| `nameOrig` | Identifiant du client émetteur |
| `oldbalanceOrg` / `newbalanceOrig` | Solde de l'émetteur avant / après la transaction |
| `nameDest` | Identifiant du destinataire (préfixe `C` = client, `M` = marchand) |
| `oldbalanceDest` / `newbalanceDest` | Solde du destinataire avant / après la transaction |
| `isFraud` | **Cible** : 1 si la transaction est frauduleuse |
| `isFlaggedFraud` | Alerte de la règle métier existante |

### Premier aperçu (vérifié sur le CSV)

| Indicateur | Valeur |
|---|---|
| Nombre de transactions | 6 362 620 |
| Nombre de fraudes | 8 213 |
| Taux de fraude | **0,129 %** (≈ 1 fraude pour 775 transactions) |
| Période couverte | steps 1 à 743 (~31 jours) |
| Types contenant de la fraude | `TRANSFER` (4 097) et `CASH_OUT` (4 116) **uniquement** |
| Transactions signalées par `isFlaggedFraud` | 16 (toutes frauduleuses) |

**Constat immédiat** : la règle métier existante est très précise (16/16) mais ne détecte que **0,19 % des fraudes** (16/8 213). Elle constitue la **baseline métier** à dépasser.

## 4. Objectifs

1. **Analytique** : comprendre les mécanismes de la fraude dans les données et valider/réfuter des hypothèses par des tests statistiques.
2. **Prédictif** : construire un modèle qui maximise la détection des fraudes sous contrainte d'un volume de fausses alertes acceptable.
3. **Décisionnel** : choisir un seuil de décision fondé sur un **coût métier** plutôt que sur le seuil par défaut de 0,5.
4. **Explicatif** : rendre les prédictions interprétables (SHAP), condition indispensable dans un contexte financier réglementé.
5. **Démonstratif** : proposer une démo Streamlit permettant de scorer une transaction et d'en expliquer la décision.

## 5. Hypothèses de recherche

Chaque hypothèse sera **testée formellement** (Phase 3 : EDA et tests statistiques) puis **confrontée à l'explicabilité du modèle** (Phase 9 : SHAP).

| # | Hypothèse | Méthode de validation |
|---|---|---|
| **H1** | La fraude dépend du type de transaction et se concentre sur `TRANSFER` et `CASH_OUT`. | Test du χ² d'indépendance, V de Cramér |
| **H2** | Les montants des transactions frauduleuses sont significativement plus élevés que ceux des transactions légitimes. | Test de Mann-Whitney U (distributions non normales), taille d'effet |
| **H3** | Les fraudes tendent à **vider le compte émetteur** (solde final nul). | Comparaison de proportions (χ²) |
| **H4** | Les **incohérences de soldes** (écart entre `oldbalance − amount` et `newbalance`) sont fortement discriminantes. | Tests de distribution, importance des variables |
| **H5** | Le taux de fraude varie selon l'**heure de la journée** (la fraude ne suit pas le cycle d'activité diurne des clients légitimes). | χ² sur le taux de fraude par heure |
| **H6** | Un modèle de Machine Learning surpasse significativement la règle métier `isFlaggedFraud`. | Comparaison des métriques sur le jeu de test |

**Question de recherche complémentaire (RQ)** : quelle stratégie de gestion du déséquilibre (pondération des classes, sous-échantillonnage, sur-échantillonnage SMOTE) offre le meilleur compromis performance / coût de calcul sur ce problème ?

## 6. Métriques d'évaluation

### Pourquoi pas l'accuracy ?
Avec 0,129 % de fraudes, un modèle qui prédit **toujours « légitime »** obtient **99,87 % d'accuracy** tout en ne détectant aucune fraude. L'accuracy est donc **inadaptée**.

### Métriques retenues

| Métrique | Rôle | Justification |
|---|---|---|
| **PR-AUC** (Average Precision) | **Métrique principale** de comparaison des modèles | Insensible au grand nombre de vrais négatifs ; plus informative que la ROC-AUC en cas de fort déséquilibre (Saito & Rehmsmeier, 2015) |
| **Recall** (rappel) | Part des fraudes détectées | Une fraude ratée = perte financière directe |
| **Precision** | Part des alertes réellement frauduleuses | Mesure la charge des fausses alertes |
| **F1 / F2-score** | Synthèse précision/rappel | F2 donne plus de poids au rappel, cohérent avec le métier |
| **ROC-AUC** | Métrique secondaire | Reportée pour comparaison avec la littérature |
| **Coût métier total** | **Choix du seuil de décision** | Traduit les erreurs en impact business |

### Fonction de coût métier

Le seuil de décision sera choisi en minimisant :

$$
\text{Coût total} = \sum_{\text{FN}} \text{montant}_i \; + \; N_{\text{FP}} \times C_{\text{FP}}
$$

- **Faux négatif (fraude ratée)** : coût = montant détourné ;
- **Faux positif (fausse alerte)** : coût fixe $C_{\text{FP}}$ (temps d'investigation + friction client), paramètre dont on étudiera la **sensibilité**.

## 7. Protocole expérimental (principes)

- **Split temporel** sur `step` (entraînement sur le passé, test sur le futur), pour simuler des conditions de production réelles et éviter une évaluation trop optimiste.
- **Jeu de test utilisé une seule fois**, à la toute fin.
- Tout prétraitement et rééquilibrage appris **uniquement sur le jeu d'entraînement** (prévention des fuites de données).
- **Reproductibilité** : graines aléatoires fixées, code versionné, pipeline rejouable.

## 8. Périmètre et limites

**Dans le périmètre** : analyse exploratoire, tests statistiques, feature engineering, modélisation supervisée, optimisation du seuil, explicabilité, démo Streamlit.

**Hors périmètre** : déploiement en production (API, conteneurisation), scoring temps réel, monitoring.

**Limites connues** :
- **Données synthétiques** : les résultats ne sont pas directement transposables à un système réel ; la simulation peut produire des motifs plus « propres » que la réalité.
- **Peu de variables** : pas d'informations client (âge, ancienneté, appareil, localisation), qui sont décisives en conditions réelles.
- **Soldes des marchands** (`M…`) non renseignés (valeurs à 0), à traiter avec prudence.
- **Disponibilité des variables** : les soldes « après transaction » sont calculables au moment de la demande (solde courant − montant), leur usage est donc légitime ; ce point sera vérifié lors de l'analyse des fuites de données.

## 9. Références

- Bolton, R. J., & Hand, D. J. (2002). *Statistical Fraud Detection: A Review*. Statistical Science, 17(3).
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A Financial Mobile Money Simulator for Fraud Detection*. EMSS.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE.
