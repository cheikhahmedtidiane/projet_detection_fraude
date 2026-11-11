"""Page d'accueil : présentation du projet et accès aux autres pages."""

from commun import fmt_money, fmt_pct, load_results, models_available, st

st.title("🛡️ Détection de fraude dans les transactions Mobile Money")
st.markdown(
    "Démo interactive d'un projet de Machine Learning de bout en bout : analyse exploratoire, "
    "tests statistiques, protocole temporel, modèles LightGBM, seuil de décision fixé par le "
    "**coût métier** et explicabilité **SHAP**. Données : simulateur PaySim (6,3 millions de "
    "transactions, 0,13 % de fraudes)."
)

test = load_results("07_test")
if test is not None:
    st.subheader("Résultats sur le jeu de test (jours 15-16, évalué une seule fois)")
    t = test.loc["LightGBM (transaction)"]
    c = test.loc["LightGBM (complet)"]
    r = test.loc["Règle isFlaggedFraud"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Montant fraudé intercepté", fmt_pct(t["montant_detecte_pct"] / 100),
                help="Modèle sans les soldes, au seuil choisi sur la validation.")
    col2.metric("Coût des fraudes + alertes", fmt_money(t["cout_total"]),
                delta=f"-{fmt_money(r['cout_total'] - t['cout_total'])} vs règle existante",
                delta_color="inverse")
    col3.metric("PR-AUC (sans les soldes)", f"{t['pr_auc']:.3f}".replace(".", ","),
                help="Hasard = 0,0016. Voir « À propos » : environ la moitié vient d'un artefact horaire.")
    col4.metric("PR-AUC (avec les soldes)", f"{c['pr_auc']:.3f}".replace(".", ","),
                help="Score parfait dû aux artefacts du simulateur (compte vidé, soldes incohérents).")

st.subheader("Explorer la démo")
col1, col2, col3 = st.columns(3)
with col1, st.container(border=True):
    st.markdown("#### 🔎 Analyser une transaction")
    st.markdown("Saisissez une transaction : score de fraude, décision et **explication** "
                "variable par variable.")
    st.page_link("vues/analyser.py", label="Ouvrir", icon="➡️")
with col2, st.container(border=True):
    st.markdown("#### 📊 Simuler une période")
    st.markdown("Rejouez une journée réelle de validation ou de test et voyez l'effet du "
                "**coût d'une fausse alerte** sur les décisions.")
    st.page_link("vues/simuler.py", label="Ouvrir", icon="➡️")
with col3, st.container(border=True):
    st.markdown("#### ℹ️ À propos du modèle")
    st.markdown("Performances, fonctionnement et **limites** : ce que le modèle a vraiment "
                "appris, et ce qui relève des artefacts du simulateur.")
    st.page_link("vues/a_propos.py", label="Ouvrir", icon="➡️")

if not models_available():
    st.warning("Les modèles ne sont pas encore entraînés : lancez `python -m src.train` "
               "depuis la racine du projet avant d'utiliser les pages d'analyse.")

st.caption("Projet réalisé par Cheikh Ahmed Tidiane — code source sur "
           "[GitHub](https://github.com/cheikhahmedtidiane/projet_detection_fraude).")
