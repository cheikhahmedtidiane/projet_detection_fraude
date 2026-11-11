"""Page « À propos du modèle » : performances, fonctionnement et limites."""

import pandas as pd

from commun import MODEL_LABELS, figure_path, fmt_int, fmt_money, get_models, load_results, require_models, st

st.title("ℹ️ À propos du modèle")
require_models()
models = get_models()

st.header("Les deux modèles")
cols = st.columns(2)
for col, name in zip(cols, ["lightgbm_transaction", "lightgbm_complet"]):
    _, meta = models[name]
    v = meta["validation"]
    with col, st.container(border=True):
        st.subheader(MODEL_LABELS[name])
        st.markdown(f"*{meta['modele']['description']}*")
        st.markdown("**Variables** : " + ", ".join(f"`{x}`" for x in meta["variables"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("PR-AUC (validation)", f"{v['pr_auc']:.3f}".replace(".", ","))
        m2.metric("Seuil", f"{meta['seuil']:.4f}".replace(".", ","))
        m3.metric("Montant intercepté", f"{v['montant_detecte_pct']:.1f} %".replace(".", ","))
        e = meta["entrainement"]
        st.caption(f"Entraîné sur {fmt_int(e['n_transactions'])} transactions "
                   f"(jours {e['jours'][0]} à {e['jours'][1]}), coût d'une fausse alerte : "
                   f"{fmt_int(meta['cout_fausse_alerte'])}. LightGBM "
                   f"{meta['environnement']['lightgbm']}, entraîné le {meta['environnement']['date'][:10]}.")

test = load_results("07_test")
if test is not None:
    st.header("Évaluation finale sur le test")
    st.markdown("Jours 15-16, jamais vus pendant l'entraînement ni le réglage, évalués **une seule fois**.")
    rows = ["Règle isFlaggedFraud", "Règle compte vidé", "Logistique (transaction)",
            "LightGBM (transaction)", "LightGBM (complet)"]
    t = test.loc[rows]
    table = pd.DataFrame({
        "PR-AUC": t["pr_auc"].round(3),
        "Alertes": t["n_alertes"].astype(int),
        "Fraudes détectées": (100 * t["rappel"]).round(1).astype(str) + " %",
        "Montant intercepté": t["montant_detecte_pct"].round(1).astype(str) + " %",
        "Coût total": t["cout_total"].map(fmt_money),
    }, index=rows)
    st.dataframe(table, width="stretch")

st.header("Ce que le modèle a appris (SHAP)")
tab1, tab2, tab3 = st.tabs(["Importance des variables", "Effet de l'heure", "Effet du montant"])
for tab, fig, caption in [
    (tab1, "08_importance_shap", "Importance moyenne des variables pour chaque modèle."),
    (tab2, "08_effet_heure", "En haut : effet de l'heure appris par le modèle. En bas : taux de fraude "
                             "réel par heure. Entre 3 h et 8 h, il n'y a aucune transaction légitime."),
    (tab3, "08_effet_montant", "Effet non linéaire du montant : les gros montants poussent fortement "
                               "vers la fraude, avec un plateau au plafond de 10 M des fraudes simulées."),
]:
    with tab:
        path = figure_path(fig)
        if path:
            st.image(str(path), caption=caption, width="stretch")

st.header("Limites : à lire avant de conclure")
st.markdown("""
Les données proviennent d'un **simulateur** (PaySim), dont certains mécanismes rendent la fraude
artificiellement facile à détecter :

- **Compte vidé** : 97,6 % des fraudes vident le compte émetteur en une seule opération, ce que ne
  fait aucune transaction légitime. Le modèle « avec les soldes » s'appuie surtout sur ce signal, d'où
  son score parfait.
- **Soldes incohérents** : les soldes des transactions légitimes sont souvent faux, ceux des fraudes
  exacts. C'est l'inverse de ce qu'on attendrait dans la réalité.
- **Heures vides** : entre 3 h et 8 h, il n'y a **aucune** transaction légitime, mais toujours des
  fraudes (environ 27 % d'entre elles). Hors de ces heures, la PR-AUC du modèle « sans les soldes »
  passe de 0,375 à **0,18** : c'est l'estimation la plus réaliste de sa performance.
- **Coût métier simplifié** : un coût fixe par fausse alerte, sans capacité limitée d'analystes ni
  récupération partielle des fonds.

Le modèle **sans les soldes** reste le plus crédible : il repose en partie sur un vrai signal
(montant élevé, type TRANSFER), utilisable sur des données réelles.
""")
