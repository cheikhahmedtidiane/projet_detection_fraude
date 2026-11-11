"""Page « Analyser une transaction » : score, décision et explication SHAP."""

import matplotlib.pyplot as plt

from commun import MODEL_LABELS, make_transaction, require_models, score_and_explain, st
from src.explain import plot_contributions
from src.viz import set_style

set_style()
st.title("🔎 Analyser une transaction")
st.markdown("Saisissez une transaction (ou partez d'un exemple) : chaque modèle donne un score, "
            "une décision au seuil retenu, et l'**explication** de ce score variable par variable.")
require_models()

PRESETS = {
    "Fraude typique : compte vidé en pleine nuit": dict(
        type_="TRANSFER", amount=850_000.0, hour=4, old_orig=850_000.0, new_orig=0.0,
        old_dest=0.0, new_dest=0.0),
    "Retrait légitime en journée": dict(
        type_="CASH_OUT", amount=180_000.0, hour=14, old_orig=250_000.0, new_orig=70_000.0,
        old_dest=1_200_000.0, new_dest=1_380_000.0),
    "Gros transfert en journée (cas limite)": dict(
        type_="TRANSFER", amount=7_500_000.0, hour=16, old_orig=12_000_000.0,
        new_orig=4_500_000.0, old_dest=3_000_000.0, new_dest=10_500_000.0),
    "Fraude plafonnée à 10 M (compte non vidé)": dict(
        type_="TRANSFER", amount=10_000_000.0, hour=13, old_orig=16_580_428.0,
        new_orig=6_580_428.0, old_dest=0.0, new_dest=0.0),
}


def apply_preset() -> None:
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


if "type_" not in st.session_state:
    st.session_state["preset"] = next(iter(PRESETS))
    apply_preset()

st.selectbox("Partir d'un exemple", list(PRESETS), key="preset", on_change=apply_preset)

with st.form("transaction"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transaction**")
        st.radio("Type", ["TRANSFER", "CASH_OUT"], key="type_", horizontal=True,
                 help="Seuls ces deux types contiennent des fraudes dans les données.")
        st.number_input("Montant", min_value=0.0, step=10_000.0, format="%.2f", key="amount")
        st.slider("Heure de la transaction", 0, 23, key="hour", format="%d h")
    with col2:
        st.markdown("**Soldes** (utilisés seulement par le modèle « avec les soldes »)")
        a, b = st.columns(2)
        a.number_input("Émetteur : solde avant", min_value=0.0, step=10_000.0, key="old_orig")
        b.number_input("Émetteur : solde après", min_value=0.0, step=10_000.0, key="new_orig")
        a.number_input("Destinataire : solde avant", min_value=0.0, step=10_000.0, key="old_dest")
        b.number_input("Destinataire : solde après", min_value=0.0, step=10_000.0, key="new_dest")
    st.form_submit_button("Analyser", type="primary", width="stretch")

tx = make_transaction(**{k: st.session_state[k] for k in
                         ["type_", "amount", "hour", "old_orig", "new_orig", "old_dest", "new_dest"]})

st.divider()
for col, model in zip(st.columns(2), ["lightgbm_transaction", "lightgbm_complet"]):
    res = score_and_explain(model, tx)
    with col:
        st.subheader(MODEL_LABELS[model])
        m1, m2 = st.columns(2)
        m1.metric("Score de fraude", f"{res['proba']:.3f}".replace(".", ","))
        m2.metric("Seuil de décision", f"{res['seuil']:.4f}".replace(".", ","),
                  help="Choisi sur la validation en minimisant le coût métier (notebook 07).")
        if res["alerte"]:
            st.error("🚨 **Alerte** : transaction à vérifier", icon=None)
        else:
            st.success("✅ **Pas d'alerte** : transaction acceptée", icon=None)

        fig, ax = plt.subplots(figsize=(7, 3.6 if model == "lightgbm_transaction" else 4.6))
        plot_contributions(ax, res["shap"], res["features"], res["base"],
                           max_features=5 if model == "lightgbm_transaction" else 7)
        ax.set_xlabel("Contribution au score (log-odds)")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

st.caption(
    "Lecture : les barres **orange** poussent vers « fraude », les **bleues** vers « légitime ». "
    "Les scores ne sont pas des probabilités calibrées (la pondération des classes les gonfle) : "
    "seule leur position par rapport au seuil compte. "
    f"Le modèle « sans les soldes » n'utilise que le type, le montant et l'heure."
)
