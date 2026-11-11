"""Page « Simuler une période » : rejouer une journée réelle et mesurer le coût des décisions."""

import matplotlib.pyplot as plt
import numpy as np

from commun import (COST_FP_CHOICES, MODEL_LABELS, SPLIT_LABELS, fmt_int, fmt_money, fmt_pct,
                    get_scored_split, optimal_threshold_for, require_models, require_splits,
                    safe_ratio, st)
from src.config import TARGET
from src.viz import COLOR_FRAUD, COLOR_LEGIT, set_style

set_style()
st.title("📊 Simuler une période")
st.markdown("Rejouez de vraies journées du jeu de validation ou de test : le modèle analyse chaque "
            "transaction et lève des alertes. Faites varier le **coût d'une fausse alerte** pour "
            "voir le seuil de décision et le coût total s'adapter.")
require_models()
require_splits()

# ----------------------------------------------------------------- paramètres
c1, c2, c3 = st.columns([1.1, 1, 1.3])
split = c1.radio("Jeu de données", list(SPLIT_LABELS), format_func=SPLIT_LABELS.get, horizontal=True)
data = get_scored_split(split)
days = sorted(data["jour"].unique())
day_choice = c2.radio("Période", ["Les 2 jours"] + [f"Jour {d}" for d in days], horizontal=True)
model = c3.radio("Modèle", ["lightgbm_transaction", "lightgbm_complet"], format_func=MODEL_LABELS.get)

c4, c5 = st.columns([2, 1])
cost_fp = c4.select_slider("Coût d'une fausse alerte (vérification, gêne client…)", COST_FP_CHOICES,
                           value=5_000, format_func=fmt_int)
opt = optimal_threshold_for(model, cost_fp)
manual = c5.toggle("Fixer le seuil à la main", value=False)
if manual:
    threshold = c5.slider("Seuil", 0.0, 1.0, value=round(opt["seuil"], 3), step=0.001, format="%.3f")
else:
    threshold = opt["seuil"]
    c5.metric("Seuil optimal (choisi sur la validation)", f"{threshold:.4f}".replace(".", ","))

# ----------------------------------------------------------------- décisions
period = data if day_choice == "Les 2 jours" else data[data["jour"] == int(day_choice.split()[-1])]
y = period[TARGET].to_numpy() == 1
score = period[f"score_{model}"].to_numpy()
alert = score >= threshold
amount = period["amount"].to_numpy()

n_alerts, tp, fp = int(alert.sum()), int((alert & y).sum()), int((alert & ~y).sum())
fraud_amount = amount[y].sum()
missed_amount = amount[y & ~alert].sum()
cost = missed_amount + fp * cost_fp

st.divider()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions", fmt_int(len(period)), help=f"dont {fmt_int(y.sum())} fraudes")
k2.metric("Alertes levées", fmt_int(n_alerts),
          help="Nombre de transactions que les analystes devraient vérifier.")
k3.metric("Fraudes détectées", f"{tp} / {int(y.sum())}", delta=fmt_pct(safe_ratio(tp, y.sum())),
          delta_color="off")
k4.metric("Précision des alertes", fmt_pct(safe_ratio(tp, n_alerts)),
          help=f"{fmt_int(fp)} fausses alertes")
k5.metric("Montant fraudé intercepté", fmt_pct(safe_ratio(fraud_amount - missed_amount, fraud_amount)))

k6, k7, k8 = st.columns(3)
k6.metric("Coût sans aucun modèle", fmt_money(fraud_amount),
          help="Toutes les fraudes passent : leur montant est perdu.")
k7.metric("Coût avec le modèle", fmt_money(cost),
          delta=f"{fmt_pct(safe_ratio(cost - fraud_amount, fraud_amount), 0)}", delta_color="inverse",
          help=f"Fraudes ratées ({fmt_money(missed_amount)}) + fausses alertes "
               f"({fmt_int(fp)} × {fmt_int(cost_fp)}).")
k8.metric("dont fausses alertes", fmt_money(fp * cost_fp))

# ----------------------------------------------------------------- graphiques par heure
hours = np.arange(24)
h = period["heure"].to_numpy()
detected = np.bincount(h[y & alert], minlength=24)
missed = np.bincount(h[y & ~alert], minlength=24)
false_alerts = np.bincount(h[~y & alert], minlength=24)

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(hours, detected, color=COLOR_FRAUD, width=0.75, label="Détectées")
    ax.bar(hours, missed, bottom=detected, color="#f6c3ad", width=0.75, label="Manquées")
    ax.set_title("Fraudes par heure", loc="left")
    ax.set_xlabel("Heure")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
with g2:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(hours, false_alerts, color=COLOR_LEGIT, width=0.75)
    ax.set_title("Fausses alertes par heure", loc="left")
    ax.set_xlabel("Heure")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ----------------------------------------------------------------- alertes
st.subheader("Les 20 alertes les plus suspectes")
top = period[alert].nlargest(20, f"score_{model}")
table = top.assign(
    **{"Score": top[f"score_{model}"].round(4),
       "Réalité": np.where(top[TARGET] == 1, "🚨 Fraude", "✅ Légitime")}
)[["jour", "heure", "type", "amount", "oldbalanceOrig", "newbalanceOrig", "Score", "Réalité"]]
table.columns = ["Jour", "Heure", "Type", "Montant", "Solde émetteur avant", "Solde émetteur après",
                 "Score", "Réalité"]
st.dataframe(table, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.0f")
                            for c in ["Montant", "Solde émetteur avant", "Solde émetteur après"]})

st.info(
    "**À garder en tête** : entre 3 h et 8 h, le simulateur ne génère **aucune** transaction "
    "légitime, alors que les fraudes surviennent à toute heure. Les alertes de ces heures sont donc "
    "trivialement justes : c'est un artefact des données, analysé dans « À propos du modèle ».",
    icon="⚠️",
)
