import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import cvxpy as cp
from datetime import datetime


# --- Chargement des données ---
df = pd.read_csv("streamlit_app/Action.csv")

# Conversion de la colonne 'date' en datetime et mise en index
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)

# Suppression des colonnes inutiles
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

# Tri par date croissante (du plus ancien au plus récent)
df = df.sort_index(ascending=True)

# --- Traitement des données ---
# Suppression des colonnes avec plus de 70% de valeurs manquantes
missing_stats = df.isnull().mean()
df = df[missing_stats[missing_stats <= 0.7].index]

# Imputation des valeurs manquantes
df = df.ffill().bfill()

# Conversion en numérique si nécessaire
df = df.apply(pd.to_numeric, errors='coerce')




# --- Fonction rendement top 5 ---
def Top_5_Action(df):
    df_mois = df[df.index.month <= 6]
    rendements = (df_mois.iloc[-1] / df_mois.iloc[0]) - 1
    rendements_tries = rendements.sort_values(ascending=False)
    return rendements_tries[:5]

# --- Fonction volatilité et coefficient de variation ---
def top_cv_volatilite_graph(df, top_n=5):
    df_6mois = df[df.index.month <= 6]
    mean_prices = df_6mois.mean()
    std_prices = df_6mois.std()
    cv = std_prices / mean_prices
    log_returns = np.log(df_6mois / df_6mois.shift(1))
    volatilite = log_returns.std()
    top_cv = cv.sort_values(ascending=False).head(top_n)
    top_vol = volatilite.sort_values(ascending=False).head(top_n)
    return top_cv, top_vol




def calcul_volatilite_1an(serie_prix, freq=252):
    """
    Calcule la volatilité annualisée sur les 12 derniers mois.

    Paramètres :
    - serie_prix : Series pandas des prix (index = dates)
    - freq : fréquence des observations (252 pour données journalières)

    Retour :
    - volatilité annualisée sur 1 an (float)
    """
    # Garder les 252 derniers points (en supposant données journalières)
    serie_1an = serie_prix.dropna().iloc[-freq:]
    rendements = serie_1an.pct_change().dropna()
    volatilite =serie_1an.std() * np.sqrt(freq)
    return volatilite


def calcul_rendement_periode(serie_prix, freq=252):
    """
    Calcule le rendement total sur la dernière période (ex : 1 an).

    Paramètres :
    - serie_prix : Series pandas des prix (index = dates)
    - freq : nombre de points correspondant à la période (252 pour 1 an de trading journalier)

    Retour :
    - rendement total sur la période (float)
    """
    serie_1an = serie_prix.dropna().iloc[-freq:]
    rendement = (serie_1an.iloc[-1] / serie_1an.iloc[0]) - 1
    return rendement






def portefeuille_streamlit(df, capital, cible_rendement=0.10, poids_max=0.25, correlation_max=0.8):
    # 1. Filtrer période de juillet 2024 à aujourd’hui
    df = df[(df.index >= pd.Timestamp('2024-07-01')) & (df.index <= pd.Timestamp(datetime.now().date()))]
    df = df.ffill().bfill()

    if df.empty or df.shape[0] < 2:
        st.error("❌ Données insuffisantes pour effectuer l'analyse.")
        return

    # 2. Rendements mensuels
    returns = df.resample('M').last().pct_change().dropna()

    # 3. Rendement total sur la période
    cum_returns = (df.iloc[-1] / df.iloc[0]) - 1
    rentables = cum_returns[cum_returns >= cible_rendement]

    if rentables.empty:
        st.warning("❌ Aucune action n'a un rendement ≥ 10%.")
        return

    # 4. Corrélation
    returns_filtered = returns[rentables.index]
    corr_matrix = returns_filtered.corr()
    avg_corr = corr_matrix.apply(lambda x: x.drop(x.name).mean())
    faiblement_correlees = avg_corr[avg_corr <= correlation_max].index.tolist()

    if not faiblement_correlees:
        st.warning("❌ Aucune action avec une corrélation moyenne ≤ seuil.")
        return

    # 5. Optimisation
    rets = returns[faiblement_correlees]
    mean_returns = rets.mean()
    cov_matrix = rets.cov()
    n = len(faiblement_correlees)

    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix.values)
    expected_return = mean_returns.values @ w

    constraints = [
        cp.sum(w) == 1,
        expected_return >= cible_rendement / 12,  # objectif mensuel
        w >= 0,
        w <= poids_max
    ]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    if w.value is None:
        st.error("❌ Optimisation échouée. Essayez d’ajuster vos contraintes.")
        return

    # Résultats
    poids = w.value
    poids = np.round(poids, 4)
    expected_return = mean_returns.values @ poids
    rendement_annuel = (1 + expected_return)**12 - 1
    # Risque du portefeuille (écart-type annualisé)
    variance_portefeuille = poids.T @ cov_matrix.values @ poids
    risque_annuel = np.sqrt(variance_portefeuille) * np.sqrt(12)

    portefeuille = pd.Series(poids, index=faiblement_correlees)
    portefeuille = portefeuille[portefeuille > 0]

    st.success("🎯 Optimisation réussie !")
    col1 ,col2, col3= st.columns([1,1,1])
    with col2:
        st.metric("Rendement annuel projeté", f"{rendement_annuel * 100:.4f}%")

    with col1:
        st.metric("Rendement mensuel attendu", f"{expected_return * 100:.4f}%")
    
    with col3:
        st.metric("📉 Risque (volatilité annualisée)", f"{risque_annuel * 100:.2f}%")

    st.write(f"📊 Diversification réelle : poids max = {poids_max * 100:.0f}%, corrélation max = {correlation_max}")

        # Résumé allocation
    st.subheader("💼 Détail du portefeuille optimal")
    allocation = pd.DataFrame({
        "Titre": portefeuille.index,
        "Poids (%)": (portefeuille * 100).round(2),
        "Montant investi (FCFA)": (portefeuille * capital).round(0).astype(int)
    }).set_index("Titre")
    st.dataframe(allocation)



###############################################"



#''''''''''''''''''''''''''''''''''''''''''''''''''
# --- Interface Streamlit ---
st.title("📊 Analyse des Actions Rendement & Volatilité (Janv-Juin 2025)")

# 1. Top 5 Rendements
#st.subheader("Top 5 des actions par rendement")
top_rendements = Top_5_Action(df)

fig_rendement = px.bar(
    x=top_rendements.values,
    y=top_rendements.index,
    orientation='h',
    labels={'x': 'Rendement', 'y': 'Action'},
    title="Top 5 des actions par rendement",
    #text=[f"{v:.2%}" for v in top_rendements.values],
    color=top_rendements.values,
    color_continuous_scale='viridis'
)
fig_rendement.update_layout(yaxis=dict(autorange="reversed"))



# 2. Top 5 Volatilité et Coefficient de Variation
top_cv, top_vol = top_cv_volatilite_graph(df)

# Volatilité
#st.subheader("Top 5 des actions les plus volatiles")
fig_vol = px.bar(
    x=top_vol.values,
    y=top_vol.index,
    orientation='h',
    labels={'x': 'Volatilité', 'y': 'Action'},
    title="Top 5 des actions les plus volatiles",
    #text=[f"{v:.2%}" for v in top_vol.values],
    color=top_vol.values,
    color_continuous_scale='viridis'
)
fig_vol.update_layout(yaxis=dict(autorange="reversed"))


# CV
#st.subheader("Top 5 des actions avec le plus fort coefficient de variation")
fig_cv = px.bar(
    x=top_cv.values,
    y=top_cv.index,
    orientation='h',
    labels={'x': 'Coefficient de Variation', 'y': 'Action'},
    title="Top 5 fort coefficient de variation",
    #text=[f"{v:.2f}" for v in top_cv.values],
    color=top_cv.values,
    color_continuous_scale='plasma'
)
fig_cv.update_layout(yaxis=dict(autorange="reversed"))



# Mise en page Streamlit
col1 ,col2 = st.columns([1,1])

with col1:
     st.plotly_chart(fig_vol, use_container_width=True) 
     #st.plotly_chart(fig_cv, use_container_width=True)

with col2:
    st.plotly_chart(fig_rendement, use_container_width=True)

st.divider()

Titre = st.selectbox("Selectionné un titre ", df.columns)
donné = df[Titre]
#redement 
rendement = donné.pct_change().fillna(0)
#st.write(rendement)
#graphe redement 
fig = px.line(rendement, title='Evolution rendement')
st.plotly_chart(fig)


st.title("Optimisation de portefeuille")

capital = st.number_input("Capital disponible (FCFA)", value=10_000_000)
cible_rendement = st.slider("Rendement cible annuel (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5) / 100
#poids_max = st.slider("Poids maximum par action (%)", min_value=5, max_value=100, value=20, step=5) / 100
#correlation_max = st.slider("Corrélation maximale autorisée", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

if st.button("Lancer l'optimisation"):
    portefeuille_streamlit(df, capital, cible_rendement, poids_max=0.20, correlation_max=0.35)