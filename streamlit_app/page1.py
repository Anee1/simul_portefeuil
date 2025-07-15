import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Chargement des données ---
df = pd.read_csv("Action.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)

# --- Traitement des données ---
missing_stats = df.isnull().mean()
df = df[missing_stats[missing_stats <= 0.7].index]
df = df.ffill().bfill()
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
    color_continuous_scale='magma'
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
col1 ,col3 = st.columns([1,1])

with col1:
     st.plotly_chart(fig_vol, use_container_width=True) 
     st.plotly_chart(fig_cv, use_container_width=True)

with col3:
    st.plotly_chart(fig_rendement, use_container_width=True)

