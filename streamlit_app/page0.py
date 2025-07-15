# Interface utilisateur avec Streamlit
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Portefeuille Optimal BRVM",
    page_icon="💼"
)

# Conteneur pour aligner les éléments horizontalement
col1, col2, col3 = st.columns([1, 4, 1])

# Colonne gauche : Image
with col1:
    st.image(
        "https://media.licdn.com/dms/image/v2/D4E03AQF7cVN_iger2w/profile-displayphoto-shrink_800_800/B4EZdURvE.HsAc-/0/1749465625806?e=1755734400&v=beta&t=4FXq1wVFGgbqDEOVVw-MHUZt9wkZWEx0kndiMZQqMwo",
        width=80,
        use_container_width=False,
    )

# Colonne centrale : Titre
with col2:
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>Construction d'un Portefeuille Optimal sur la BRVM</h1>
        """,
        unsafe_allow_html=True,
    )

# Colonne droite : Nom et lien LinkedIn
with col3:
    st.markdown(
        """
        <div style='text-align: right;'>
            <a href="https://www.linkedin.com/in/marcel-an%C3%A9e-2aa3091bb/" target="_blank" style='text-decoration: none; color: #0077b5;'>
                <strong>ANEE MARCEL</strong>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write(" ")
st.write(" ")

# Section de présentation du projet
st.title("Présentation du projet")

st.markdown(
    """
    Ce projet vise à construire un **portefeuille optimal d'actions cotées sur la BRVM**, basé sur une approche quantitative
    et des contraintes de performance et de diversification.

    L’objectif est de :
    - Maximiser la diversification
    - Garantir un rendement cible annuel
    - Minimiser le risque (volatilité) du portefeuille
    - Contrôler la corrélation entre les titres sélectionnés
    - Respecter un poids maximal par actif

    Nous utilisons les bibliothèques **Pandas, NumPy, Plotly** pour l'analyse de données, et **cvxpy** pour la modélisation et
    la résolution du problème d'optimisation quadratique sous contraintes.
    """
)

st.write(" ")
st.write(" ")

# Phase 1 - Collecte et Préparation
st.markdown("## **Phase 1 : Collecte et Préparation des Données**")
st.markdown(
    """
    - Les données de cours sont collectées à partir des historiques des actions BRVM.
    - Les colonnes avec trop de valeurs manquantes sont supprimées.
    - Les données sont remplies (`forward fill`, `backward fill`) et converties en pourcentages de rendement mensuel.
    - La période d’analyse porte sur **les données depuis juillet 2024 jusqu’à aujourd’hui**.
    """
)

# Phase 2 - Filtrage
st.markdown("## **Phase 2 : Filtrage des Actifs**")
st.markdown(
    """
    - Seules les actions dont le **rendement cumulé** sur la période est supérieur à **10%** sont retenues.
    - Ensuite, on calcule la **corrélation moyenne** de chaque actif avec les autres.
    - Les actions dont la corrélation moyenne dépasse un seuil (ex. 0.35) sont exclues.
    """
)

# Phase 3 - Optimisation
st.markdown("## **Phase 3 : Optimisation du Portefeuille**")
st.markdown(
    """
    - Le problème d’optimisation est formulé pour **minimiser le risque** (volatilité du portefeuille),
      sous plusieurs contraintes :
        - Poids total des actifs = 100%
        - Rendement attendu ≥ rendement cible (converti en mensuel)
        - Poids de chaque actif ≤ limite imposée (ex. 25%)
    - Nous utilisons `cvxpy` pour résoudre ce problème d'optimisation quadratique.
    - À l’issue de l’optimisation, seuls les titres avec un poids strictement positif sont retenus dans le portefeuille.
    """
)

# Phase 4 - Résultats
st.markdown("## **Phase 4 : Résultats Affichés**")
st.markdown(
    """
    - 📈 **Rendement annuel attendu** du portefeuille
    - 📉 **Risque (volatilité annualisée)** minimal obtenu
    - 📊 **Tableau du portefeuille optimal** avec les titres sélectionnés, leurs poids et le montant à allouer
    - ✅ L’utilisateur peut ajuster dynamiquement le capital, le rendement cible, le poids maximal ou la corrélation autorisée
    """
)

st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px; color:gray;">
        Application développée dans le cadre d'un projet d'un entretien.
    </p>
    """,
    unsafe_allow_html=True,
)
