import streamlit as st
from pathlib import Path
from PIL import Image

# Interface utilisateur avec Streamlit
st.set_page_config(
    layout="wide",
    page_title=" Data Analyst sur Film",
    page_icon="🎬"  # Emoji Unicode directement
)

# Conteneur pour aligner les éléments horizontalement
col1, col2, col3 = st.columns([1, 4, 1])

# Colonne gauche : Image
with col1:
    st.image(
        "https://media.licdn.com/dms/image/v2/D4E03AQF7cVN_iger2w/profile-displayphoto-shrink_800_800/B4EZdURvE.HsAc-/0/1749465625806?e=1755734400&v=beta&t=4FXq1wVFGgbqDEOVVw-MHUZt9wkZWEx0kndiMZQqMwo",  # Remplacez par le chemin de votre image
        width=80,     # Ajustez la taille si nécessaire
        use_container_width=False,
    )

# Colonne centrale : Titre
with col2:
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>Exploration des Données MovieLens</h1>
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


st.title("Presentation du projet")
st.markdown(" Ce projet a pour objectif de concevoir une architecture API en Python, robuste et modulaire, " \
"dédiée à la gestion, à l’analyse et à la visualisation des données MovieLens." \
" Il s’appuie sur une stack technologique moderne comprenant FastAPI pour la création de l’API, " \
"SQLAlchemy pour la gestion de la base de données SQLite, Pydantic pour la validation des données," \
" ainsi que des bibliothèques d’analyse et de visualisation telles que Pandas, NumPy et Plotly. " \
"L’ensemble vise à offrir une interface performante et flexible, facilitant l’exploitation des données MovieLens.")

st.write(" ")
st.write(" ")
# Titre
st.markdown("# **Phase 1 : Conception de l’architecture et développement de l’API en Python**")
# Afficher l'image séparément


st.image("https://github.com/Anee1/Analyse_Films/blob/main/App_streamlit/architecture.png?raw=true", use_container_width=True)

st.markdown(
        """
        <a href="https://github.com/Anee1/Moovie_Backend" target="_blank">
            <button style="background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-size: 16px;">
                📦 Cliquer pour voir le Code de la Phase 1
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

st.write(" ")
st.write(" ")
st.write(" ")


# Titre
st.markdown("# **Phase 2 : Analyse exploratoire et visualisation statistique**")
# Afficher l'image séparément
st.image("https://github.com/Anee1/Analyse_Films/blob/main/App_streamlit/architecturephase.png?raw=true", use_container_width=True)

st.markdown(
        """
        <a href="https://github.com/Anee1/Analyse_Films" target="_blank">
            <button style="background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-size: 16px;">
                📊 Cliquer pour voir le Code de la Phase 2
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )