import streamlit as st

# Navigation
page_0 = st.Page("page0.py", title="Acceuil", icon="🏠")  
page_1 = st.Page("page1.py", title="Analyse des titres", icon="📈")     # Film clapperboard
#page_2 = st.Page("page2.py", title=" Statistiques des tags", icon="📊")  # Bar chart
#page_3 = st.Page("page3.py", title="Catalogue de films", icon="🔎") # Magnifying glass

pg = st.navigation([page_0, page_1])
pg.run()