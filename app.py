import streamlit as st
from style import inject_css

st.set_page_config(page_title="Plateforme Deep Learning", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

pages = [
    st.Page("views/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("views/performance_etudiants.py", title="Performance étudiants", icon="📊"),
    st.Page("views/banque.py", title="Banque", icon="💳"),
    st.Page("views/cifar10.py", title="CIFAR-10", icon="🖼️"),
    st.Page("views/tuberculose.py", title="Tuberculose", icon="🩻"),
    st.Page("views/pneumonie.py", title="Pneumonie", icon="🩺"),
]

navigation = st.navigation(pages)
navigation.run()