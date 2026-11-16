import streamlit as st
from style import hero, card

hero("🧠", "Plateforme de modèles Deep Learning",
     "Cinq modèles, un seul site : données tabulaires, images et radiographies pulmonaires.")

st.subheader("Choisis un modèle dans le menu à gauche 👈")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    card("📊", "Performance étudiants",
         "Estime l'indice de performance d'un étudiant.", "DNN · Régression")
with c2:
    card("💳", "Banque",
         "Prédit la réponse d'un client à partir de son profil.", "DNN · Classification")
with c3:
    card("🖼️", "CIFAR-10",
         "Reconnaît l'objet présent sur une image (10 catégories).", "CNN · 32×32")

c4, c5, _ = st.columns(3, gap="medium")
with c4:
    card("🩻", "Tuberculose",
         "Analyse une radiographie thoracique.", "VGG16 · Transfer learning")
with c5:
    card("🩺", "Pneumonie",
         "Analyse une radiographie thoracique.", "VGG16 · Transfer learning")

st.info("💡 Les pages médicales sont éducatives et ne remplacent pas un avis médical.")