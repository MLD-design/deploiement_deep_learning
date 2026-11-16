import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

from style import hero, result_box

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = "cnn_model_cifar10.keras"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

hero("🖼️", "CIFAR-10",
     "Réseau convolutif : reconnaît l'une des 10 catégories (image redimensionnée en 32×32).")

if not (MODELS / MODEL_FILE).exists():
    st.error(f"Fichier introuvable : `models/{MODEL_FILE}`. Copie ton modèle dans le dossier `models`.")
    st.stop()


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger(nom_fichier):
    from tensorflow.keras.models import load_model
    return load_model(MODELS / nom_fichier)


model = charger(MODEL_FILE)

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("📤 Image")
    f = st.file_uploader("Charge une image", type=["jpg", "jpeg", "png"],
                         label_visibility="collapsed")
    st.caption("Catégories : " + ", ".join(CLASSES))
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img, caption="Image chargée", use_container_width=True)

with right:
    st.subheader("📈 Résultat")
    if not f:
        result_box("info", "En attente d'une image", "Charge une image à gauche.")
    else:
        x = np.array(img.resize((32, 32)), dtype="float32") / 255.0
        p = model.predict(x[np.newaxis, ...], verbose=0)[0]
        top = np.argsort(p)[::-1][:3]
        result_box("ok", CLASSES[top[0]], f"Confiance du modèle : {p[top[0]]:.1%}")
        st.markdown("**Top 3 des prédictions**")
        st.bar_chart(pd.DataFrame({"Probabilité": [float(p[i]) for i in top]},
                                  index=[CLASSES[i] for i in top]))