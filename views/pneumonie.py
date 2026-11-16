import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

from style import hero, result_box

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = "vgg16_model_pneumonia.keras"
CLASSES = ("Normal", "Pneumonie")     # classe 0, classe 1
SIZE = 150
PREPROCESS = "rescale"                # "rescale" (/255) ou "vgg16" (preprocess_input)

hero("🩺", "Détection de pneumonie",
     "Transfer learning VGG16 : analyse d'une radiographie thoracique (150×150).")
st.warning("⚠️ Outil éducatif : il ne remplace pas l'avis d'un professionnel de santé.")

if not (MODELS / MODEL_FILE).exists():
    st.error(f"Fichier introuvable : `models/{MODEL_FILE}`.")
    st.stop()


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger(nom_fichier):
    from tensorflow.keras.models import load_model
    return load_model(MODELS / nom_fichier)


def preparer(img):
    x = np.array(img.convert("RGB").resize((SIZE, SIZE)), dtype="float32")
    if PREPROCESS == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
    else:
        x = x / 255.0
    return x[np.newaxis, ...]


model = charger(MODEL_FILE)

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("📤 Radiographie")
    f = st.file_uploader("Charge une image", type=["jpg", "jpeg", "png"],
                         label_visibility="collapsed")
    with st.expander("⚙️ Réglages avancés"):
        seuil = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05)
    if f:
        img = Image.open(f)
        st.image(img, caption="Image chargée", use_container_width=True)

with right:
    st.subheader("📈 Résultat")
    if not f:
        result_box("info", "En attente d'une image", "Charge une radiographie à gauche.")
    else:
        with st.spinner("Analyse en cours..."):
            sortie = model.predict(preparer(img), verbose=0)[0]
        p1 = float(sortie[0]) if sortie.size == 1 else float(sortie[1])
        p0 = 1 - p1
        if p1 >= seuil:
            result_box("warn", f"{CLASSES[1]} probable", f"Confiance du modèle : {p1:.1%}")
        else:
            result_box("ok", CLASSES[0], f"Confiance du modèle : {p0:.1%}")
        c1, c2 = st.columns(2)
        c1.metric(CLASSES[0], f"{p0:.1%}")
        c2.metric(CLASSES[1], f"{p1:.1%}")
        st.progress(min(max(p1, 0.0), 1.0), text=f"Probabilité « {CLASSES[1]} »")