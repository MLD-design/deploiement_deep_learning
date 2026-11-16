import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from style import hero, result_box

DOSSIER = Path(__file__).resolve().parent.parent / "models" / "student"
SCALER_FILE = "regression_scaler.joblib"
ENCODER_FILE = "regression_encoders.joblib"

# Plage observée à l'entraînement (à vérifier avec df.describe() dans ton notebook)
LIMITES = {
    "Hours Studied": (1, 9),
    "Previous Scores": (40, 100),
    "Sleep Hours": (4, 9),
    "Sample Question Papers Practiced": (0, 9),
}

hero("📊", "Student Performance · Performance des étudiants",
     "Neural network regression · Régression par réseau de neurones : "
     "estimates the Performance Index (0–100) · estime l'indice de performance.")

# Le fichier du modèle est détecté automatiquement (.keras ou .h5)
modeles = sorted(list(DOSSIER.glob("*.keras")) + list(DOSSIER.glob("*.h5")))
if not modeles:
    st.error(f"Aucun modèle `.keras` ou `.h5` trouvé dans `{DOSSIER}`.")
    st.stop()
for f in (SCALER_FILE, ENCODER_FILE):
    if not (DOSSIER / f).exists():
        st.error(f"Fichier introuvable : `models/student/{f}`. "
                 f"Fichiers présents : {[p.name for p in DOSSIER.iterdir()]}")
        st.stop()


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger(dossier, model_name, scaler_name, encoder_name):
    from tensorflow.keras.models import load_model
    d = Path(dossier)
    return (load_model(d / model_name),
            joblib.load(d / scaler_name),
            joblib.load(d / encoder_name))


model, scaler, encoders = charger(str(DOSSIER), modeles[0].name, SCALER_FILE, ENCODER_FILE)

# L'encoder peut être un seul encoder ou un dictionnaire {colonne: encoder}
if isinstance(encoders, dict):
    cle = "Extracurricular Activities" if "Extracurricular Activities" in encoders else next(iter(encoders))
    enc = encoders[cle]
else:
    enc = encoders
options = list(getattr(enc, "classes_", ["Yes", "No"]))
FR = {"Yes": "Yes · Oui", "No": "No · Non"}

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.subheader("📝 Student profile · Profil de l'étudiant")
    with st.form("student"):
        a, b = st.columns(2)
        hours = a.number_input("Hours Studied · Heures d'étude par jour", 0, 24, 6,
                               help="Plage d'entraînement : 1 à 9 h")
        prev = b.number_input("Previous Scores · Notes précédentes (/100)", 0, 100, 70,
                              help="Plage d'entraînement : 40 à 100")
        sleep = a.number_input("Sleep Hours · Heures de sommeil par nuit", 0, 24, 7,
                               help="Plage d'entraînement : 4 à 9 h")
        papers = b.number_input("Sample Question Papers Practiced · Sujets d'examen pratiqués",
                                0, 50, 5, help="Plage d'entraînement : 0 à 9")
        extra = st.radio("Extracurricular Activities · Activités extrascolaires",
                         options, horizontal=True, format_func=lambda v: FR.get(v, v))
        go = st.form_submit_button("🔮 Predict · Prédire", type="primary",
                                   use_container_width=True)

with right:
    st.subheader("📈 Result · Résultat")
    if not go:
        result_box("info", "Waiting · En attente",
                   "Fill in the form, then click Predict · Remplis le formulaire puis clique sur « Predict ».")
    else:
        try:
            ligne = {
                "Hours Studied": hours,
                "Previous Scores": prev,
                "Extracurricular Activities": enc.transform([extra])[0],
                "Sleep Hours": sleep,
                "Sample Question Papers Practiced": papers,
            }
            hors = [k for k, (lo, hi) in LIMITES.items() if not lo <= ligne[k] <= hi]
            if hors:
                st.warning("⚠️ Out of training range · Hors de la plage d'entraînement ("
                           + ", ".join(hors) + ") : prediction may be unreliable · "
                           "la prédiction peut être peu fiable.")
            colonnes = list(getattr(scaler, "feature_names_in_", ligne.keys()))
            X = scaler.transform(pd.DataFrame([ligne])[colonnes])
            pred = float(model.predict(X, verbose=0)[0][0])
            pred = max(0.0, min(100.0, pred))
            genre = "ok" if pred >= 70 else ("info" if pred >= 40 else "warn")
            result_box(genre, f"Performance Index · Indice de performance : {pred:.1f} / 100",
                       "Model estimate based on the values entered · "
                       "Estimation du modèle à partir des valeurs saisies.")
            st.progress(pred / 100)
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")