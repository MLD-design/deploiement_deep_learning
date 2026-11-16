import joblib
import numpy as np
import streamlit as st
from pathlib import Path

from style import hero, result_box

DOSSIER = Path(__file__).resolve().parent.parent / "models" / "banking"
MODEL_FILE = "DNN_modele.keras"
SCALER_FILE = "scaler.joblib"
ENCODER_FILE = "encoders.joblib"

# Ordre exact des 15 colonnes vues par le scaler et le modèle
CATEGORIELLES = ["job", "marital", "education", "default", "housing", "loan",
                 "contact", "month", "day_of_week", "poutcome"]   # encoders[0..9] (encoders[10] = cible y)
ORDRE = ["age", "job", "marital", "education", "default", "housing", "loan",
         "contact", "month", "day_of_week", "duration", "campaign",
         "pdays", "previous", "poutcome"]

# Traduction des valeurs (affichage seulement : le modèle reçoit la valeur anglaise)
FR = {
    "admin.": "Administrative", "blue-collar": "Ouvrier", "entrepreneur": "Entrepreneur",
    "housemaid": "Employé(e) de maison", "management": "Cadre", "retired": "Retraité",
    "self-employed": "Indépendant", "services": "Services", "student": "Étudiant",
    "technician": "Technicien", "unemployed": "Sans emploi", "unknown": "Inconnu",
    "divorced": "Divorcé", "married": "Marié", "single": "Célibataire",
    "basic.4y": "Primaire (4 ans)", "basic.6y": "Primaire (6 ans)", "basic.9y": "Collège (9 ans)",
    "high.school": "Lycée", "illiterate": "Analphabète",
    "professional.course": "Formation professionnelle", "university.degree": "Université",
    "no": "Non", "yes": "Oui",
    "cellular": "Portable", "telephone": "Téléphone fixe",
    "apr": "Avril", "aug": "Août", "dec": "Décembre", "jul": "Juillet", "jun": "Juin",
    "mar": "Mars", "may": "Mai", "nov": "Novembre", "oct": "Octobre", "sep": "Septembre",
    "mon": "Lundi", "tue": "Mardi", "wed": "Mercredi", "thu": "Jeudi", "fri": "Vendredi",
    "failure": "Échec", "nonexistent": "Inexistant", "success": "Succès",
}
LIBELLES = {
    "job": "Job · Profession", "marital": "Marital status · Situation familiale",
    "education": "Education · Niveau d'études", "default": "Credit in default · Crédit en défaut",
    "housing": "Housing loan · Prêt immobilier", "loan": "Personal loan · Prêt personnel",
    "contact": "Contact type · Type de contact", "month": "Last contact month · Mois du dernier contact",
    "day_of_week": "Last contact day · Jour du dernier contact",
    "poutcome": "Previous campaign outcome · Résultat de la campagne précédente",
}
# Plage observée dans le jeu de données (pour avertir en cas de valeur inhabituelle)
LIMITES = {"age": (17, 98), "duration": (0, 4918), "campaign": (1, 56),
           "pdays": (0, 999), "previous": (0, 7)}


def fmt(v):
    return f"{v} · {FR[v]}" if v in FR else str(v)


hero("🏦", "Bank Marketing · Souscription bancaire",
     "Neural network classification · Classification par réseau de neurones : "
     "will the client subscribe to a term deposit? · le client va-t-il souscrire à un dépôt à terme ?")

for f in (MODEL_FILE, SCALER_FILE, ENCODER_FILE):
    if not (DOSSIER / f).exists():
        st.error(f"Fichier introuvable : `models/banking/{f}`. "
                 f"Fichiers présents : {[p.name for p in DOSSIER.iterdir()] if DOSSIER.exists() else 'dossier absent'}")
        st.stop()


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger(dossier, model_name, scaler_name, encoder_name):
    from tensorflow.keras.models import load_model
    d = Path(dossier)
    return (load_model(d / model_name),
            joblib.load(d / scaler_name),
            joblib.load(d / encoder_name))


model, scaler, encoders = charger(str(DOSSIER), MODEL_FILE, SCALER_FILE, ENCODER_FILE)
enc = dict(zip(CATEGORIELLES, encoders[:10]))

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("📝 Client profile · Profil du client")
    with st.form("bank"):
        st.markdown("**Client · Client**")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age · Âge", 17, 100, 40)
        valeurs = {}
        valeurs["job"] = c2.selectbox(LIBELLES["job"], list(enc["job"].classes_), format_func=fmt)
        valeurs["marital"] = c1.selectbox(LIBELLES["marital"], list(enc["marital"].classes_), format_func=fmt)
        valeurs["education"] = c2.selectbox(LIBELLES["education"], list(enc["education"].classes_), format_func=fmt)

        st.markdown("**Financial situation · Situation financière**")
        c1, c2, c3 = st.columns(3)
        valeurs["default"] = c1.selectbox(LIBELLES["default"], list(enc["default"].classes_), format_func=fmt)
        valeurs["housing"] = c2.selectbox(LIBELLES["housing"], list(enc["housing"].classes_), format_func=fmt)
        valeurs["loan"] = c3.selectbox(LIBELLES["loan"], list(enc["loan"].classes_), format_func=fmt)

        st.markdown("**Last contact · Dernier contact**")
        c1, c2, c3 = st.columns(3)
        valeurs["contact"] = c1.selectbox(LIBELLES["contact"], list(enc["contact"].classes_), format_func=fmt)
        valeurs["month"] = c2.selectbox(LIBELLES["month"], list(enc["month"].classes_), format_func=fmt)
        valeurs["day_of_week"] = c3.selectbox(LIBELLES["day_of_week"], list(enc["day_of_week"].classes_), format_func=fmt)
        duration = st.number_input("Duration · Durée de l'appel (secondes)", 0, 5000, 180)

        st.markdown("**Campaign history · Historique des campagnes**")
        c1, c2, c3 = st.columns(3)
        campaign = c1.number_input("Campaign · Contacts pendant cette campagne", 1, 60, 2)
        pdays = c2.number_input("Pdays · Jours depuis le dernier contact (999 = jamais)", 0, 999, 999)
        previous = c3.number_input("Previous · Contacts avant cette campagne", 0, 10, 0)
        valeurs["poutcome"] = st.selectbox(LIBELLES["poutcome"], list(enc["poutcome"].classes_), format_func=fmt)

        with st.expander("⚙️ Advanced · Réglages avancés"):
            seuil = st.slider("Decision threshold · Seuil de décision", 0.05, 0.95, 0.50, 0.05)
        go = st.form_submit_button("🔮 Predict · Prédire", type="primary", use_container_width=True)

with right:
    st.subheader("📈 Result · Résultat")
    if not go:
        result_box("info", "Waiting · En attente",
                   "Fill in the profile, then click Predict · Remplis le profil puis clique sur « Predict ».")
    else:
        try:
            brut = {"age": age, "duration": duration, "campaign": campaign,
                    "pdays": pdays, "previous": previous}
            hors = [k for k, (lo, hi) in LIMITES.items() if not lo <= brut[k] <= hi]
            if hors:
                st.warning("⚠️ Unusual values · Valeurs inhabituelles (" + ", ".join(hors) + ") : "
                           "prediction may be unreliable · la prédiction peut être peu fiable.")
            ligne = []
            for col in ORDRE:
                if col in enc:
                    ligne.append(float(enc[col].transform([valeurs[col]])[0]))
                else:
                    ligne.append(float(brut[col]))
            X = scaler.transform(np.array([ligne]))
            sortie = model.predict(X, verbose=0)[0]
            p = float(sortie[0]) if sortie.size == 1 else float(sortie[1])
            if p >= seuil:
                result_box("ok", "YES · OUI : subscription likely · souscription probable",
                           f"Probability · Probabilité : {p:.1%}")
            else:
                result_box("warn", "NO · NON : subscription unlikely · souscription peu probable",
                           f"Probability of YES · Probabilité du OUI : {p:.1%}")
            st.progress(min(max(p, 0.0), 1.0), text="Probability of YES · Probabilité du OUI")
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")