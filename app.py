"""
Application Streamlit - Détection de maladies de plantes
==========================================================
Gère deux types de modèles pour le benchmark :
- CNN (MobileNetV2, transfer learning) -> Grad-CAM disponible
- XGBoost (features HSV + texture faites main) -> pas de Grad-CAM
  (affiche à la place les features les plus discriminantes)

Auteur : Marie-Grégory (IMSP - M1 Data Science)
"""

import time
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from xgboost import XGBClassifier

from config import MODELS, CLASS_NAMES, RISK_LEVELS, IMAGE_SIZE
from gradcam import make_gradcam_heatmap, overlay_heatmap
from xgb_features import extract_high_performance_features

# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Détection de maladies de plantes",
    page_icon="🌿",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Chargement des modèles (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_xgb_model(model_path: str):
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def preprocess_for_cnn(image: Image.Image, target_size):
    image_resized = image.convert("RGB").resize(target_size)
    array = img_to_array(image_resized) / 255.0
    return np.expand_dims(array, axis=0), image_resized


def get_risk_info(class_name: str):
    for keyword, info in RISK_LEVELS.items():
        if keyword.lower() in class_name.lower():
            return info
    return RISK_LEVELS.get("default", {"level": "Indéterminé", "color": "gray", "advice": "—"})


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ----------------------------------------------------------------------------
# Barre latérale
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
model_label = st.sidebar.selectbox("Choisir le modèle :", list(MODELS.keys()))
model_info = MODELS[model_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **À propos**
    Cette application compare deux approches pour détecter les
    maladies de plantes à partir d'une photo de feuille :
    - un **CNN** (deep learning, transfer learning MobileNetV2)
    - un **XGBoost** (machine learning classique sur features
      faites main : histogramme de couleur + texture)

    Téléversez une image pour voir leurs prédictions et,
    pour le CNN, une explication visuelle (Grad-CAM).
    """
)

# ----------------------------------------------------------------------------
# Titre principal
# ----------------------------------------------------------------------------
st.title("🌿 Détection de maladies de plantes")
st.write(
    "Téléversez une photo de feuille pour obtenir un diagnostic automatique, "
    "le niveau de risque, et une explication de la prédiction."
)

uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------------------------------
# Traitement principal
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image soumise")
        st.image(image, use_container_width=True)

    with st.spinner(f"Analyse en cours avec **{model_label}**..."):

        if model_info["type"] == "keras":
            try:
                model = load_keras_model(model_info["path"])
            except Exception as e:
                st.error(f"Impossible de charger le modèle '{model_label}': {e}")
                st.stop()

            img_array, resized_img = preprocess_for_cnn(image, IMAGE_SIZE)

            start = time.time()
            predictions = model.predict(img_array, verbose=0)[0]
            inference_time = time.time() - start

            gradcam_available = model_info["last_conv_layer"] is not None
            if gradcam_available:
                try:
                    top_idx = int(np.argmax(predictions))
                    heatmap = make_gradcam_heatmap(
                        img_array, model, model_info["last_conv_layer"], top_idx
                    )
                    overlayed = overlay_heatmap(np.array(resized_img), heatmap)
                    gradcam_ok = True
                except Exception as e:
                    gradcam_ok = False
                    gradcam_error = str(e)
            else:
                gradcam_ok = False
                gradcam_error = "Grad-CAM non applicable à ce type de modèle."

        elif model_info["type"] == "xgboost":
            try:
                model = load_xgb_model(model_info["path"])
            except Exception as e:
                st.error(
                    f"Impossible de charger le modèle XGBoost : {e}\n\n"
                    f"Vérifie que le fichier `{model_info['path']}` (modèle ENTRAÎNÉ, "
                    f"pas seulement les hyperparamètres) est bien présent dans `models/`."
                )
                st.stop()

            features = extract_high_performance_features(image).reshape(1, -1)

            start = time.time()
            raw_scores = model.predict_proba(features)[0]
            inference_time = time.time() - start
            predictions = raw_scores  # déjà des probabilités

            gradcam_ok = False
            gradcam_error = "Le modèle XGBoost n'a pas de couches convolutives : Grad-CAM ne s'applique pas."
            resized_img = image.convert("RGB").resize((300, 300))

        else:
            st.error(f"Type de modèle inconnu : {model_info['type']}")
            st.stop()

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        risk_info = get_risk_info(predicted_class)

    with col2:
        st.subheader("Résultat du diagnostic")
        st.markdown(f"### 🔍 {predicted_class.replace('_', ' ').replace('  ', ' ')}")
        st.metric("Confiance du modèle", f"{confidence * 100:.1f} %")

        risk_color = risk_info["color"]
        st.markdown(
            f"""
            <div style="padding:10px;border-radius:8px;background-color:{risk_color}22;
                        border:1px solid {risk_color};">
                <b>Niveau de risque :</b>
                <span style="color:{risk_color};font-weight:bold;">{risk_info['level']}</span>
                <br><small>{risk_info['advice']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Modèle utilisé : {model_label} • Temps d'inférence : {inference_time*1000:.0f} ms")

    st.markdown("---")

    # ---- Top-5 des classes les plus probables ----
    st.subheader("📊 Probabilités détaillées")
    top5_idx = np.argsort(predictions)[::-1][:5]
    top5_data = {CLASS_NAMES[i].replace("_", " "): float(predictions[i]) for i in top5_idx}
    st.bar_chart(top5_data)

    # ---- Explication visuelle ----
    st.subheader("🧠 Pourquoi cette prédiction ?")
    if gradcam_ok:
        col3, col4 = st.columns(2)
        with col3:
            st.image(resized_img, caption="Image analysée", use_container_width=True)
        with col4:
            st.image(overlayed, caption="Zones ayant influencé la décision", use_container_width=True)
        st.caption(
            "Les zones en rouge/jaune sont celles que le modèle a le plus regardées "
            "pour prendre sa décision (Grad-CAM)."
        )
    else:
        st.info(gradcam_error)
        if model_info["type"] == "xgboost":
            st.caption(
                "Pour ce modèle, la décision repose sur 160 features faites main : "
                "un histogramme de couleur (teinte/saturation/valeur) et des mesures "
                "de texture (gradients Sobel) calculées sur 16 zones de l'image. "
                "Une teinte dominante anormale ou une texture irrégulière dans "
                "certaines zones augmente la probabilité d'une classe malade."
            )

else:
    st.info("👆 Téléversez une image de feuille pour démarrer l'analyse.")
