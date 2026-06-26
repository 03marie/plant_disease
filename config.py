"""
Configuration centrale de l'application.
==========================================
=> C'EST LE SEUL FICHIER QUE TU DOIS ADAPTER À TON PROJET.

1. MODELS       : chemins vers tes fichiers .h5 / .keras
2. CLASS_NAMES  : liste des classes dans le MÊME ORDRE que pendant l'entraînement
                  (généralement train_generator.class_indices ou os.listdir(train_dir))
3. RISK_LEVELS  : mapping mot-clé -> niveau de risque affiché
"""

IMAGE_SIZE = (224, 224)  # adapte selon la taille utilisée à l'entraînement

# ----------------------------------------------------------------------------
# 1. MODÈLES DISPONIBLES
# ----------------------------------------------------------------------------
# "last_conv_layer" = nom de la dernière couche Conv2D du modèle (sert au Grad-CAM).
# Pour le trouver : model.summary()  -> repère la dernière couche "conv2d_xx" ou
# le nom du dernier bloc conv de ton backbone (ex: "top_conv" pour EfficientNet,
# "conv5_block3_out" pour ResNet50, "block5_conv3" pour VGG16).
MODELS = {
    "CNN (MobileNetV2 - transfer learning)": {
        "path": "models/best_cnn_model.keras",
        "last_conv_layer": "out_relu",
        "type": "keras",
    },
    "XGBoost (features HSV + texture)": {
        "path": "models/final_xgb_model.json",
        "last_conv_layer": None,  # pas de Grad-CAM pour XGBoost (pas de couches conv)
        "type": "xgboost",
    },
}

# ----------------------------------------------------------------------------
# 2. NOMS DES CLASSES (exemple : jeu de données PlantVillage)
# ----------------------------------------------------------------------------
# IMPORTANT : remplace cette liste par la tienne, dans le même ordre que lors
# de l'entraînement. Tu peux la récupérer avec :
#   import json
#   class_names = list(train_generator.class_indices.keys())
#   json.dump(class_names, open("class_names.json", "w"))
# puis charger ce json ici.
CLASS_NAMES = sorted([
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
])
# ⚠️ VÉRIFICATION OBLIGATOIRE : ouvre ton dossier d'entraînement et lance
#   sorted(os.listdir(data_dir)) == CLASS_NAMES
# pour confirmer que l'ordre correspond EXACTEMENT à celui utilisé par
# sklearn.datasets.load_files (tri alphabétique des noms de dossiers).
# Si un seul nom de dossier diffère légèrement (espace, underscore, virgule),
# tout le mapping classe -> prédiction sera décalé.

# ----------------------------------------------------------------------------
# 3. NIVEAUX DE RISQUE
# ----------------------------------------------------------------------------
# La clé est cherchée (insensible à la casse) dans le nom de la classe prédite.
# "default" s'applique si rien ne correspond.
RISK_LEVELS = {
    "healthy": {
        "level": "Aucun risque",
        "color": "#2e7d32",
        "advice": "La plante semble saine. Continuez une surveillance régulière.",
    },
    "rust": {
        "level": "Risque modéré",
        "color": "#f9a825",
        "advice": "Rouille détectée : améliorer l'aération et traiter avec un fongicide adapté.",
    },
    "blight": {
        "level": "Risque élevé",
        "color": "#d32f2f",
        "advice": "Mildiou/brûlure détecté : isoler les plants atteints et traiter rapidement.",
    },
    "rot": {
        "level": "Risque élevé",
        "color": "#d32f2f",
        "advice": "Pourriture détectée : retirer les parties atteintes et désinfecter les outils.",
    },
    "spot": {
        "level": "Risque modéré",
        "color": "#f9a825",
        "advice": "Taches foliaires détectées : surveiller l'évolution et limiter l'humidité.",
    },
    "mold": {
        "level": "Risque modéré",
        "color": "#f9a825",
        "advice": "Moisissure détectée : améliorer la ventilation autour des plants.",
    },
    "default": {
        "level": "À vérifier",
        "color": "#757575",
        "advice": "Maladie détectée, consultez un spécialiste agricole pour confirmation.",
    },
}
