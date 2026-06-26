"""
Extraction de features pour le modèle XGBoost.
Réplique EXACTEMENT la fonction `extract_high_performance_features` du notebook
(resolution.ipynb) : histogramme HSV (128 features) + texture Sobel par grille
4x4 (32 features) = 160 features, sur une image redimensionnée en 128x128.

⚠️ Ne JAMAIS modifier cette logique sans modifier aussi le notebook,
sinon le modèle XGBoost recevra des features incohérentes avec celles
utilisées pendant l'entraînement.
"""

import cv2
import numpy as np
from PIL import Image


def extract_high_performance_features(pil_image: Image.Image) -> np.ndarray:
    """Prend une image PIL (RGB) et retourne un vecteur de 160 features."""
    # PIL est en RGB, le notebook utilise cv2 (BGR) -> on convertit pour rester cohérent
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))

    # 1. Histogramme de couleur HSV (8 x 4 x 4 = 128 features)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 255, 0, 255])
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()

    # 2. Texture (gradients Sobel) sur une grille 4x4 (32 features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    texture_features = []
    h_step, w_step = 32, 32
    for i in range(4):
        for j in range(4):
            region = magnitude[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            texture_features.append(np.mean(region))
            texture_features.append(np.std(region))

    return np.concatenate([hist_hsv, texture_features]).astype("float32")
