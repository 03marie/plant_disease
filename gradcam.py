"""
Grad-CAM : explique visuellement pourquoi le modèle a pris sa décision.
Fonctionne avec n'importe quel modèle Keras CNN (baseline ou transfer learning).
"""

import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Génère la heatmap Grad-CAM pour une image donnée.

    img_array : image prétraitée, shape (1, H, W, 3)
    model : modèle Keras chargé
    last_conv_layer_name : nom de la dernière couche convolutive
    pred_index : index de la classe à expliquer (par défaut : classe prédite)
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Superpose la heatmap Grad-CAM sur l'image originale.

    original_image : image RGB en array numpy (H, W, 3), valeurs 0-255
    heatmap : heatmap normalisée (0-1) issue de make_gradcam_heatmap
    """
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    if original_image.dtype != np.uint8:
        original_image = np.uint8(original_image)

    overlayed = colored_heatmap * alpha + original_image * (1 - alpha)
    return np.uint8(overlayed)
