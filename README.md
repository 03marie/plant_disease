# Application de détection de maladies de plantes — Guide rapide

## ⚠️ Fichier manquant : modèle XGBoost
Ton modèle CNN (`best_cnn_model.keras`) est déjà inclus et testé dans ce
dossier. Il ne manque que **`final_xgb_model.json`** (le modèle XGBoost
entraîné, sauvegardé via `final_xgb_model.save_model(...)` dans ton
notebook) — `best_xgb_params.json` que tu m'as envoyé ne contient que les
hyperparamètres, pas le modèle.

Retrouve ce fichier (Colab, Drive, ou ton dossier API) et place-le dans
`models/`. Sans lui, seul le CNN sera utilisable dans le sélecteur.

## 1. Structure du projet
```
plant_app/
├── app.py              # interface Streamlit (point d'entrée)
├── config.py            # ⚠️ À PERSONNALISER (modèles, classes, risques)
├── gradcam.py            # explicabilité visuelle (Grad-CAM)
├── requirements.txt
└── models/               # mets ici tes fichiers .h5 / .keras
    ├── cnn_baseline.h5
    └── mobilenetv2.h5
```

## 2. Installation
```bash
cd plant_app
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Étapes pour personnaliser (fichier `config.py` UNIQUEMENT)

### a) Copier tes modèles
Place tes fichiers `.h5` / `.keras` dans le dossier `models/`, puis mets à jour
le dictionnaire `MODELS` avec :
- le nom à afficher dans le menu déroulant
- le chemin du fichier
- le nom de la dernière couche convolutive (visible avec `model.summary()`)
  → nécessaire pour Grad-CAM

### b) Mettre les bons noms de classes
Très important : l'ordre doit être EXACTEMENT celui utilisé pendant
l'entraînement. Dans ton notebook, récupère-le ainsi :
```python
import json
class_names = list(train_generator.class_indices.keys())
json.dump(class_names, open("class_names.json", "w"))
```
Puis colle la liste dans `CLASS_NAMES` (ou charge le fichier JSON directement).

### c) Ajuster les niveaux de risque si besoin
Le dictionnaire `RISK_LEVELS` associe des mots-clés (ex: "blight", "rust",
"healthy") à un niveau de risque + conseil. Adapte selon tes cultures.

## 4. Lancer l'application
```bash
streamlit run app.py
```
Une page va s'ouvrir dans le navigateur (par défaut http://localhost:8501).

## 5. Déploiement (optionnel, gratuit)
1. Pousse le dossier `plant_app/` sur un repo GitHub (inclure les modèles si
   < 100 Mo, sinon utiliser Git LFS ou un stockage externe comme Google Drive
   + téléchargement au démarrage).
2. Va sur https://share.streamlit.io
3. Connecte ton repo GitHub, choisis `app.py`, déploie.

## 6. Pour ton rapport / soutenance
- La section Grad-CAM répond directement à l'exigence "affichage du résultat
  avec explication de la décision".
- Le sélecteur de modèle te permet de montrer en live les différences de
  comportement entre tes modèles benchmarkés (utile pour ton tableau
  comparatif Accuracy / F1 / temps / taille).
- Pense à capturer des captures d'écran de l'appli pour ton notebook /
  rapport final.

## Problèmes fréquents
| Problème | Cause probable | Solution |
|---|---|---|
| Erreur Grad-CAM "layer not found" | mauvais nom de couche | vérifier avec `model.summary()` |
| Prédictions incohérentes | mauvais ordre dans `CLASS_NAMES` | vérifier `class_indices` du générateur d'entraînement |
| Modèle très lent à charger | gros fichier `.h5` | c'est normal au premier chargement, ensuite mis en cache |
| `ModuleNotFoundError: cv2` | opencv mal installé | `pip install opencv-python-headless` |
