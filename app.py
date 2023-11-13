import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as eff_preprocess_input


st.set_page_config(layout="wide")

# Obtenez le répertoire du script en cours d'exécution
script_dir = os.path.dirname(os.path.abspath(__file__))

# Affichage des classes du jeu de données
mapping_file_path = os.path.join(script_dir, 'LOC_synset_mapping.txt')

# Chemin vers les images à afficher
images_folder_path = os.path.join(script_dir, 'img\\Display')

# Affichage des 10 premières classes du dataset
with open(mapping_file_path, 'r') as mapping_file:
    first_10_lines = [next(mapping_file) for _ in range(10)]

# Charger les modèles pré-entrainés
vgg_model = VGG16(weights='imagenet')
eff_model = EfficientNetB0(weights='imagenet')

# Titre de la page
st.title("Classification d'images avec VGG16 et EfficientNet")

# Affichage des images du dataset
st.header("Analyse exploratoire des données")

st.subheader("1. Exemples d'images du jeu de données d'entraînement pour les 10 premières classes")

# Liste des fichiers d'images dans le sous-dossier
image_files = os.listdir(images_folder_path)

# Afficher toutes les images avec leur titre
for i, image_file in enumerate(image_files):
    image_path = os.path.join(images_folder_path, image_file)
    
    # Charger l'image avec PIL
    image = Image.open(image_path)
    
    # Afficher l'image avec le titre correspondant
    st.image(image, caption=f"Image {i + 1}", use_column_width=True)
    

# Afficher les 10 premières lignes du fichier 'LOC_synset_mapping.txt'
st.subheader("2. Classes d'images du jeu d'entraînement")
for line in first_10_lines:
    st.write(line.strip())


st.header("Prédiction de la classe d'une image")

# Sélectionner une image depuis l'ordinateur
uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")

# Fonction de prédiction
def predict(model, image):
    img_array = np.array(image_pil)
    img_array = tf.image.resize(img_array, (224, 224))  # Adapter la taille de l'image au modèle
    img_array = np.expand_dims(img_array, axis=0)
     # Vérifier si la méthode preprocess_input est définie pour le modèle
    if hasattr(model, 'preprocess_input'):
        img_array = model.preprocess_input(img_array)
    else:
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array.astype('float32'))

    predictions = model.predict(img_array)
    return predictions

# Si une image est téléchargée
if uploaded_file is not None:
    # Afficher l'image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Image téléchargée", use_column_width=True)

    # Faire la prédiction avec VGG16
    vgg_pred = predict(vgg_model, image)
    vgg_label = tf.keras.applications.vgg16.decode_predictions(vgg_pred)[0][0][1]
    st.subheader("Prédiction VGG16:")
    st.write(f"Classe: {vgg_label}")

    # Faire la prédiction avec EfficientNet
    eff_pred = predict(eff_model, image)
    eff_label = tf.keras.applications.efficientnet.decode_predictions(eff_pred)[0][0][1]
    st.subheader("Prédiction EfficientNet:")
    st.write(f"Classe: {eff_label}")