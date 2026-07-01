import json
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Clasificador de Flores", layout="centered")
st.title("🌼 Clasificador de Flores 🌷")
st.markdown("Sube una imagen para determinar si es **margarita, diente de león, rosa, girasol o tulipán**.")
st.markdown("Desarrollado por **Erik Guillen Reyes** — *IS-701 Inteligencia Artificial*")


@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("flowers_mobilenet.h5")


@st.cache_resource
def cargar_clases():
    with open("class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
CLASES_RAW = cargar_clases()

LABELS_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de león",
    "rose": "Rosa",
    "sunflower": "Girasol",
    "tulip": "Tulipán",
}
CLASES = [LABELS_ES.get(c, c) for c in CLASES_RAW]

imagen_subida = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Clasificando..."):
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediccion = modelo.predict(img_array, verbose=0)
        clase = CLASES[np.argmax(prediccion)]
        confianza = np.max(prediccion) * 100

    st.success(f"**Resultado: {clase}**")
    st.info(f"Confianza: {confianza:.2f}%")
