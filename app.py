from fastcore.all import *
from fastai.vision.all import *
import streamlit as st


def label_function(x): return x.parent.name

## LOAD MODEl
learn_inf = load_learner("bagarre_model.pkl")
## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

## STREAMLIT
st.title("Détecteur de Bagarres !")

bytes_data = None
uploaded_image = st.file_uploader("Importer une image:")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Image Chargée !", use_column_width=True)   
if bytes_data:
    classify = st.button("CLASSER !")
    if classify:
        label, confidence = classify_img(bytes_data)
        st.write(f"C'est une scène de {label.replace('_', ' ')}! (Probabilité: {confidence:.04f})")

