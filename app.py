import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Orqa fon rangini qo'shish
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, lightblue, blue);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Mushuk va Kuchukni klassifikatsiya qilish")
files = st.file_uploader("Rasim yuklash", type=["jpg", "svg", "png"])

if files:
    st.image(files, caption="Yuklangan rasim", use_column_width=True)
    
    # Save the uploaded file temporarily
    img = PILImage.create(files)
    
    # Load your pre-trained model (make sure the 'model.pkl' file is accessible)
    model = load_learner('model.pkl')
    
    # Make a prediction
    pred, pred_idx, probs = model.predict(img)

    # Check if the prediction is 'cat' or 'dog'
    if str(pred).lower() in ["cat", "dog"]:
        st.write(f"Bashorat: {pred}")
        st.write(f"Ishonch darajasi: {probs[pred_idx]*100:.2f}")
    else:
        st.write("Faqat kuchuk yoki mushuk rasmini yuklashingiz mumkin.")


import streamlit as st

st.write('Hello world!')