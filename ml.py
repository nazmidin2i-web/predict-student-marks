import streamlit as st
import warnings 
import numpy as np
import pickle
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Marks Predictor",page_icon="",layout="centered")

st.title("Student Marks Predictor")
st.write("Enter The Numbers of hourse Studied (1-10) and **Click** Predict To See The Predicted Marks")

hours = st.number_input("Hours Studied", 
                        min_value=1.0,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.1f"
                        )

#Load Model

def load_model(model):
    with open(model,"rb") as f:
            slr =pickle.load(f)
    return slr

try:
    model = load_model("lin_reg.pkl")
except Exception as e:
        st.error('Your pickle file not found...')
        st.exception("Failed to Load The Model :",e)
        st.stop() 

if st.button("Predict"):
    try:
        x=np.array([[hours]])
        predictions = model.predict(x)
        predictions = predictions[0]

        st.success(f'Predicted Marks : {predictions :.1f}')
        st.write("Note : This Is ML Prediction **Result May Vary**")

    except Exception as e:
        st.error(f'Prediction Failed : {e}') 