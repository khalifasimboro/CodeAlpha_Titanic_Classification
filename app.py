import streamlit as st
import numpy as np
import joblib

st.title('Prediction of Survival Status')
st.image('https://media.wired.com/photos/5f9a0d462cc0d6153d3f963c/master/pass/backchannel-how-to-escape-the-titanic.jpg', width=550, output_format="jpg")

#Chargement du modèle
model = joblib.load(filename='best_model.pkl')

#definition d'une fonction d'interference
def infer(cabine,gender,fare_paid,port_of_embarkation):
    tb = np.array([cabine,gender,fare_paid,port_of_embarkation])
    tb = tb.reshape(1,-1)
    y = model.predict(tb)
    if y == 0:
        y = 'This passenger did not survive the Titanic sinking  🔴😢.'
        return (y)

    else:
        y = 'This passenger survived the Titanic sinking  🟢😊.'
        return (y)

st.sidebar.header("Passager's informations")
#Caractéristiques à saisir pour l'utilisateur
cabine = st.sidebar.number_input('class:',value=0)
gender = st.sidebar.number_input('gender:',value=0)
fare_paid = st.sidebar.number_input('fare paid:',value=0)
port_of_embarkation = st.sidebar.number_input('port of embarkation:',value=0)


st.sidebar.subheader('application developed by Rydouane:')
st.sidebar.subheader('Email: simbororydouane2001@gmail.com github: https://github.com/khalifasimboro')

#Création du boutton predict
if st.button('predict'):
    prediction = infer(cabine,gender,fare_paid,port_of_embarkation)
    st.success(prediction)
