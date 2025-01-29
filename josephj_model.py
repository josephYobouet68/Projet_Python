import streamlit as st
from sklearn.linear_model import LogisticRegression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle


scaler = StandardScaler()

# --- Application Streamlit ---
st.title("COVID-19 Prédiction du risque")

training_cols = ['USMER_1', 'USMER_2', 'SEX_1', 'SEX_2', 'PATIENT_TYPE_1',
       'PATIENT_TYPE_2', 'PNEUMONIA_1', 'PNEUMONIA_2', 'PREGNANT_1',
       'PREGNANT_2', 'DIABETES_1', 'DIABETES_2', 'COPD_1', 'COPD_2',
       'ASTHMA_1', 'ASTHMA_2', 'INMSUPR_1', 'INMSUPR_2', 'HIPERTENSION_1',
       'HIPERTENSION_2', 'OTHER_DISEASE_1', 'OTHER_DISEASE_2',
       'CARDIOVASCULAR_1', 'CARDIOVASCULAR_2', 'OBESITY_1', 'OBESITY_2',
       'RENAL_CHRONIC_1', 'RENAL_CHRONIC_2', 'TOBACCO_1', 'TOBACCO_2', 'AGE']

# Créer un formulaire de saisie pour l'utilisateur
st.sidebar.header("Paramètres de saisie")
user_input = {
    "USMER": st.sidebar.selectbox("USMER", ['Oui', 'Non']),
    "MEDICAL_UNIT": st.sidebar.selectbox("MEDICAL_UNIT", ['Oui', 'Non']),
    "SEX": st.sidebar.selectbox("SEX", ['Oui', 'Non']),
    "PATIENT_TYPE": st.sidebar.selectbox("PATIENT_TYPE", ['Oui', 'Non']),
    "PNEUMONIA": st.sidebar.selectbox("PNEUMONIA", ['Oui', 'Non']),
    "PREGNANT": st.sidebar.selectbox("PREGNANT", ['Oui', 'Non']),
    "DIABETES": st.sidebar.selectbox("DIABETES", ['Oui', 'Non']),
    "COPD": st.sidebar.selectbox("COPD", ['Oui', 'Non']),
    "ASTHMA": st.sidebar.selectbox("ASTHMA", ['Oui', 'Non']),
    "INMSUPR": st.sidebar.selectbox("INMSUPR", ['Oui', 'Non']),
    "HIPERTENSION": st.sidebar.selectbox("HIPERTENSION", ['Oui', 'Non']),
    "OTHER_DISEASE": st.sidebar.selectbox("OTHER_DISEASE", ['Oui', 'Non']),
    "CARDIOVASCULAR": st.sidebar.selectbox("CARDIOVASCULAR", ['Oui', 'Non']),
    "OBESITY": st.sidebar.selectbox("OBESITY", ['Oui', 'Non']),
    "RENAL_CHRONIC": st.sidebar.selectbox("RENAL_CHRONIC", ['Oui', 'Non']),
    "TOBACCO": st.sidebar.selectbox("TOBACCO", ['Oui', 'Non']),
    "AGE": st.sidebar.slider("AGE", 0, 120, 42),
}


input_df = pd.DataFrame([user_input])
input_df = input_df.drop(columns=["MEDICAL_UNIT"])
user_data = input_df.drop(columns=['AGE'])

user_data['AGE'] = input_df['AGE']

user_data = scaler.fit_transform(user_data)

with open('lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

prediction = lr_model.predict(user_data)

# print(prediction)

# Affichage du résultat
st.write("Risque élevé" if prediction[0] == 1 else "Faible risque")

