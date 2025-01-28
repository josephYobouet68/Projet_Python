import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

# Initialisation du scaler pour normaliser les données
scaler = StandardScaler()

# --- Application Streamlit ---
st.title("COVID-19 Prédiction du risque")

# Liste des colonnes utilisées pour l'entraînement du modèle
training_cols = [
    'USMER_1', 'USMER_2', 'SEX_1', 'SEX_2', 'PATIENT_TYPE_1', 'PATIENT_TYPE_2', 
    'PNEUMONIA_1', 'PNEUMONIA_2', 'PREGNANT_1', 'PREGNANT_2', 'DIABETES_1', 
    'DIABETES_2', 'COPD_1', 'COPD_2', 'ASTHMA_1', 'ASTHMA_2', 'INMSUPR_1', 
    'INMSUPR_2', 'HIPERTENSION_1', 'HIPERTENSION_2', 'OTHER_DISEASE_1', 
    'OTHER_DISEASE_2', 'CARDIOVASCULAR_1', 'CARDIOVASCULAR_2', 'OBESITY_1', 
    'OBESITY_2', 'RENAL_CHRONIC_1', 'RENAL_CHRONIC_2', 'TOBACCO_1', 'TOBACCO_2', 'AGE'
]

# --- Création d'un formulaire de saisie pour l'utilisateur ---
st.sidebar.header("Paramètres de saisie")

# Dictionnaire pour collecter les entrées utilisateur
user_input = {
    "USMER": st.sidebar.selectbox("USMER", [1, 2]),
    "SEX": st.sidebar.selectbox("SEX", [1, 2]),
    "PATIENT_TYPE": st.sidebar.selectbox("PATIENT_TYPE", [1, 2]),
    "PNEUMONIA": st.sidebar.selectbox("PNEUMONIA", [1, 2]),
    "PREGNANT": st.sidebar.selectbox("PREGNANT", [1, 2]),
    "DIABETES": st.sidebar.selectbox("DIABETES", [1, 2]),
    "COPD": st.sidebar.selectbox("COPD", [1, 2]),
    "ASTHMA": st.sidebar.selectbox("ASTHMA", [1, 2]),
    "INMSUPR": st.sidebar.selectbox("INMSUPR", [1, 2]),
    "HIPERTENSION": st.sidebar.selectbox("HIPERTENSION", [1, 2]),
    "OTHER_DISEASE": st.sidebar.selectbox("OTHER_DISEASE", [1, 2]),
    "CARDIOVASCULAR": st.sidebar.selectbox("CARDIOVASCULAR", [1, 2]),
    "OBESITY": st.sidebar.selectbox("OBESITY", [1, 2]),
    "RENAL_CHRONIC": st.sidebar.selectbox("RENAL_CHRONIC", [1, 2]),
    "TOBACCO": st.sidebar.selectbox("TOBACCO", [1, 2]),
    "AGE": st.sidebar.slider("AGE", 0, 120, 42),
}

# Conversion des données d'entrée utilisateur en DataFrame
input_df = pd.DataFrame([user_input])

# Transformation des données catégoriques en variables indicatrices (dummies)
user_data_dummies = pd.get_dummies(
    input_df.drop(columns=["AGE"]).astype("category"), drop_first=False
).astype(int)

# Ajout de la colonne AGE et normalisation
user_data_dummies['AGE'] = scaler.fit_transform(input_df[['AGE']])

# Ajout des colonnes manquantes avec des valeurs par défaut (0)
for col in training_cols:
    if col not in user_data_dummies:
        user_data_dummies[col] = 0

# Réordonner les colonnes pour correspondre à l'ordre du modèle
user_data_dummies = user_data_dummies[training_cols]

# --- Chargement du modèle et prédiction ---
with open('model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

# Prédiction du risque
prediction = lr_model.predict(user_data_dummies)

# print(prediction )

# Affichage du résultat
st.write("Risque élevé" if prediction[0] == 1 else "Faible risque")

