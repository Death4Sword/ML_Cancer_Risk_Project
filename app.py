import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

model = joblib.load("pred_cancer_logistic_regression.joblib")

ohe_encoding = OneHotEncoder(sparse_output=False, categories=[['Female', 'Male']])
ord_encoding = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])

st.title("Prédiction du Risque de Cancer de la Thyroïde")
st.subheader("Entrez les informations du patient :")

gender = st.selectbox("Genre", ["Male", "Female"])
family_history = st.selectbox("Antécédents familiaux", ["No", "Yes"])
radiation_exposure = st.selectbox("Exposition aux radiations", ["No", "Yes"])
iodine_deficiency = st.selectbox("Carence en iode", ["No", "Yes"])
smoking = st.selectbox("Fumeur", ["No", "Yes"])
obesity = st.selectbox("Obésité", ["No", "Yes"])
diabetes = st.selectbox("Diabète", ["No", "Yes"])
tsh_level = st.number_input("Niveau de TSH", value=5.0, format="%.2f")
t3_level = st.number_input("Niveau de T3", value=2.0, format="%.2f")
t4_level = st.number_input("Niveau de T4", value=8.0, format="%.2f")
nodule_size = st.number_input("Taille du nodule", value=3.0, format="%.2f")
thyroid_cancer_risk = st.selectbox("Risque estimé par le médecin", ["Low", "Medium", "High"])

input_data = pd.DataFrame({
    "Family_History": [1 if family_history == "Yes" else 0],
    "Radiation_Exposure": [1 if radiation_exposure == "Yes" else 0],
    "Iodine_Deficiency": [1 if iodine_deficiency == "Yes" else 0],
    "Smoking": [1 if smoking == "Yes" else 0],
    "Obesity": [1 if obesity == "Yes" else 0],
    "Diabetes": [1 if diabetes == "Yes" else 0],
    "TSH_Level": [tsh_level],
    "T3_Level": [t3_level],
    "T4_Level": [t4_level],
    "Nodule_Size": [nodule_size],
    "Thyroid_Cancer_Risk": ord_encoding.fit_transform([[thyroid_cancer_risk]])[0]
})

gender_encoded = ohe_encoding.fit_transform([[gender]])
gender_columns = ohe_encoding.get_feature_names_out(["Gender"])
df_gender = pd.DataFrame(gender_encoded, columns=gender_columns)

input_data = pd.concat([input_data, df_gender], axis=1)

input_data = input_data[model.feature_names_in_]

if st.button("Prédire"):
    prediction = model.predict(input_data)[0]

    prediction_label = "Malignant" if prediction == 1 else "Benign"

    if prediction_label == "Malignant":
        st.error('Le modèle prédis que le cancer est malin')
    elif prediction_label == 'Benign':
        st.success('Le modèle prédis que le cancer est bénign')
    # st.write("Données utilisées pour la prédiction :")
    # st.dataframe(input_data)