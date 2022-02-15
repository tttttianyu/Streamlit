import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from xgboost import XGBClassifier
import numpy as np

image = Image.open('SNCF_logo.png')

st.image(image, width=200)

st.write("""
# XGBoost classifier for bridge scour risk prediction
This application is trained on a French SNCF dataset to predict bridge pier's scour risk. 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    C1 = st.sidebar.selectbox('Flow type (C1)',('Fluvial','Other','Torrential'))
    C2 = st.sidebar.number_input('Slope of riverbed (%) (C2)', min_value=0.01, max_value=3.08)
    C3 = st.sidebar.number_input('Flood flow (C3)', min_value=2.09, max_value=5457.26)
    C4 = st.sidebar.number_input('WV/WC (C4)', min_value=1.52, max_value=226.68)
    C5 = st.sidebar.selectbox('Topography (C5)',('Plain','Other','Mountain'))
    C6 = st.sidebar.selectbox('Sinuosity (C6)',('Almost straight', 'Sinuous', 'Extremely sinuous'))
    C7 = st.sidebar.selectbox('Riverbed material (C7)',('Rock','Cohesive soil', 'Cohesionless soil'))
    B8 = st.sidebar.selectbox('Pier shape (B8)',('Triangular-nosed', 'Circular or oblong','Rectangular'))
    B9 = st.sidebar.selectbox('Foundation type (B9)',('Concrete/ciment','Timber piles','Caisson'))
    B10 = st.sidebar.selectbox('Existence of foundation scour countermeasures (B10)',('No','Yes'))
    H11 = st.sidebar.selectbox('Scour history (H11)',('No','Yes'))
    H12 = st.sidebar.selectbox('Flood history (H12)',('No','Yes'))
    I13 = st.sidebar.selectbox('Susceptible of scour (I13)',('No','Yes'))
    I14 = st.sidebar.selectbox('Channel rating (I14)',('Very good', 'Good', 'Fair', 'Poor','Very Bad'))
    I15 = st.sidebar.selectbox('Riverbank rating (I15)',('Very good', 'Good', 'Fair', 'Poor','Very Bad'))
    I16 = st.sidebar.selectbox('Existence of dislocation or deformation (I16)',('No','Yes'))
    I17 = st.sidebar.selectbox('Existence of local scour (I17)',('No','Yes'))
    I18 = st.sidebar.selectbox('Rating of other damages (corrosion, timber piles degradation, cracks, etc.) (I18)', ('Very good', 'Good', 'Fair', 'Poor'))
    data = {'C1': C1,
            'C2': C2,
            'C3': C3,
            'C4': C4,
            'C5': C5,
            'C6': C6,
            'C7': C7,
            'B8': B8,
            'B9': B9,
            'B10': B10,
            'H11': H11,
            'H12': H12,
            'I13': I13,
            'I14': I14,
            'I15': I15,
            'I16': I16,
            'I17': I17, 
            'I18': I18 }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

df_num = input_df.copy()

ordinal_var = {"C1": {"Fluvial": 0, "Other": 1, "Torrential": 2},
              "C5":{"Plain": 0, "Other": 1, "Mountain": 2},
              "C6":{"Almost straight": 0, "Sinuous": 1, "Very sinuous": 1,"Extremely sinuous":2},
              "C7":{"Rock": 0, "Cohesive soil": 1, "Cohesionless soil": 2},
              "B8":{"Triangular-nosed" :0, "Circular or oblong":1, "Rectangular":2},
              "B10":{"No" :0, "Yes":1},
              "H11":{"No" :0, "Yes":1},
              "H12":{"No" :0, "Yes":1},
              "I13":{"No" :0, "Yes":1},
              "I14":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3,'Very Bad':4},
              "I15":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3,'Very Bad':4},
              "I16":{"No" :0, "Yes":1},
              "I17":{"No" :0, "Yes":1},
              "I18":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3}}
df_num = input_df.replace(ordinal_var)

dforiginal = pd.read_csv('pier.csv')
df_X = dforiginal.drop(columns=['O19'])
df = pd.concat([df_num,df_X],axis=0)

# Encoding of ordinal features
encode = ['B9']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]    
df = df[:1] # Selects only the first row (the user input data)

clf = pickle.load(open('xgb_clf.pkl', 'rb'))
# Apply model to make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted risk level')
risk = np.array(['Low ','High '])
st.write(risk[prediction])

st.subheader('Prediction Probability')
st.write ('0: Low risk; 1: High risk')
st.write(prediction_proba)

st.subheader('SHAP global plot')
image = Image.open('Bar_plot.png')
st.image(image, width=800)

st.subheader('SHAP global plot')
image = Image.open('summary_plot.png')
st.image(image, width=800)

image = Image.open('xgb_shap.png')
st.image(image, width=400)