import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and data
pipe = joblib.load('pipe.pkl')       # Trained pipeline
df = joblib.load('df.pkl')           # DataFrame with dropdown options

st.title("Laptop Price Predictor")

# User inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen',['No','Yes'])
ips = st.selectbox('IPS',['No','Yes'])
screen_size = st.slider('Screen Size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert Yes/No to 0/1
    touchscreen = 1 if touchscreen=='Yes' else 0
    ips = 1 if ips=='Yes' else 0

    # Calculate PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Prepare input DataFrame
    query = pd.DataFrame([[company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os]],
                         columns=['Company','TypeName','Ram','Weight','Touchscreen','Ips','Ppi','Cpu brand','HDD','SSD','Gpu brand','os'])

    # Predict price
    price = pipe.predict(query)[0]
    st.title("The predicted price of this configuration is " + str(int(np.exp(price))))
