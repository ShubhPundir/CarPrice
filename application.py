import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Car Price Predictor")

# car_name
car_name = st.selectbox('Car Name',df['Car_Name'].unique())

# year
year = st.selectbox('Year',df['Year'].unique())

# Present_Price
present_Price = st.number_input('Original_Price')

# kms
kms = st.number_input('Kilometers driven')

# fuel
fuel = st.selectbox('Fuel',df['Fuel_Type'].unique())

# Selling_Type
selling_type = st.selectbox('Selling Type',df['Selling_type'].unique())

# transmission size
transmission = st.selectbox('Transmission Type',df['Transmission'].unique())

# Number of Owners
owners = int(st.number_input('Number of owners before you'))

if st.button('Predict Price'):
    # query
    
    query = np.array([car_name,year,present_Price,kms,fuel,selling_type,transmission,owners])
    query = query.reshape(1,8)
    st.title("The predicted price of this Car is " + pipe.predict(query)[0] )

