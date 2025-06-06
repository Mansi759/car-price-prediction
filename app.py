import pickle 
import streamlit as st 
import numpy as np 
import sklearn


with open("linear.pkl", "rb") as file:
    model = pickle.load(file)

# Load the encoders
with open("encoders.pkl", "rb") as file:
    encoding = pickle.load(file)

st.title("🚗 Car Price Prediction App")

st.subheader("🛠 Enter Car Details for Prediction")
year = st.number_input("Enter Car Year:", min_value=2000, max_value=2024, value=2020)
mileage = st.number_input("Enter Mileage (in km):", min_value=0, max_value=300000, value=289944)

if st.button("Predict Price 💰"):
    input_data = np.array([[year, mileage]])
    predicted_price = model.predict(input_data)
    st.success(f"🚘 Estimated Car Price: ${predicted_price[0]:,.2f}")
