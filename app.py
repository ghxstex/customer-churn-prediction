import streamlit as st
import pickle
import numpy as np

# Load model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction App")
st.subheader("Will this customer leave the service?")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has a partner?", ["No", "Yes"])
dependents = st.selectbox("Has dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 1)
phoneservice = st.selectbox("Phone Service", ["No", "Yes"])
multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing", ["No", "Yes"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
totalcharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

# Encoding user inputs similar to training
def encode_input():
    data = []

    # Binary
    data.append(1 if gender == "Male" else 0)
    data.append(1 if senior == "Yes" else 0)
    data.append(1 if partner == "Yes" else 0)
    data.append(1 if dependents == "Yes" else 0)
    data.append(tenure)
    data.append(1 if phoneservice == "Yes" else 0)

    # One-hot
    data += [
        1 if multiplelines == "Yes" else 0,
        1 if multiplelines == "No phone service" else 0,
        1 if internetservice == "Fiber optic" else 0,
        1 if internetservice == "No" else 0,
        1 if onlinesecurity == "Yes" else 0,
        1 if onlinesecurity == "No internet service" else 0,
        1 if onlinebackup == "Yes" else 0,
        1 if onlinebackup == "No internet service" else 0,
        1 if deviceprotection == "Yes" else 0,
        1 if deviceprotection == "No internet service" else 0,
        1 if techsupport == "Yes" else 0,
        1 if techsupport == "No internet service" else 0,
        1 if streamingtv == "Yes" else 0,
        1 if streamingtv == "No internet service" else 0,
        1 if streamingmovies == "Yes" else 0,
        1 if streamingmovies == "No internet service" else 0,
        1 if contract == "One year" else 0,
        1 if contract == "Two year" else 0,
        1 if paperlessbilling == "Yes" else 0,
        1 if paymentmethod == "Credit card (automatic)" else 0,
        1 if paymentmethod == "Electronic check" else 0,
        1 if paymentmethod == "Mailed check" else 0,
        monthlycharges,
        totalcharges
    ]
    return np.array(data).reshape(1, -1)

if st.button("Predict"):
    features = encode_input()
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.warning("🟡 This customer is likely to churn.")
    else:
        st.success("🟢 This customer is likely to stay.")
