import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load the encoder, scaler, and model pickle files
encoder_1 = load_pickle_file('encoder_1.pkl')
encoder_2 = load_pickle_file('encoder_2.pkl')
encoder_3 = load_pickle_file('encoder_3.pkl')
scaler = load_pickle_file('scaler.pkl')
model = load_pickle_file('model.pkl')

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # Input form for user data
    st.header("Enter Customer Details")

    CustomerID = st.text_input("Customer ID")
    Age = st.number_input("Age", min_value=0, max_value=120, value=25)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Tenure = st.number_input("Tenure (in months)", min_value=0, max_value=240, value=12)
    Usage_Frequency = st.number_input("Usage Frequency (times per month)", min_value=0, value=10)
    Support_Calls = st.number_input("Support Calls (last month)", min_value=0, value=1)
    Payment_Delay = st.number_input("Payment Delay (in days)", min_value=0, value=0)
    Subscription_Type = st.selectbox("Subscription Type", ['Basic', 'Standard', 'Premium'])
    Contract_Length = st.selectbox("Contract Length (in months)", ['Annual','Quarterly','Monthly'])
    Total_Spend = st.number_input("Total Spend (USD)", min_value=0, value=500)
    Last_Interaction = st.number_input("Last Interaction (days ago)", min_value=0, value=30)

    # Prediction button
    if st.button("Predict Churn"):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CustomerID': [CustomerID],
            'Age': [Age],
            'Gender': [Gender],
            'Tenure': [Tenure],
            'Usage Frequency': [Usage_Frequency],
            'Support Calls': [Support_Calls],
            'Payment Delay': [Payment_Delay],
            'Subscription Type': [Subscription_Type],
            'Contract Length': [Contract_Length],
            'Total Spend': [Total_Spend],
            'Last Interaction': [Last_Interaction]
        })
        # Encode categorical variables
        input_data['Gender'] = encoder_1.transform(input_data['Gender'])
        input_data['Subscription Type'] = encoder_3.transform(input_data['Subscription Type'])
        input_data['Contract Length'] = encoder_2.transform(input_data['Contract Length'])
 
        # Drop CustomerID as it's not needed for prediction
        input_data.columns = input_data.columns.str.replace(' ', '_')
        input_data = input_data.drop(columns=['CustomerID'])

        # Scale numerical features
        scaled_data = scaler.transform(input_data)

        # Make predictions
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)[0][1]

        # Display the result
        if prediction[0] == 1:
            st.error(f"Customer is likely to churn with a probability of {prediction_proba:.2f}.")
        else:
            st.success(f"Customer is not likely to churn with a probability of {1 - prediction_proba:.2f}.")

if __name__ == "__main__":
    main()