import streamlit as st
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the saved model and vectorizer
model = joblib.load('disease_prediction_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to make predictions
def predict_disease(symptom_text):
    # Transform the input text using the loaded TF-IDF vectorizer
    input_vec = vectorizer.transform([symptom_text])
    
    # Make a prediction using the loaded model
    predicted_label = model.predict(input_vec)
    
    return predicted_label[0]

# Streamlit app UI
def main():
    # Title and description
    st.title("Symptom to Disease Prediction")
    st.write("""
    This app predicts the disease based on symptom descriptions using a machine learning model.
    """)

    # Input for symptom description
    symptom_text = st.text_area("Enter your symptom description here")

    # When the user clicks the button, make a prediction
    if st.button("Predict Disease"):
        if symptom_text:
            prediction = predict_disease(symptom_text)
            st.success(f"Predicted Disease: {prediction}")
        else:
            st.error("Please enter a symptom description.")

# Run the app
if __name__ == '__main__':
    main()
