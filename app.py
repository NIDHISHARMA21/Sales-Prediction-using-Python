import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the prediction function with debugging information
def predict_sales(tv, radio, newspaper):
    # Prepare inputs for scaling
    raw_inputs = np.array([[tv, radio, newspaper]])
    st.write(f"Raw inputs: {raw_inputs}")
    scaled_inputs = scaler.transform(raw_inputs)
    st.write(f"Scaled inputs: {scaled_inputs}")

    # Make a prediction
    prediction = model.predict(scaled_inputs)
    return prediction[0]

# Streamlit app layout
st.set_page_config(page_title="Sales Prediction", layout="centered")

st.title("Sales Prediction App")
st.write("Enter your advertising data below to predict sales.")

# Sidebar inputs
st.sidebar.header("Input Features")
tv = st.sidebar.number_input("TV Advertising Spend (in thousands)", min_value=0.0, max_value=300.0, step=0.1)
radio = st.sidebar.number_input("Radio Advertising Spend (in thousands)", min_value=0.0, max_value=50.0, step=0.1)
newspaper = st.sidebar.number_input("Newspaper Advertising Spend (in thousands)", min_value=0.0, max_value=50.0, step=0.1)

# Prediction button
if st.sidebar.button("Predict Sales"):
    prediction = predict_sales(tv, radio, newspaper)
    st.success(f"Predicted Sales: {prediction:.2f} units")

# Add basic HTML for styling if needed
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

st.write("This app uses a pre-trained Random Forest model to predict sales based on advertising spend.")
