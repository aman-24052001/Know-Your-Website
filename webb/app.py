import streamlit as st
import pandas as pd
from web_project import load_data, preprocess_data, load_model, predict_threat

def main():
    # Load the dataset
    data = load_data()

    # Preprocess the data
    data_processed = preprocess_data(data)

    # Split data into features (X) and target (y)
    X = data_processed.drop('Threat', axis=1)
    y = data_processed['Threat']

    # Load the model with training data
    model = load_model(X, y)

    # Sidebar with input fields
    st.sidebar.title("Input Parameters")
    request_method = st.sidebar.selectbox("Request Method", X['Request Method'].unique())
    request_path = st.sidebar.text_input("Request Path", "")
    request_parameters = st.sidebar.text_input("Request Parameters", "")
    user_agent = st.sidebar.text_input("User-Agent", "")
    referrer = st.sidebar.text_input("Referrer", "")
    ip_address = st.sidebar.text_input("IP Address", "")
    content_type = st.sidebar.text_input("Content-Type", "")
    response_code = st.sidebar.text_input("Response Code", "")

    # Button to trigger prediction
    if st.sidebar.button("Predict"):
        input_data = {
            'Request Method': request_method,
            'Request Path': request_path,
            'Request Parameters': request_parameters,
            'User-Agent': user_agent,
            'Referrer': referrer,
            'IP Address': ip_address,
            'Content-Type': content_type,
            'Response Code': response_code
        }

        # Make prediction
        threat_status = predict_threat(model, input_data)

        # Display prediction result with increased font size and colored text
        if threat_status == 0:
            st.markdown("<p style='font-size:20px;color:green;'>Predicted Threat: NO THREAT</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:20px;color:red;'>Predicted Threat: POSSIBLE ATTACK THREAT</p>", unsafe_allow_html=True)

    # Display model accuracy matrix
    accuracies = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "Gradient Boosting", "Neural Network"],
        "Accuracy": [0.7435897435897436, 0.8461538461538461, 0.7948717948717948, 0.8205128205128205, 0.717948717948718, 0.7692307692307693]
    }

    st.write("Model Accuracy Matrix:")
    df_accuracy = pd.DataFrame(accuracies)
    st.table(df_accuracy)

if __name__ == "__main__":
    main()
