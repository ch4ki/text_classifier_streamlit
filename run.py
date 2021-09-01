import streamlit as st
from predict import predict_text


st.title('Text Classification with Machine Learning & NLP')
user_input_text = st.text_input('Text ')
if st.button('Predict the topic'):
    topic_name = predict_text(user_input_text, "naive_bayes_classifier.pkl")
    st.write(f"Topic category is: {topic_name}")