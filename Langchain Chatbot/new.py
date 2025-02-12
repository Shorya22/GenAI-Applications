import streamlit as st
from transformers import pipeline

# Load the model pipeline
pipe = pipeline("text-generation", model="gpt2", tokenizer="gpt2",truncation=True,max_new_tokens=128)

# Streamlit UI
st.title("Local Chatbot")

# Get user input
user_input = st.text_input("Ask a question:")

if user_input:
    output = pipe(user_input, max_length=100, num_return_sequences=1)
    st.write(output[0]['generated_text'])
