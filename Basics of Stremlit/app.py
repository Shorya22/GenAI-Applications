import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Hello,Steamlit!")
st.write("Welcome to your first streamlit app!")

input=st.text_input("Enter Your Name:")
st.write(f"Hello, {input}!")

age= st.slider("Select Your Age",18,100,1)
st.write(f"Your age is {age}")

data = pd.read_csv(r"C:\Users\Shorya Sharma\Downloads\weather_classification_data.csv")
st.line_chart(data,y= ['Temperature','Humidity','Wind Speed'])

if st.button("OPEN"):
    st.write("I Love You Mera Baccha...Pr sbse ese bye good night krogi to pitai hogi")
    
if st.checkbox("Show Data"):
    st.write(data)
    
option = st.selectbox(
    'Which number do you like best?',
    [1, 2, 3, 4, 5,10,20,30,40,50,60]
)

st.write(f'You selected: {option}')

col1, col2 = st.columns(2)

with col1:
    st.write("This is column 1")

with col2:
    st.write("This is column 2")
    
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.write("This is Tab 1")

with tab2:
    st.write("This is Tab 2")



# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Iris Flower Classification")

sepal_length = st.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)

st.write(f"Prediction: {iris.target_names[prediction[0]]}")


