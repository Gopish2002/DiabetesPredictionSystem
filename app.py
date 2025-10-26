import numpy as np 
import pandas as pd 
import pickle 
import streamlit as st

# loading the model

loaded_model = pickle.load(open('C:/Users/GOPISH/Diabetes Prediction/classifer_model.pkl','rb'))

def diabetes_prediction(input_data):
   
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

    reshaped_dataset = np.asarray(input_data).reshape(1,-1)

    input_dataframe = pd.DataFrame(reshaped_dataset, columns=columns)

    prediction = loaded_model.predict(input_dataframe)

    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():
    st.title("Diabetes prediction Web Application")

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin level")
    Bmi = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")

    result = ''
    diabetes_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,DiabetesPedigreeFunction,Age]

    if st.button("Diabetes Test Result"):
        result = diabetes_prediction(diabetes_data)
        st.success(result)
   

if __name__=='__main__':

    main()










