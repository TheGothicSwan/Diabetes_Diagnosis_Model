import numpy as np
import streamlit as st
import pickle
import sklearn

#Load saved model
loaded_model = pickle.load(open('diabetes_model.sav', "rb"))

#Creating prediction function
def diabetes_prediction(input_data):
    #Convert input data into a numpy array
    input_data_as_numpy=np.asarray(input_data)

    #Reshape input data to predict for one instance only
    input_data_reshape= input_data_as_numpy.reshape(1,-1)

    #Making Prediction
    prediction= loaded_model.predict(input_data_reshape)
    print(prediction)

    if prediction[0]== 0:
        print("This patient is not Diabetic")
    else:
        print("This patient is Diabetic")

#Defining a system for the Web app
#Creating the app interface
def main():

    #Interface title
    st.title('Krystal Diabetes Prediction Web App')

    #Creating Inputter for getting user input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of Patient')

    #Creating code for Diagnosis
    diagnosis = ''

    #Create prediction button
    if st.button('Diabetes Result'):
        # Get the prediction result (0 or 1) from the model
        prediction = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])
        

    # Display the diagnosis message based on the prediction result
        if prediction == 1:
            diagnosis = 'This patient is diabetic'
        else:
            diagnosis = 'This patient is not diabetic'

    st.success(diagnosis)

    
if __name__ == '__main__':
    main()

