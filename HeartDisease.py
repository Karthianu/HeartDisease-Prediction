# -*- coding: utf-8 -*-



import numpy as np
import pickle
import streamlit as st



loaded_model=pickle.load(open('C:/Users/Vignesan/Documents/ML Projects/HeartDisease Prediction/trained_model.sav','rb'))

#def app():

def heart_prediction(ip_data):
    
    ip_arr=np.array(ip_data)
    res_arr=ip_arr.reshape(1,-1)

    pred=loaded_model.predict(res_arr)
    print(pred)
    if (pred=='Presence'):
        return("The User Have Heart Disease")
    else:
        return("The User Have Healthy Heart")
    
    
def main():
    #giving title
    st.title('HEART DISEASE PREDICTION')
    
    #Getting the input data from user
    #Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium,Heart Disease    
    
    Age=st.text_input("Age")
    Gender=st.text_input("Gender")
    ChestPainType=st.text_input("ChestPainType")
    BloodPressure=st.text_input("BloodPressure")
    Cholesterol=st.text_input("Cholesterol")
    FBSover120=st.text_input("FBSover120")
    EKG_results=st.text_input("EKG_results")
    Max_HR=st.text_input("Max_HR")
    Exercise_angina=st.text_input("Exercise_angina")
    ST_depression=st.text_input("ST_depression")
    Slope_of_ST=st.text_input("Slope_of_ST")
    Number_of_vessels_fluro=st.text_input("Number_of_vessels_fluro")
    Thallium=st.text_input("Thallium")
   
    #Code for Prediction
    
    diagnosis=''
    
    
    #creating a button for prediction
    
    if st.button('HeartDisease Prediction'):
        diagnosis=heart_prediction([Age,Gender,ChestPainType,BloodPressure,Cholesterol,FBSover120,EKG_results,Max_HR,Exercise_angina,ST_depression,Slope_of_ST,Number_of_vessels_fluro,Thallium])
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()

    
    
