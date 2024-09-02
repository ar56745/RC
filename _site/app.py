import streamlit as st
import pandas as pd
import os
import pickle


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Sensor Substitution")
st.write("Enter your values")

with open("D:/vsc/ml0/temp/_temp_rf.pkl","rb") as f:
    loaded_models_temp = pickle.load(f)
temp_model1 = loaded_models_temp['temp_model1']
temp_model2 = loaded_models_temp['temp_model2']
temp_model3 = loaded_models_temp['temp_model3']
def temp_model_1_predict(Humidity,Pressure): 
    try:
        Humidity = float(Humidity)
        Pressure = float(Pressure)                                 #temp
    except ValueError:
        return "Please enter valid inputs"
    pred = temp_model1.predict([[Humidity, Pressure]])
    return pred[0]  
def temp_model_2_predict(Humidity,Wind_Speed): 
    try:
        Humidity = float(Humidity)
        Wind_Speed = float(Wind_Speed)
    except ValueError:
        return "Please enter valid inputs"
    pred = temp_model2.predict([[Humidity, Wind_Speed]])
    return pred[0]  

with open("D:/vsc/ml0/humidity/_humidity_gb.pkl","rb") as f:
    loaded_models_humidity = pickle.load(f)
humidity_model1 = loaded_models_humidity['humidity_model1']
def humidity_model_1_predict(Temperature,Dewpt): 
    try:
        Temperature = float(Temperature)                                     #humid
        Dewpt = float(Dewpt)
    except ValueError:
        return "Please enter valid inputs"
    pred = humidity_model1.predict([[Temperature, Dewpt]])
    return pred[0]  

with open("D:/vsc/ml0/pressure/_pressure_rf.pkl","rb") as f:
    loaded_models_pressure= pickle.load(f)
pressure_model1 = loaded_models_pressure['pressure_model1']
pressure_model2 = loaded_models_pressure['pressure_model2']
def pressure_model_1_predict(Temperature,Humidity,Precip): 
    try:
        Temperature= float(Temperature)
        Humidity = float(Humidity)
        Precip = float(Precip)
    except ValueError:                                                         #pressure
        return "Please enter valid inputs"
    pred = pressure_model1.predict([[Temperature,Humidity, Precip]])
    return pred[0]  
def pressure_model_2_predict(Dewpt,Wind_dir_deg): 
    try:
        Dewpt = float(Dewpt)
        Wind_dir_deg = float(Wind_dir_deg)
    except ValueError:
        return "Please enter valid inputs"
    pred = pressure_model2.predict([[Dewpt,Wind_dir_deg]])
    return pred[0]  

with open("D:/vsc/ml0/precipitation/_precipitation_rf.pkl","rb") as f:
    loaded_models_precipitation = pickle.load(f)
precipitation_model1 = loaded_models_precipitation['precipitation_model1']
def precipitation_model_1_predict(Cloudcover,Dewpt): 
    try:
        Cloudcover= float(Cloudcover)
        Dewpt= float(Dewpt)
    except ValueError:        
        return "Please enter valid inputs"                                                 #precip
    pred = precipitation_model1.predict([[Cloudcover, Dewpt]])
    return pred[0]  

with open("D:/vsc/ml0/dew point/_dewpt_rf.pkl","rb") as f:
    loaded_models_dewpt = pickle.load(f)
dewpt_model1 = loaded_models_dewpt['dewpt_model1']
def dewpt_model_1_predict(Humidity,Cloudcover): 
    try:
        Cloudcover= float(Cloudcover)
        Humidity= float(Humidity)
    except ValueError:        
        return "Please enter valid inputs"                                                 #dewpt
    pred = dewpt_model1.predict([[Humidity,Cloudcover]])
    return pred[0]  

with open("D:/vsc/ml0/wind speed/_windspeed_rf.pkl","rb") as f:
    loaded_models_windspeed = pickle.load(f)
windspeed_model1 = loaded_models_windspeed['windspeed_model1']
windspeed_model2 = loaded_models_windspeed['windspeed_model2']
windspeed_model3 = loaded_models_windspeed['windspeed_model3']
windspeed_model4 = loaded_models_windspeed['windspeed_model4']
windspeed_model5 = loaded_models_windspeed['windspeed_model5']
def windspeed_model_1_predict(Temperature,Humidity,Pressure): 
    try:
        Temperature= float(Temperature)
        Humidity = float(Humidity)
        Pressure = float(Pressure)                                 #windspeed
    except ValueError:
        return "Please enter valid inputs"
    pred = windspeed_model1.predict([[Temperature,Humidity, Pressure]])
    return pred[0]  
def windspeed_model_2_predict(Temperature,Humidity): 
    try:
        Temperature= float(Temperature)
        Humidity = float(Humidity)
    except ValueError:
        return "Please enter valid inputs"
    pred = windspeed_model2.predict([[Temperature,Humidity]])
    return pred[0]  
def windspeed_model_3_predict(Temperature,Pressure): 
    try:
        Temperature= float(Temperature)
        Pressure = float(Pressure)
    except ValueError:
        return "Please enter valid inputs"
    pred = windspeed_model3.predict([[Temperature,Pressure]])
    return pred[0]  
def windspeed_model_4_predict(Humidity,Pressure): 
    try:
        Humidity= float(Humidity)
        Pressure = float(Pressure)
    except ValueError:
        return "Please enter valid inputs"
    pred = windspeed_model4.predict([[Humidity,Pressure]])
    return pred[0]  
def windspeed_model_5_predict(Heatindex,Temperature): 
    try:
        Heatindex= float(Heatindex)
        Temperature = float(Temperature)
    except ValueError:
        return "Please enter valid inputs"
    pred = windspeed_model5.predict([[Heatindex,Temperature]])
    return pred[0]  

with open("D:/vsc/ml0/heat index/_heatindex_rf.pkl","rb") as f:
    loaded_models_heatindex= pickle.load(f)
heatindex_model1 = loaded_models_heatindex['heatindex_model1']
heatindex_model2 = loaded_models_heatindex['heatindex_model2']
def heatindex_model_1_predict(Temperature,Humidity): 
    try:
        Temperature= float(Temperature)
        Humidity = float(Humidity)
    except ValueError:                                                         #heatind
        return "Please enter valid inputs"
    pred = heatindex_model1.predict([[Temperature,Humidity]])
    return pred[0]  
def heatindex_model_2_predict(Temperature,Windchill): 
    try:
        Temperature= float(Temperature)
        Windchill= float(Windchill)
    except ValueError:
        return "Please enter valid inputs"
    pred = heatindex_model2.predict([[Temperature, Windchill]])
    return pred[0]  

with open("D:/vsc/ml0/cloud cover/_cloudcover_rf.pkl","rb") as f:
    loaded_models_cloudcover= pickle.load(f)
cloudcover_model1 = loaded_models_cloudcover['cloudcover_model1']
cloudcover_model2 = loaded_models_cloudcover['cloudcover_model2']
def cloudcover_model_1_predict(Precipitation,Humidity): 
    try:
        Precipitation= float(Precipitation)
        Humidity = float(Humidity)
    except ValueError:                                                         #cloudcover
        return "Please enter valid inputs"
    pred = cloudcover_model1.predict([[Precipitation,Humidity]])
    return pred[0]  
def cloudcover_model_2_predict(Precipitation,Dewpt): 
    try:
        Precipitation= float(Precipitation)
        Dewpt= float(Dewpt)
    except ValueError:
        return "Please enter valid inputs"
    pred = cloudcover_model2.predict([[Precipitation, Dewpt]])
    return pred[0]  

with open("D:/vsc/ml0/visibilty/_visibility_gb.pkl","rb") as f:
    loaded_models_visibility= pickle.load(f)
visibility_model1 = loaded_models_visibility['visibility_model1']
visibility_model2 = loaded_models_visibility['visibility_model2']
def visibility_model_1_predict(Humidity,Wind_speed): 
    try:
        Wind_speed= float(Wind_speed)
        Humidity = float(Humidity)
    except ValueError:                                                         #visibility
        return "Please enter valid inputs"
    pred = visibility_model1.predict([[Humidity, Wind_speed]])
    return pred[0]  
def visibility_model_2_predict(Humidity,Precipitation): 
    try:
        Precipitation= float(Precipitation)
        Humidity= float(Humidity)
    except ValueError:
        return "Please enter valid inputs"
    pred = visibility_model2.predict([[Humidity,Precipitation]])
    return pred[0]  

with open("D:/vsc/ml0/wind chill/_windchill_svr.pkl","rb") as f:
    loaded_models_windchill= pickle.load(f)
windchill_model1 = loaded_models_windchill['windchill_model1']
windchill_model2 = loaded_models_windchill['windchill_model2']
def windchill_model_1_predict(Temperature,Wind_speed): 
    try:
        Wind_speed= float(Wind_speed)
        Temperature = float(Temperature)
    except ValueError:                                                         #windchill
        return "Please enter valid inputs"
    pred = windchill_model1.predict([[Temperature, Wind_speed]])
    return pred[0]  
def windchill_model_2_predict(Temperature,Dewpt): 
    try:
        Temperature= float(Temperature)
        Dewpt= float(Dewpt)
    except ValueError:
        return "Please enter valid inputs"
    pred = windchill_model2.predict([[Temperature,Dewpt]])
    return pred[0]  






nav= st.sidebar.radio("What would you like to predict?",["Home","Temperature", "Humidity", "Pressure","Precipitation","Dew Point","Wind Speed","Heat Index","Cloud Cover","Visibility","Wind Chill",])

#st.image("green-background-1480062887P1D.jpg")
if nav=="Temperature":
    Humidity=st.text_input("Humidity","in %")
    Pressure=st.text_input("Pressure","in MB")
    if(st.button("Predict")):
        prediction = temp_model_1_predict(Humidity, Pressure)
        st.success(f"Prediction: {prediction} °C")
    st.write("")                                          #temp
    st.write("")
    st.write("")   

    Humidity2=st.text_input("Humidity ","in %")
    Wind_Speed=st.text_input("Wind Speed","in kmph")
    if(st.button("Predict ")):
        prediction = temp_model_2_predict(Humidity2, Wind_Speed)
        st.success(f"Prediction: {prediction} °C")

if nav=="Humidity":
    Temperature=st.text_input("Temperature","in °C")
    Dewpt=st.text_input("Dew Point","in °C")
    if(st.button("Predict")):                                         #humid
        prediction = humidity_model_1_predict(Temperature, Dewpt)
        st.success(f"Prediction: {prediction} %")
    
if nav=="Pressure":
    Temperature=st.text_input("Temperature","in °C")
    Humidity=st.text_input("Humidity","in %")
    Precip=st.text_input("Precipitation","in mm")                  #pressure
    if(st.button("Predict")):
        prediction = pressure_model_1_predict(Temperature,Humidity, Precip)
        st.success(f"Prediction: {prediction} MB")
    st.write("")
    st.write("")
    st.write("")   
    Dewpt= st.text_input("Dew Point","in °C")
    Wind_dir_deg= st.text_input("Wind Dir. Degree","in °")
    if(st.button("Predict ")):
        prediction = pressure_model_2_predict(Dewpt,Wind_dir_deg)
        st.success(f"Prediction: {prediction} MB")

if nav=="Precipitation":
    Cloudcover=st.text_input("Cloud Cover","in %")
    Dewpt=st.text_input("Dew Point","in °C")
    if(st.button("Predict")):                                         #precip
        prediction = precipitation_model_1_predict(Cloudcover, Dewpt)
        st.success(f"Prediction: {prediction} mm")

if nav=="Dew Point":
    Humidity=st.text_input("Humidity","in %")
    Cloudcover=st.text_input("Cloud Cover","in %")
    if(st.button("Predict")):                                         #dewpt
        prediction = dewpt_model_1_predict(Humidity,Cloudcover)
        st.success(f"Prediction: {prediction} °C")

if nav=="Wind Speed":
    Temperature= st.text_input("Temperature","in °C")
    Humidity=st.text_input("Humidity","in %")
    Pressure=st.text_input("Pressure","in MB")
    if(st.button("Predict")):                                         #windspeed
        prediction = windspeed_model_1_predict(Temperature,Humidity,Pressure)
        st.success(f"Prediction: {prediction} kmph")
    st.write("")
    st.write("")
    st.write("")   
    Temperature= st.text_input("Temperature ","in °C")
    Humidity= st.text_input("Humidity ","in %")
    if(st.button("Predict ")):
        prediction = windspeed_model_2_predict(Temperature, Humidity)
        st.success(f"Prediction: {prediction} kmph")
    st.write("")
    st.write("")
    st.write("")   
    Temperature= st.text_input("Temperature  ","in °C")
    Pressure= st.text_input("Pressure  ","in MB")
    if(st.button("Predict  ")):
        prediction = windspeed_model_3_predict(Temperature, Pressure)
        st.success(f"Prediction: {prediction} kmph")
    st.write("")
    st.write("")
    st.write("")   
    Humidity= st.text_input("Humidity   ","in %")
    Pressure= st.text_input("Pressure   ","in MB")
    if(st.button("Predict   ")):
        prediction = windspeed_model_4_predict(Humidity, Pressure)
        st.success(f"Prediction: {prediction} kmph")
    st.write("")
    st.write("")
    st.write("")   
    Heatindex= st.text_input("Heat Index   ","in °C")
    Temperature= st.text_input("Temperature    ","in °C")
    if(st.button("Predict    ")):
        prediction = windspeed_model_5_predict(Heatindex, Temperature)
        st.success(f"Prediction: {prediction} kmph")

if nav=="Heat Index":
    Temperature=st.text_input("Temperature","in °C")
    Humidity=st.text_input("Humidity","in %")              #heatind
    if(st.button("Predict")):
        prediction = heatindex_model_1_predict(Temperature,Humidity)
        st.success(f"Prediction: {prediction} °C")
    st.write("")
    st.write("")
    st.write("")   
    Temperature=st.text_input("Temperature ","in °C")
    Windchill= st.text_input("Wind Chill","in °C")
    if(st.button("Predict ")):
        prediction = heatindex_model_2_predict(Temperature,Windchill)
        st.success(f"Prediction: {prediction} °C")

if nav=="Cloud Cover":
    Precipitation=st.text_input("Precipitation","in mm")
    Humidity=st.text_input("Humidity","in %")              #cloudcover
    if(st.button("Predict")):
        prediction = cloudcover_model_1_predict(Precipitation,Humidity)
        st.success(f"Prediction: {prediction} %")
    st.write("")
    st.write("")
    st.write("")   
    Precipitation=st.text_input("Precipitation ","in mm")
    Dewpt= st.text_input("Dew Point","in °C")
    if(st.button("Predict ")):
        prediction = cloudcover_model_2_predict(Precipitation, Dewpt)
        st.success(f"Prediction: {prediction} %")
    
if nav=="Visibility":
    Humidity=st.text_input("Humidity","in %")    
    Wind_Speed= st.text_input("Wind Speed","in kmph")          #visibility
    if(st.button("Predict")):
        prediction = visibility_model_1_predict(Humidity,Wind_Speed)
        st.success(f"Prediction: {prediction} km")
    st.write("")
    st.write("")
    st.write("")   
    Humidity=st.text_input("Humidity ","in %")
    Precipitation=st.text_input("Precipitation ","in inches")
    if(st.button("Predict ")):
        prediction = visibility_model_2_predict(Humidity, Precipitation)
        st.success(f"Prediction: {prediction} km")

if nav=="Wind Chill":
    Temperature=st.text_input("Temperature","in °C")    
    Wind_Speed= st.text_input("Wind Speed","in kmph")          #windchill
    if(st.button("Predict")):
        prediction = windchill_model_1_predict(Temperature,Wind_Speed)
        st.success(f"Prediction: {prediction} °C")
    st.write("")
    st.write("")
    st.write("")   
    Temperature=st.text_input("Temperature ","in °C")
    Dewpt=st.text_input("Dewpt ","in °C")
    if(st.button("Predict ")):
        prediction = windchill_model_2_predict(Temperature, Dewpt)
        st.success(f"Prediction: {prediction} °C")
    

