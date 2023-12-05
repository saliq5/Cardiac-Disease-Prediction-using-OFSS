#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay

pipe = pickle.load(open('pipeline_voting_temp.pkl','rb'))

st.set_page_config(page_title='NITK-AI ML APP')

###############################################################

import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('img2t.jpg')

###################################################################


def predict(Age,Sex,Chest_Pain,Rest_BP,Fasting_Sugar,Exercise_Angina,Major_Vessels, Defect_Type):
    input=np.array([Age,Sex,Chest_Pain,Rest_BP,Fasting_Sugar,Exercise_Angina,Major_Vessels,Defect_Type] ,dtype=object).reshape(1,8)
    prediction = pipe.predict(input)[0]
    return int(prediction)

def main():
    st.title("")
    html_temp = """
    <div style="background:#025246 ;padding:10px; font-size:1">
    <h2 style="color:white;text-align:center;"> ML APP FOR CARDIAC DISEASE PREDICTION </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    age_max=77
    age_min=29
    restbp_min=94
    restbp_max=200
    
    Age = st.number_input("Age",min_value=0,max_value=120,value=None,step=1,placeholder="Enter Value")
    Sex = st.number_input("Gender: Enter 1 for Male and 0 for Female",min_value=0,max_value=1,value=None,step=1,placeholder="Enter Value")
    Chest_Pain = st.number_input("Chest Pain Severity: 0/1/2/3",min_value=0,max_value=3,value=None,step=1,placeholder="Enter Value")
    Rest_BP = st.number_input("Resting Blood Pressure: Systolic",min_value=0,max_value=300,step=1,value=None,placeholder="Enter Value")
    Fasting_Sugar = st.number_input("Fasting Sugar: If greater than 120mg/dl, Enter 1 else 0",min_value=0,max_value=1,step=1,placeholder="Enter Value",value=None)
    Exercise_Angina = st.number_input("Exercise Induced Angina: Enter 1 for Yes else 0",min_value=0,max_value=1,step=1,placeholder="Enter Value",value=None)
    Major_Vessels = st.number_input("Major Vessels: Colored by Fluoroscopy- 0/1/2/3/4",min_value=0,max_value=4,step=1,placeholder="Enter Value",value=None)
    Defect_Type = st.number_input("Defect Type: Normal- 0/ Fixed- 1/ Reversable- 2 / Irreversable - 3",min_value=0,max_value=3,step=1,placeholder="Enter Value",value=None)
    
    
    
    
    safe_html ="""  
        <div style="background:#025246 ;padding:5px" >
        <h2 style="color:white;text-align:center;"> CARDIAC DISEASE NOT DETECTED! </h2>
        </div>
        """
    unsafe_html ="""  
        <div style="background:#025246 ; padding:5px">
        <h2 style="color:white;text-align:center;"> CARDIAC DISEASE DETECTED!</h2>
        </div>
        """
    
     
    if st.button("Predict"):
        if Age is None:
            st.warning("Please enter Age!")
            return
        if Sex is None:
            st.warning("Please enter Gender!")
            return
        if Chest_Pain is None:
            st.warning("Please enter type of Chest Pain!")
            return 
        if Rest_BP is None:
            st.warning("Please enter Rest BP!")
            return 
        
        if Fasting_Sugar is None:
            Fasting_Sugar = np.nan
        if Exercise_Angina is None:
            Exercise_Angina = np.nan
        if Major_Vessels is None:
            Major_Vessels = np.nan
        if Defect_Type is None:
            Defect_Type = np.nan
        if Rest_BP is not None:
            if Rest_BP > restbp_max or Rest_BP < restbp_min:
                Rest_BP=np.nan
        if Age is not None:
            if Age > age_max or Age < age_min:
                Age=np.nan
            
            
        output = predict(Age,Sex,Chest_Pain,Rest_BP,Fasting_Sugar,Exercise_Angina,Major_Vessels,Defect_Type)    
        
        if output == 1:            
            st.markdown(unsafe_html,unsafe_allow_html=True)
        elif output == 0:            
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()

