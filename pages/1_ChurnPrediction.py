#--Team--
# Tutorial Group: 	T01 Group 4 

# Student Name 1:	Ryan Liam Poon Yang
# Student Number: 	S10222131E 
# Student Name 2:	Teh Zhi Xian
# Student Number: 	S10221851J
# Student Name 3:	Chuah Kai Yi
# Student Number: 	S10219179E
# Student Name 4:	Don Sukkram
# Student Number: 	S10223354J
# Student Name 5:	Darryl Koh
# Student Number: 	S10221893J

#--Import statements--
import streamlit as st
import pandas as pd
import requests
import numpy as np
#import joblib - Some error importing
from snowflake.snowpark import Session
import json
from snowflake.snowpark.functions import call_udf, col

# Get account credentials from a json file
with open("data_scientist_auth.json") as f:
    data = json.load(f)
    username = data["username"]
    password = data["password"]
    account = data["account"]

# Specify connection parameters
connection_parameters = {
    "account": account,
    "user": username,
    "password": password,
    "role": "TASTY_BI",
    "warehouse": "TASTY_BI_WH",
    "database": "frostbyte_tasty_bytes",
    "schema": "analytics",
}

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

#--Introduction--
st.set_page_config(page_title="Churn Prediction", page_icon="📈")

st.markdown("# Churn Prediction")
st.sidebar.header("Churn Prediction Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

#--File Upload--
st.markdown("## Multiple File Upload")
uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)
for f in uploaded_files:
    st.write(f)
data_list = []
for f in uploaded_files:
    temp_data = pd.read_csv(f)
    data_list.append(temp_data)

data = pd.concat(data_list)


#--Get Prediction--
def get_prediction(data):
#   for feature, fit in joblib.load('assets/labelEncoder_fit.jbl'):
#     if feature != 'Churn':
#       data[feature] = fit.transform(data[feature])

#   for feature in data.drop(['MonthlyCharges', 'tenure'], axis=1).columns:
#     data[feature] = data[feature].astype('category')

#   for feature, scaler in joblib.load('assets/minMaxScaler_fit.jbl'):
#     data[feature] = scaler.transform(data[feature].values.reshape(-1,1))

  model = joblib.load('./assets/churn-prediction-model.jbl')
  import sys
  #file-dependencies of UDFs are available in snowflake_import_directory
  IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
  import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
  model_name = 'xgboost_model.sav'

  return pd.DataFrame({
    'Not Churn': f'{model.predict_proba(data)[0][0] * 100 :.1f}%',
    'Churn': f'{model.predict_proba(data)[0][1] * 100 :.1f}%'}, index=['Predictions'])

# #-- Prediction Result --
# st.write('## Prediction Results:')

# prediction = get_prediction(data)
# predictionMsg = '***Not Churn***' if float(prediction['Churn'][0][:-1]) <= 50 else '***Churn***'
# predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

# st.write(f'The model predicted a percentage of **{predictionPercent}** that the custumer will {predictionMsg}!')
# st.write(prediction)

st.button("Re-run")