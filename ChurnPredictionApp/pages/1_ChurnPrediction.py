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
import joblib 
import time
from snowflake.snowpark import Session
import json
from snowflake.snowpark.functions import call_udf, col
import snowflake.snowpark.types as T
from cachetools import cached

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

#--Functions--
# Function to load the model from file and cache the result
@cached(cache={})
#Load model
def load_model(model_path: str) -> object:
    from joblib import load
    model = load(model_path)
    return model

#Get predictions
def udf_score_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
    import sys
    # file-dependencies of UDFs are available in snowflake_import_directory
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'xgboost_model.sav'
    model = load_model(import_dir+model_name)
    df.columns = feature_cols
    scored_data = pd.Series(model.predict(df))
    return scored_data

def udf_proba_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
    import sys
    # file-dependencies of UDFs are available in snowflake_import_directory
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'xgboost_model.sav'
    model = load_model(import_dir+model_name)
    df.columns = feature_cols
    scored_data = pd.Series(model.predict(df))
    proba_data = pd.Series(model.predict_proba(df)[:, 1])
    return proba_data

def transforma(data):
#   for feature, fit in joblib.load('assets/labelEncoder_fit.jbl'):
#     if feature != 'Churn':
#       data[feature] = fit.transform(data[feature])

#   for feature in data.drop(['MonthlyCharges', 'tenure'], axis=1).columns:
#     data[feature] = data[feature].astype('category')

#   for feature, scaler in joblib.load('assets/minMaxScaler_fit.jbl'):
#     data[feature] = scaler.transform(data[feature].values.reshape(-1,1))
    return


#--Introduction--
st.set_page_config(page_title="Churn Prediction", page_icon="📈")

st.markdown("# Churn Prediction")
st.sidebar.header("Churn Prediction Demo")

st.write("""
  ## How to use this tool?

  You only need to provide the parameters to the machine learning model at the sidebar on the left side of this page. And the predictions made by the model will be outputted right below.
""")

st.write("""
  ## Parameters Imputed:

  - Down below are the parameters setted up to the model by the inputs of the sidebar.
""")
test_data=pd.read_csv('assets/testdata.csv').drop(['CHURNED'],axis=1)

st.write(test_data)
type='Example'
customer_id = test_data.pop("CUSTOMER_ID")

#--File Upload--
st.markdown("## Multiple File Upload")
uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)


if uploaded_files!=[]:
    type='Your'
    for f in uploaded_files:
        st.write(f)
    data_list = []
    for f in uploaded_files:
        temp_data = pd.read_csv(f)
        data_list.append(temp_data)

    data = pd.concat(data_list)

    st.dataframe(data)

    # #-- Prediction Result --
    # st.write('## Prediction Results:')

    # prediction = get_prediction(data)
    # predictionMsg = '***Not Churn***' if float(prediction['Churn'][0][:-1]) <= 50 else '***Churn***'
    # predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

    # st.write(f'The model predicted a percentage of **{predictionPercent}** that the custumer will {predictionMsg}!')
    # st.write(prediction)
else:
    data=test_data.copy()
#--Get Prediction--

# get feature columns
feature_cols = test_data.columns
with st.spinner('Wait for it...'):
    udf_score_xgboost_model_vec_cached  = session.udf.register(func=udf_proba_xgboost_model_vec_cached, 
                                                                    name="udf_score_xgboost_model", 
                                                                    stage_location='@MODEL_STAGE',
                                                                    input_types=[T.FloatType()]*len(feature_cols),
                                                                    return_type = T.FloatType(),
                                                                    replace=True, 
                                                                    is_permanent=True, 
                                                                    imports=['@MODEL_STAGE/xgboost_model.sav'],
                                                                    packages=[f'xgboost==1.7.3'
                                                                                ,f'joblib==1.1.1'
                                                                                ,f'cachetools==4.2.2'], 
                                                                    session=session)
    data = pd.concat([customer_id, data], axis=1)
    data=session.create_dataframe(data)
    proba_data=udf_score_xgboost_model_vec_cached(*feature_cols)
    pred=data.with_column('CHURN_PROBABILITY', proba_data)
    st.markdown("# "+type+" Results")
    st.write('*Tips: Click on column name to sort!')
    st.dataframe(pred[['CUSTOMER_ID','CHURN_PROBABILITY']])  
    st.success('Done!')  


st.button("Re-run")