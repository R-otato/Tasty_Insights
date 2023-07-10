#--Import statements--
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import requests
import numpy as np
import joblib 
import time
# from snowflake.snowpark import Session
import json
# from snowflake.snowpark.functions import call_udf, col
# import snowflake.snowpark.types as T
from cachetools import cached

# Setting page configuration
st.set_page_config(page_title="Marketing", page_icon="📈")

# Page title
st.markdown("# Marketing")

# How to use this page
st.write('Hello! I see you are browsing my shit. What you trying do huh?')

with st.expander("How to Use This Page"):
    #Going to add some stuff here 
    st.write('Hello! I see you are still browsing my shit. What you trying do huh?')

# Input data
## File Upload section
st.markdown("## Input Data")
uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
df=''
### If uploaded file is not empty
if uploaded_files!=[]:
    data_list = []
    #Append all uploaded files into the list
    for f in uploaded_files:
        st.write(f)
        temp_data = pd.read_csv(f)
        data_list.append(temp_data)
    st.success("Uploaded your file!")
    #concat the files together if there are more than one file uploaded
    df = pd.concat(data_list)
else:
    st.info("Using the last updated data of the members in United States. Upload a file above to use your own data!")
    df=pd.read_csv('StreamlitApp/assets/without_transformation.csv')

## Display uploaded or defaul file
with st.expander("Raw Dataframe"):
    st.write(df)
#df = clean_data(df)
with st.expander("Cleaned and Transformed Data"):
    df=pd.read_csv('StreamlitApp/assets/with_transformation.csv')
    st.write(df)

## Removing Customer ID column
customer_id = df.pop("CUSTOMER_ID")

# Visualizations using the model
## Model loading
model = XGBClassifier()
model.load_model("assets/improvedmodel.json")