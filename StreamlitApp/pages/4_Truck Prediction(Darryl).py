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

import snowflake.connector

def pipeline(data):
    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/windsorizer_gau.jbl")
    yjt = joblib.load("assets/yjt.jbl")
    ohe_enc = joblib.load("assets/ohe_enc.jbl")
    minMaxScaler = joblib.load("assets/minMaxScaler.jbl")

    # Apply the transformations to the data
    data = windsorizer_iqr.transform(data)  # Apply IQR Windsorization
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization
    data = yjt.transform(data)  # Apply Yeo-Johnson Transformation
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data.columns = data.columns.str.upper()
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    return data

@cached(cache={})
#Load model
def load_model(model_path: str) -> object:
    model = joblib.load(model_path)
    return model   

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# hide this using secrets
my_cnx = snowflake.connector.connect(
    user = "RLIAM",
    password = "Cats2004",
    account = "LGHJQKA-DJ92750",
    role = "TASTY_BI",
    warehouse = "TASTY_BI_WH",
    database = "frostbyte_tasty_bytes",
    schema = "analytics"
)

my_cur = my_cnx.cursor()
my_cur.execute("select * from churn_to_sales")
churn_to_sales = my_cur.fetchall()

st.set_page_config(page_title="Truck Prediction", page_icon="📈")

st.markdown("# Truck Prediction")
tab1, tab2 = st.tabs(['Explore', 'Cluster'])

with tab1:
    st.markdown("##")
    
# Input data
    ## File Upload section
    st.markdown("## Input Data")
    uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
    df=''
    ### If uploaded file is not empty
    if uploaded_files:
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
        #df=pd.read_csv('StreamlitApp/assets/without_transformation.csv')
        df=pd.read_csv('assets/without_transformation.csv')

    ## Display uploaded or default file
    with st.expander("Raw Dataframe"):
        st.write("This is the data set prior to any transformations")
        st.write(df)
    
    # # Display clean data
    # df=pipeline(df)

    # with st.expander("Cleaned and Transformed Data"):
    #     st.write("This is the data set after cleaning and transformation")
    #     st.write(df)
    
with tab2:
    st.markdown("## Cluster")
    st.markdown("What is clustering? <br> Clustering is the task of dividing the population \
                or data points into a number of groups such that data") 
    
    st.markdown("## Clustering variables")