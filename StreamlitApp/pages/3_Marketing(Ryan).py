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
import plotly.express as px

def pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Removing Customer ID column
    customer_id = data.pop("CUSTOMER_ID")

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
    
    #Concat Customer ID back
    data=pd.concat([customer_id, data], axis=1)

    return data

@cached(cache={})
#Load model
def load_model(model_path: str) -> object:
    model = joblib.load(model_path)
    return model    

def main() -> None:
    # Page title
    st.markdown("# Marketing")

    # How to use this page
    with st.expander("How to Use This Page"):
        #Going to add some stuff here 
        st.write('How to Use This Page')

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
        #df=pd.read_csv('StreamlitApp/assets/without_transformation.csv')
        df=pd.read_csv('assets/without_transformation.csv')

    ## Display uploaded or defaul file
    with st.expander("Raw Dataframe"):
        st.write(df.head(10))

    # #Get categoorical columns
    # demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
    # beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

    clean_df=pipeline(df)

    with st.expander("Cleaned and Transformed Data"):
        st.write(clean_df.head(10))

    # Setup: Model loading, predictions and combining the data
    model = load_model("assets/churn-prediction-model.jbl")
    predictions= pd.DataFrame(model.predict(clean_df.drop('CUSTOMER_ID',axis=1)),columns=['CHURNED'])
    # demo_df = pd.concat([demo_df, predictions], axis=1)
    # beha_df = pd.concat([beha_df, predictions], axis=1)
    data=pd.concat([df, predictions], axis=1)

    # Display predictions
    st.markdown("## Churn Prediction Output")
    st.dataframe(data.value_counts('CHURNED'))
    st.write(data)

 

if __name__ == "__main__":
    # Setting page configuration
    st.set_page_config(
        "Tasty Bytes Marketing by Ryan Liam",
        "📊",
        #initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
