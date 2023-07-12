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
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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

def cat_plots(data):
    # Convert 'CHURNED' column to string type
    data['CHURNED'] = data['CHURNED'].astype("string")
    
    # Create a stacked bar chart for 'GENDER' against 'CHURNED'
    gender_df = data.groupby(by=["GENDER", "CHURNED"]).size().reset_index(name="counts")
    gender_chart = px.bar(data_frame=gender_df, x="GENDER", y="counts", color="CHURNED", barmode="group")

    # Create a stacked bar chart for 'MARITAL_STATUS' against 'CHURNED'
    marital_status_df = data.groupby(by=["MARITAL_STATUS", "CHURNED"]).size().reset_index(name="counts")
    marital_status_chart = px.bar(data_frame=marital_status_df, x="MARITAL_STATUS", y="counts", color="CHURNED", barmode="group")

    # Create a stacked bar chart for 'CITY' against 'CHURNED'
    city_df = data.groupby(by=["CITY", "CHURNED"]).size().reset_index(name="counts")
    city_chart = px.bar(data_frame=city_df, x="CITY", y="counts", color="CHURNED", barmode="group")

    # Create a stacked bar chart for 'CHILDREN_COUNT' against 'CHURNED'
    children_count_df = data.groupby(by=["CHILDREN_COUNT", "CHURNED"]).size().reset_index(name="counts")
    children_count_chart = px.bar(data_frame=children_count_df, x="CHILDREN_COUNT", y="counts", color="CHURNED", barmode="group")
    children_count_chart.update_layout(xaxis_type='category')
    
    # Render the charts using Streamlit in a 2 by 2 grid layout
    col1, col2 = st.columns(2)
    col1.plotly_chart(gender_chart)
    col2.plotly_chart(marital_status_chart)

    col3, col4 = st.columns(2)
    col3.plotly_chart(city_chart)
    col4.plotly_chart(children_count_chart)

def main() -> None:
    # Page title
    st.markdown("# Marketing")

    # How to use this page
    with st.expander("How to Use This Page"):
        #Going to add some stuff here 
        st.write('Hello! I see you are still browsing my stuff. What you trying do huh?')

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
        st.write(df)

    ## Removing Customer ID column
    customer_id = df.pop("CUSTOMER_ID")
    #Get categoorical columns
    cat_df=df.select_dtypes(include='object')
    df=pipeline(df)

    with st.expander("Cleaned and Transformed Data"):
        st.write(df)


    # Visualizations using the model
    ## Setup: Model loading, predictions and data
    model = load_model("assets/churn-prediction-model.jbl")
    predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
    cat_df = pd.concat([cat_df, predictions], axis=1)
    data=pd.concat([customer_id, predictions], axis=1)

    ## Visualization 1: Churn by demographics
    st.markdown("## Churn by Demographics")
    cat_plots(cat_df)

    ## Visualization 2: Churn by segmentation



    #st.dataframe(data.value_counts('CHURNED'))

if __name__ == "__main__":
    # Setting page configuration
    st.set_page_config(
        "Tasty Bytes Marketing by Ryan Liam",
        "📊",
        #initial_sidebar_state="expanded",
        layout="wide",
    )
    main()