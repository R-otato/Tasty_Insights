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

def demo_plots(data):
    # Convert 'CHURNED' column to string type
    data['CHURNED'] = data['CHURNED'].astype("string")

    # Create a grouped  bar chart for 'GENDER' against 'CHURNED'
    gender_df = data.groupby(by=["GENDER", "CHURNED"]).size().reset_index(name="counts")
    gender_chart = px.bar(data_frame=gender_df, x="GENDER", y="counts", color="CHURNED", barmode="group",color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})

    # Create a grouped  bar chart for 'MARITAL_STATUS' against 'CHURNED'
    marital_status_df = data.groupby(by=["MARITAL_STATUS", "CHURNED"]).size().reset_index(name="counts")
    marital_status_chart = px.bar(data_frame=marital_status_df, x="MARITAL_STATUS", y="counts", color="CHURNED", barmode="group",color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})

    # Create a grouped  bar chart for 'CITY' against 'CHURNED'
    city_df = data.groupby(by=["CITY", "CHURNED"]).size().reset_index(name="counts")
    city_chart = px.bar(data_frame=city_df, x="CITY", y="counts", color="CHURNED", barmode="group",color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})

    # Create a grouped bar chart for 'CHILDREN_COUNT' against 'CHURNED'
    children_count_df = data.groupby(by=["CHILDREN_COUNT", "CHURNED"]).size().reset_index(name="counts")
    children_count_chart = px.bar(data_frame=children_count_df, x="CHILDREN_COUNT", y="counts", color="CHURNED", barmode="group",color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    children_count_chart.update_layout(xaxis_type='category')

    # Create a box plot for 'Age' against 'CHURNED'
    age_chart=px.box(data_frame=data, x="CHURNED", y="AGE", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    
    # Render the charts using Streamlit in a 2 by 2 grid layout
    col1, col2  = st.columns(2)
    col1.plotly_chart(gender_chart)
    col2.plotly_chart(marital_status_chart)

    col3, col4 = st.columns(2)
    col3.plotly_chart(city_chart)
    col4.plotly_chart(children_count_chart)
    st.plotly_chart(age_chart)

def num_plots(data):
    # Create a box plot for 'RECENCY' against 'CHURNED'
    recency_chart=px.box(data_frame=data, x="CHURNED", y="RECENCY", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    # Create a box plot for 'FREQUENCY' against 'CHURNED'
    frequency_chart=px.box(data_frame=data, x="CHURNED", y="FREQUENCY", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    # Create a box plot for 'MONETARY' against 'CHURNED'
    monetary_chart=px.box(data_frame=data, x="CHURNED", y="MONETARY", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    # Create a box plot for 'LENGTH_OF_RELATIONSHIP' against 'CHURNED'
    lor_chart=px.box(data_frame=data, x="CHURNED", y="LENGTH_OF_RELATIONSHIP", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})
    # Create a box plot for 'FNUM_OF_LOCATIONS_VISITED' against 'CHURNED'
    nov_chart=px.box(data_frame=data, x="CHURNED", y="NUM_OF_LOCATIONS_VISITED", color='CHURNED',color_discrete_map={
                "1": "crimson",
                "0": "cornflowerblue"})

    # Render the charts using Streamlit in a 2 by 2 grid layout
    col1, col2  = st.columns(2)
    col1.plotly_chart(recency_chart)
    col2.plotly_chart(frequency_chart)

    col3, col4 = st.columns(2)
    col3.plotly_chart(monetary_chart)
    col4.plotly_chart(lor_chart)
    
    st.plotly_chart(nov_chart)

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
        st.write(df)

    ## Removing Customer ID column
    customer_id = df.pop("CUSTOMER_ID")
    #Get categoorical columns
    demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
    beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

    df=pipeline(df)

    with st.expander("Cleaned and Transformed Data"):
        st.write(df)


    # Visualizations using the model
    ## Setup: Model loading, predictions and data
    model = load_model("assets/churn-prediction-model.jbl")
    predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
    demo_df = pd.concat([demo_df, predictions], axis=1)
    beha_df = pd.concat([beha_df, predictions], axis=1)
    data=pd.concat([customer_id, predictions], axis=1)

    ## Visualization 1: Churn by demographics
    st.markdown("## Churn by Member Demographics")
    demo_plots(demo_df)

    ## Visualization 2: Churn by behavior
    st.markdown("## Churn by Member Behavior")
    num_plots(beha_df)

    ## Visualization 3: Churn by segmentation
    st.markdown("## Churn by Member Segmentation")


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