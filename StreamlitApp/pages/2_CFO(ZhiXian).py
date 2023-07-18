#--Team--
# Tutorial Group: 	T01 Group 4 

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

# not really efficient figure out a way to extract snowflake dataframe instead of tuple
df = pd.DataFrame(churn_to_sales, columns=["YEAR", "MONTH", "CHURN RATE", "SALES"])

df = df.sort_values(by=["YEAR", "MONTH"])

st.set_page_config(page_title="CFO-Sales Prediction",
                   page_icon="🍌")

st.markdown("# Sales Prediction")
tab1, tab2 = st.tabs(['Churn to Sales Relation', 'Prediction'])



with tab1:

    st.write("""
    ## How does Churn link to Sales?

    According to prelimnary research during the pitch phase we have established a link between Days to Next Order (target variable)
    which is what we derive our churn prediction from, there is a clear link between the average days to next order in a month with 
    that months sales.

    Through the prediction of churn rate we will imply the average days to next orders and in turn predict the potential sales of
    following month allowing for projection in sales and allow for informed steps to be taken in terms of operations and strategy.
    """)

    # insert table lining the average DTNO with the Churn rate of month with order totals and explain the percentage changes

    # showing historical data linking the churn rate to the changes in sales
    st.dataframe(
        df,
        hide_index=True
    )

with tab2:
    # page title
    st.markdown("## CFO")

    # page guide
    with st.expander("Guide to Using Page"):
        st.write("""This page provides data to be used in the PowerBI visualiser.
                 By inputing data regarding our customers based on their latest transactions we are able to make a
                 prediction of the churn of customers (churn is a twin of customer recursion).
                 The output data will contain the churn of each customer as well as some categorising information
                 to assist in visualising and showing areas of churn. This information will help to predict the effct
                 of churn on sales in different regions.""")
    
    # copied from @ryan
    # Input data
    ## File Upload section
    st.markdown("## Input Data")
    uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
    df=''

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

    with st.expander("Raw Dataframe"):
        st.write("This is the data set prior to any transformations")
        st.write(df)

    ## Removing Customer ID column
    customer_id = df.pop("CUSTOMER_ID")
    # Get categoorical columns
    demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
    beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

    df=pipeline(df)

    with st.expander("Cleaned and Transformed Data"):
        st.write("This is the data set after cleaning and transformation")
        st.write(df)

    model = load_model("assets/churn-prediction-model.jbl")
    predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
    demo_df = pd.concat([demo_df, predictions], axis=1)
    beha_df = pd.concat([beha_df, predictions], axis=1)
    data=pd.concat([customer_id, predictions], axis=1)

    # st.write(demo_df)
    # st.write(beha_df)
    # st.write(data)

    # data points to present
    ## Churn - generate churn rate
    ## Customer Location - Identify places of lower profitability and places of higher profitability (in relation to churn)
    ## Order Timing - Time of day where operations are more busy
    
    output_data = pd.concat([data, demo_df[["GENDER", "CITY", "AGE"]]], axis=1)
    
    churn_rate = predictions[['CHURNED']].sum()/predictions.count()
    
    # compared value/month
    

    # Presenting Churn Rate
    value = round(churn_rate.iloc[0] * 100, 2)
    st.metric('Churn Rate', f"{value}%")

    
