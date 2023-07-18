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

st.set_page_config(page_title="CFO-Sales Prediction")

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