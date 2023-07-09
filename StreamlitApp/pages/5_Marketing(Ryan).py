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

st.set_page_config(page_title="Marketing", page_icon="📈")

st.markdown("# Marketing")

st.write("""
    ## Last updated customer information
    Based on the last transaction of all the customers in the United States
    """)


test_data=pd.read_csv('assets/without_transformation.csv').drop(['CHURNED'],axis=1,errors='ignore')

st.write(test_data.head())
customer_id = test_data.pop("CUSTOMER_ID")

#--File Upload--
st.markdown("## Multiple File Upload")
uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)


model = XGBClassifier()
model.load_model("assets/model.json")
