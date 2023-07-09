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

# Default customer information
st.write("""
    ## Default Data
    The default data is the last updated information of the members in United States.
    """)

# Loading test data
test_data = pd.read_csv('assets/without_transformation.csv').drop(['CHURNED'], axis=1, errors='ignore')
st.write(test_data.head())

# Populating customer ID column
customer_id = test_data.pop("CUSTOMER_ID")

# File Upload section
st.markdown("## Add your own data by uploading files")
uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)

# Model loading
model = XGBClassifier()
model.load_model("assets/model.json")
