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

st.set_page_config(page_title="CFO-Sales Prediction")

st.markdown("# Sales Prediction")
tab1, tab2 = st.tabs(['Churn to Sales Relation', 'Prediction'])

# link_data = 

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
    # st.table(link_data)