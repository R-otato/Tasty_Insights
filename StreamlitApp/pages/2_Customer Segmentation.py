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

st.set_page_config(page_title="Customer Segmentation", page_icon="📈")

st.markdown("# Customer Segmentation")
tab1, tab2 = st.tabs(['Explore', 'Cluster'])

with tab1:
    st.markdown("## KMS")
    
with tab2:
    st.markdown("## Cluster")
    st.markdown("What is clustering? <br> Clustering is the task of dividing the population \
                or data points into a number of groups such that data") 
    
    st.markdown("## Clustering variables")