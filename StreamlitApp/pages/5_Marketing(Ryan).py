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

st.set_page_config(page_title="Marketing", page_icon="📈")

st.markdown("# Marketing")

st.write("""
    ## Welcome to our Marketing Insights Dashboard!
    """)
