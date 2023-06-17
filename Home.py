# --Team--
# Tutorial Group: 	T01 Group 4

# Student Name 1:	Ryan Liam Poon Yang
# Student Number: 	S10222131E
# Student Name 2:	Teh Zhi Xian
# Student Number: 	S10221851J
# Student Name 3:	Chuah Kai Yi
# Student Number: 	S10219179E
# Student Name 4:	Don Sukkram
# Student Number: 	S10223354J
# Student Name 5:	Darryl Koh
# Student Number: 	S10221893J

# --Import statements--
import streamlit as st
import pandas as pd
import requests
import numpy as np
from joblib import load
from snowflake.snowpark import Session
import json
from snowflake.snowpark.functions import call_udf, col

# --Page 1--
st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")
