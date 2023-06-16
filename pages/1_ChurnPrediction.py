#--Team--
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

#--Import statements--
import streamlit as st
import pandas as pd
import requests
import numpy as np
import time


st.set_page_config(page_title="Churn Prediction", page_icon="📈")

st.markdown("# Churn Prediction")
st.sidebar.header("Churn Prediction Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

uploaded_file = st.file_uploader('Multiple File Upload', accept_multiple_files=True))

st.button("Re-run")