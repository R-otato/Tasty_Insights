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

st.sidebar.write('# **Parameters Input**')

def input_features():
  def get_sidebar_radio(title, choices=('No', 'Yes'), horizontal=True):
    return st.sidebar.radio(f'**{title}:**', choices, horizontal=horizontal)

  gender = get_sidebar_radio('Gender', ('Male', 'Female'))
  seniorCitizen = get_sidebar_radio('Senior')
  partner = get_sidebar_radio('Partner')
  dependents = get_sidebar_radio('Dependents')
  phoneService = get_sidebar_radio('Phone Service')
  multipleLines = get_sidebar_radio('Multiple Lines')
  internetService = get_sidebar_radio('Internet Service', ('No', 'Fiber optic', 'DSL'), horizontal=False)
  onlineSecurity = get_sidebar_radio('Online Security')
  onlineBackup = get_sidebar_radio('Online Backup')
  deviceProtection = get_sidebar_radio('Device Protection')
  techSupport = get_sidebar_radio('Tech Support')
  streamingTV = get_sidebar_radio('TV Streaming')
  streamingMovies = get_sidebar_radio('Movie Streaming')
  contract = get_sidebar_radio('Contract', ('Month-to-month', 'One year', 'Two year'), horizontal=False)
  paperlessBilling = get_sidebar_radio('Paperless Billing')
  paymentMethod = get_sidebar_radio('Payment Method', ('Credit card (automatic)', 'Bank transfer (automatic)', 'Electronic check', 'Mailed check'), horizontal=False)
  tenure = st.sidebar.slider('**Tenure (Months):**', min_value=0, max_value=70, value=12)
  monthlyCharges = st.sidebar.number_input('**Monthly Charges ($):**', min_value=10.0, max_value=120.0, value=65.0)

data = input_features()

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")