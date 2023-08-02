#--Team--
# Tutorial Group: 	T01 Group 4 

#--Import statements--
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

import plotly.express as px
import joblib 

def init_model():
    
    # churn prediction
    yeojohnsontransformer = joblib.load(path + '/models/yeojohnsontransformer.jbl')
    onehotencoder = joblib.load(path + '/models/onehotencoder.jbl')
    minmaxscaler = joblib.load(path + 'models/minmaxscaler.jbl')
    model = joblib.load(path + 'models/xgb_churn_model.jbl')
    
    # customer demographic
    onehotencoder_cust_demo = joblib.load(path + 'models/cust_demographic_ohe.jbl')
    model_cust_demo = joblib.load(path + 'models/cust_demographic_model.jbl')
    
    return yeojohnsontransformer, onehotencoder, minmaxscaler, model, \
            onehotencoder_cust_demo, model_cust_demo

def init_dataset():
    
    # display dataset
    customer_data = pd.read_csv(path + 'datasets/relavent_original_dataset.csv')
    
    # for churn prediction proxy values
    df_before_scaling_sample = pd.read_csv(path + 'datasets/before_scaling_dataset.csv')
    
    # display cluster information
    cluster_information = pd.read_csv(path + 'datasets/cluster_information.csv')
    
    return customer_data, df_before_scaling_sample, cluster_information

def cust_name_search(customer_data):
    # to search for a specific customer by first name, last name, or combination
    search_term = st.text_input("Search for a customer by First Name, Last Name, or combination:")

    if search_term:
        # Filter the customer data based on the search term using partial matching 
        selected_customer = customer_data[customer_data['FIRST_NAME'].str.contains(search_term, case=False, na=False) |
                                        customer_data['LAST_NAME'].str.contains(search_term, case=False, na=False) |
                                        customer_data.apply(lambda row: search_term in f"{row['FIRST_NAME']} {row['LAST_NAME']}".lower(), axis=1)]

        
        if not selected_customer.empty:
            st.subheader("Selected Customer Details")
            st.dataframe(selected_customer)
        else:
            st.write("No matching customer found.")
            
def validate_int_input(input):
    
    try:
        return int(input)
    
    except ValueError:
        return None
    
def cust_id_search(customer_data):
    # to search for a specific customer by first name, last name, or combination
    search_term = st.text_input("Search for a customer by their ID:")

    search_term = validate_int_input(search_term)
    
    if search_term is None:
        
        st.write("Please enter a valid customer ID.")
    
    else:
        
        # Filter the customer data based on the search term using partial matching 
        selected_customer = customer_data[customer_data['CUSTOMER_ID']==search_term]

        if not selected_customer.empty:
            st.subheader("Selected Customer Details")
            st.dataframe(selected_customer)
            return selected_customer
        else:
            st.write("No matching customer found.")
            
def user_input_features(selected_customer=None):
        
    if selected_customer is not None:
        default_CITY = selected_customer['CITY']
        default_GENDER = selected_customer['GENDER']
        default_MARITAL_STATUS = selected_customer['MARITAL_STATUS']
        default_CHILDREN_COUNT = selected_customer['CHILDREN_COUNT']
        default_SIGN_UP_DATE = selected_customer['SIGN_UP_DATE']
        default_BIRTHDAY_DATE = selected_customer['BIRTHDAY_DATE']
        default_ = selected_customer['GENDER']
    else:
        default_CITY = selected_customer['CITY']
        default_GENDER = selected_customer['GENDER']
        default_MARITAL_STATUS = selected_customer['MARITAL_STATUS']
        default_CHILDREN_COUNT = selected_customer['CHILDREN_COUNT']
        default_SIGN_UP_DATE = selected_customer['SIGN_UP_DATE']
        default_BIRTHDAY_DATE = selected_customer['BIRTHDAY_DATE']
        default_ = selected_customer['GENDER']

    
    # User Input for Dropdowns
    city = st.selectbox("City", np.sort(customer_data['CITY'].unique()), index = 1,)
    gender = st.selectbox("Gender", np.sort(customer_data['GENDER'].unique()))
    marital_status = st.selectbox("Marital Status", np.sort(customer_data['MARITAL_STATUS'].unique()), index = 1)
    children_count = st.selectbox("Number of Children", np.sort(customer_data['CHILDREN_COUNT'].unique()), index = 2)


    sign_up_date = st.date_input("Select Sign Up Date", 
                                min_value = customer_data['SIGN_UP_DATE'].min(), 
                                max_value = customer_data['SIGN_UP_DATE'].max(),
                                value = customer_data['SIGN_UP_DATE'].max()-pd.Timedelta(days=365*1))
    birthday_date = st.date_input("Select Birthday Date", 
                                min_value = customer_data['BIRTHDAY_DATE'].min(), 
                                max_value = customer_data['BIRTHDAY_DATE'].max(),
                                value = customer_data['BIRTHDAY_DATE'].max()-pd.Timedelta(days=365*23))

    # Sliders for days since last order, number of orders, and amount spent per order
    days_since_last_order = st.slider("Days Since Last Order:", 1, 40, 10)
    num_of_orders = st.slider("Number of Orders:", 1, 60, 40)
    amount_spent_per_order = st.slider("Amount Spent per Order ($):", 2, 30, 8)

    df_user_input = pd.DataFrame({
        'CITY': [city],
        'GENDER': [gender],
        'MARITAL_STATUS': [marital_status],
        'CHILDREN_COUNT': [children_count],
        'SIGN_UP_DATE': [sign_up_date],
        'BIRTHDAY_DATE': [birthday_date],
        'Days_Since_Last_Order': [days_since_last_order],
        'Number_of_Orders': [num_of_orders],
        'Amount_Spent_Per_Order': [amount_spent_per_order]
    })
    
    return df_user_input

# globals

# path to assets
path = 'assets/'    

# datasets
customer_data, df_before_scaling_sample, cluster_information = init_dataset()  
    
# models
yeojohnsontransformer, onehotencoder, minmaxscaler, model, onehotencoder_cust_demo, model_cust_demo = init_model()
   
# set page conifg
st.set_page_config(page_title="Churn Prediction", page_icon="💀") 

# Show basic statistics about the customer data
# TODO - connect to Snowflake with SQL to get transaction data
st.title("Customer Data Overview")
st.write("Total number of Transactions:", "673M")
st.write("Total number of Transactions by Members:", "37M")
st.write('')
st.write("Number of Members:", "222K")
st.write("Number of Unique Data on Members:", str(len(customer_data.columns)))

# Display a sample of the customer data
st.subheader("Customer Data")
st.dataframe(customer_data)

# search bar for customer data
cust_name_search(customer_data)
    
# Show insights on customer churn
st.header("Customer Churn Predictor")
st.write("The following section will predict the likelihood of a customer churning based on their demographic and transactional data.")
st.subheader("Select a customer to predict their churn")
selected_customer = cust_id_search(customer_data)
st.write("or select input a customer's data manually")
df_user_input = user_input_features(selected_customer)

        
  


    
    
    
    
    
    