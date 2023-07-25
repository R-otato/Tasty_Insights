#--Team--
# Tutorial Group: 	T01 Group 4 

#--Import statements--
# import streamlit as st
# import pandas as pd
# from xgboost import XGBClassifier

# import plotly.express as px
# from snowflake.snowpark.functions import call_udf, col
# import snowflake.snowpark.types as T
# from snowflake.snowpark import Session
# import requests
# import numpy as np
# import joblib 
# import time
# import json
# from cachetools import cached
#----------------------Snowflake---------------------------------#
# # Get account credentials from a json file
# with open("data_scientist_auth.json") as f:
#     data = json.load(f)
#     username = data["username"]
#     password = data["password"]
#     account = data["account"]

# # Specify connection parameters
# connection_parameters = {
#     "account": account,
#     "user": username,
#     "password": password,
#     "role": "TASTY_BI",
#     "warehouse": "TASTY_BI_WH",
#     "database": "frostbyte_tasty_bytes",
#     "schema": "analytics",
# }

# # Create Snowpark session
# session = Session.builder.configs(connection_parameters).create()

# #--Functions--
# # Function to load the model from file and cache the result
# @cached(cache={})
# #Load model
# def load_model(model_path: str) -> object:
#     from joblib import load
#     model = load(model_path)
#     return model

# #Get predictions
# def udf_score_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
#     import sys
#     # file-dependencies of UDFs are available in snowflake_import_directory
#     IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
#     import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
#     model_name = 'xgboost_model.sav'
#     model = load_model(import_dir+model_name)
#     df.columns = feature_cols
#     scored_data = pd.Series(model.predict(df))
#     return scored_data

# def udf_proba_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
#     import sys
#     # file-dependencies of UDFs are available in snowflake_import_directory
#     IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
#     import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
#     model_name = 'xgboost_model.sav'
#     model = load_model(import_dir+model_name)
#     df.columns = feature_cols
#     scored_data = pd.Series(model.predict(df))
#     proba_data = pd.Series(model.predict_proba(df)[:, 1])
#     return proba_data

# def transforma(data):
# #   for feature, fit in joblib.load('assets/labelEncoder_fit.jbl'):
# #     if feature != 'Churn':
# #       data[feature] = fit.transform(data[feature])

# #   for feature in data.drop(['MonthlyCharges', 'tenure'], axis=1).columns:
# #     data[feature] = data[feature].astype('category')

# #   for feature, scaler in joblib.load('assets/minMaxScaler_fit.jbl'):
# #     data[feature] = scaler.transform(data[feature].values.reshape(-1,1))
#     return


#--Introduction--
# st.set_page_config(page_title="Churn Prediction", page_icon="💀")

# st.markdown("# Churn Prediction")
# tab1, tab2 = st.tabs(['Explore', 'Predict'])

# with tab1:

#     df = pd.read_csv('assets/testcoord.csv')
#     df.columns = ['lat', 'lon', 'name', 'sum']

#     st.write("""
#     ## What is this?

#     This is a map that represents the distribution of the churn based on the level at which the data is analysed. i.e. Country/City/Truck
#     """)

#     fig = px.scatter_mapbox(df, lat="lat", lon="lon", size = df['sum'])

#     fig.update_layout(mapbox_style="open-street-map")
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#     st.plotly_chart(fig)

# with tab2:
# st.sidebar.header("Churn Prediction Demo")

# st.write("""
# ## How to use this tool?

# You only need to provide the parameters to the machine learning model at the sidebar on the left side of this page. And the predictions made by the model will be outputted right below.
# """)

# st.write("""
# ## Parameters Imputed:

# - Down below are the parameters setted up to the model by the inputs of the sidebar.
# """)
# test_data=pd.read_csv('assets/testdata.csv').drop(['CHURNED'],axis=1,errors='ignore')

# st.write(test_data)
# type='Example'
# customer_id = test_data.pop("CUSTOMER_ID")

# #--File Upload--
# st.markdown("## Multiple File Upload")
# uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)


# if uploaded_files!=[]:
#     type='Your'
#     for f in uploaded_files:
#         st.write(f)
#     data_list = []
#     for f in uploaded_files:
#         temp_data = pd.read_csv(f)
#         data_list.append(temp_data)

#     data = pd.concat(data_list)

#     st.dataframe(data)

#     # #-- Prediction Result --
#     # st.write('## Prediction Results:')

#     # prediction = get_prediction(data)
#     # predictionMsg = '***Not Churn***' if float(prediction['Churn'][0][:-1]) <= 50 else '***Churn***'
#     # predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

#     # st.write(f'The model predicted a percentage of **{predictionPercent}** that the custumer will {predictionMsg}!')
#     # st.write(prediction)
# else:
#     data=test_data.copy()
# #--Get Prediction--

# # get feature columns
# feature_cols = test_data.columns

# with st.spinner('Wait for it...'):
#     # udf_score_xgboost_model_vec_cached  = session.udf.register(func=udf_proba_xgboost_model_vec_cached, 
#     #                                                                 name="udf_score_xgboost_model", 
#     #                                                                 stage_location='@MODEL_STAGE',
#     #                                                                 input_types=[T.FloatType()]*len(feature_cols),
#     #                                                                 return_type = T.FloatType(),
#     #                                                                 replace=True, 
#     #                                                                 is_permanent=True, 
#     #                                                                 imports=['@MODEL_STAGE/xgboost_model.sav'],
#     #                                                                 packages=[f'xgboost==1.7.3'
#     #                                                                             ,f'joblib==1.1.1'
#     #                                                                             ,f'cachetools==4.2.2'], 
#     #                                                                 session=session)
#     # data = pd.concat([customer_id, data], axis=1)
#     # data=session.create_dataframe(data)
#     # proba_data=udf_score_xgboost_model_vec_cached(*feature_cols)
#     # pred=data.with_column('CHURN_PROBABILITY', proba_data)
#     # st.markdown("# "+type+" Results")
#     # st.write('*Tips: Click on column name to sort!')
#     # st.dataframe(pred[['CUSTOMER_ID','CHURN_PROBABILITY']])  
#     # st.success('Done!')  
#     model = XGBClassifier()
#     model.load_model("assets/model.json")
#     predictions= pd.DataFrame(model.predict_proba(data),columns=['NotChurn','Churned'])
    
#     st.dataframe(predictions)
#     data=pd.concat([customer_id, predictions], axis=1)
    
#     st.dataframe(data)

# st.button("Re-run")

#--Team--
# Tutorial Group: 	T01 Group 4 

#--Import statements--
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

import plotly.express as px
import joblib 

# path to assets
path = 'assets/'

st.set_page_config(page_title="Churn Prediction", page_icon="💀")

# Page Title
st.title("Churn Prediction")

# Display a brief overview of the company
st.subheader("What is Churn?")
st.markdown("""
            Churn is the percentage of customers that stopped using our service during a certain time frame. 
            
            For Tasty Bytes, that consititudes as customers who have not made a purchase in the last 7 days.
            
            We can calculate churn rate by dividing the number of customers we lost during 7 days by the number of customers we had at the beginning of that time period.
            """)

# Load the pre-trained machine learning model
model = joblib.load(path + 'churn-prediction-model.jbl')

# Load the customer data from a CSV file
customer_data = pd.read_csv(path + 'datasets/relavent_original_dataset.csv')

# Show basic statistics about the customer data
# TODO - connect to Snowflake with SQL to get transaction data
st.header("Customer Data Overview")
st.write("Total number of Transactions:", "673M")
st.write("Total number of Transactions by Members:", "37M")
st.write('')
st.write("Number of Members:", "222K")
st.write("Number of Unique Data on Members:", len(customer_data.columns))

# Display a sample of the customer data
st.subheader("Customer Data")
st.dataframe(customer_data)

# to search for a specific customer by first name, last name, or combination
search_term = st.text_input("Search for a customer by First Name, Last Name, or combination:")

if search_term:
    # Filter the customer data based on the search term using partial matching (LIKE)
    selected_customer = customer_data[customer_data['FIRST_NAME'].str.contains(search_term, case=False, na=False) |
                                      customer_data['LAST_NAME'].str.contains(search_term, case=False, na=False) |
                                      customer_data.apply(lambda row: search_term in f"{row['FIRST_NAME']} {row['LAST_NAME']}".lower(), axis=1)]

    
    if not selected_customer.empty:
        st.subheader("Selected Customer Details")
        st.dataframe(selected_customer)
    else:
        st.write("No matching customer found.")

# Show insights on customer churn
st.header("Customer Churn Predictor")

customer_data['SIGN_UP_DATE'] = pd.to_datetime(customer_data['SIGN_UP_DATE'])
customer_data['BIRTHDAY_DATE'] = pd.to_datetime(customer_data['BIRTHDAY_DATE'])

# User Input for Dropdowns
city = st.selectbox("Select City", np.sort(customer_data['CITY'].unique()))
gender = st.selectbox("Select Gender", np.sort(customer_data['GENDER'].unique()))
marital_status = st.selectbox("Select Marital Status", np.sort(customer_data['MARITAL_STATUS'].unique()))
children_count = st.selectbox("Select Children Count", np.sort(customer_data['CHILDREN_COUNT'].unique()))


sign_up_date = st.date_input("Select Sign Up Date", 
                             min_value = customer_data['SIGN_UP_DATE'].min(), 
                             max_value = customer_data['SIGN_UP_DATE'].max(),
                             value = customer_data['SIGN_UP_DATE'].min())
birthday_date = st.date_input("Select Birthday Date", 
                             min_value = customer_data['BIRTHDAY_DATE'].min(), 
                             max_value = customer_data['BIRTHDAY_DATE'].max(),
                             value = customer_data['BIRTHDAY_DATE'].min())

# Sliders for days since last order, number of orders, and amount spent per order
days_since_last_order = st.slider("Days Since Last Order:", 1, 40, 1)
num_of_orders = st.slider("Number of Orders:", 1, 60, 1)
amount_spent_per_order = st.slider("Amount Spent per Order ($):", 2, 30, 2)

# DataFrame based on user input
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

df_user_input['BIRTHDAY_DATE'] = pd.to_datetime(df_user_input['BIRTHDAY_DATE'])
df_user_input['SIGN_UP_DATE'] = pd.to_datetime(df_user_input['SIGN_UP_DATE'])


# Display the filtered data
st.subheader("Customer to Predict Data")
st.dataframe(df_user_input)

if st.button('Transform Input'):
    # RFM
    df_user_input['RECENCY'] = df_user_input['Days_Since_Last_Order']
    df_user_input['FREQUENCY'] = df_user_input['Number_of_Orders']
    df_user_input['MONETARY'] = df_user_input['Amount_Spent_Per_Order']*df_user_input['Number_of_Orders']
    
    # Average Time Difference, Max and Min Days without Purchase
    # TODO - ryan suggests Or maybe take the mean or median of the 3 columnsAnd say the values are set to this Because...
    df_before_scaling_sample = pd.read_csv(path + 'datasets/before_scaling_dataset.csv')
    df_user_input['AVG_DAYS_BETWEEN_PURCHASE'] = df_before_scaling_sample['AVG_DAYS_BETWEEN_PURCHASE'].mean()
    df_user_input['MAX_DAYS_WITHOUT_PURCHASE'] = df_before_scaling_sample['MAX_DAYS_WITHOUT_PURCHASE'].mean()
    df_user_input['MIN_DAYS_WITHOUT_PURCHASE'] = df_before_scaling_sample['MIN_DAYS_WITHOUT_PURCHASE'].mean()
    
    # Age
    latest_date = pd.to_datetime('2022-10-31')
    df_user_input['AGE'] = (latest_date - df_user_input['BIRTHDAY_DATE']).dt.days/365
    # Number of locations visited
    df_user_input['NUM_OF_LOCATIONS_VISITED'] = df_user_input['Number_of_Orders'] \
                                                - round(df_user_input['Number_of_Orders']*0.05)
    # Length of relationship
    df_user_input['LENGTH_OF_RELATIONSHIP'] = (latest_date - df_user_input['SIGN_UP_DATE']).dt.days
    # Relative Purchase Frequency and Monetary
    df_user_input['RELATIVE_PURCHASE_FREQUENCY'] = df_user_input['FREQUENCY'] / df_user_input['LENGTH_OF_RELATIONSHIP']
    df_user_input['RELATIVE_PURCHASE_MONETARY'] = df_user_input['MONETARY'] / df_user_input['LENGTH_OF_RELATIONSHIP']

    # drop unnecessary columns
    df_user_input = df_user_input.drop(columns=['Days_Since_Last_Order', 'Number_of_Orders','Amount_Spent_Per_Order'])
    
    # add dummy columns needed for numerical transformation
    df_user_input = df_user_input.assign(
        CUSTOMER_ID=1,
        COUNTRY='United States',
        MAX_ORDER_TS=latest_date,
        ORDER_TS=latest_date,
        DAYS_TO_NEXT_ORDER=999,
    )
    
    # perform numerical transformation
    yeojohnsontransformer = joblib.load(path + '/models/yeojohnsontransformer.jbl')
    df_user_input = yeojohnsontransformer.transform(df_user_input)
    
    df_user_input = df_user_input.drop(columns=['CUSTOMER_ID','DAYS_TO_NEXT_ORDER',
                                'MAX_ORDER_TS','ORDER_TS',
                                'COUNTRY','BIRTHDAY_DATE','SIGN_UP_DATE'])
    
    st.markdown(df_user_input.columns)
    
    # add dummy columns needed for the one hot encoding
    df_user_input = df_user_input.assign(
        CHURNED = 0,
    )
    
    # perform one hot encoding
    onehotencoder = joblib.load(path + '/models/onehotencoder.jbl')
    df_user_input = onehotencoder.transform(df_user_input)
    df_user_input.rename({'MARITAL_STATUS_Divorced/Seperated':'MARITAL_STATUS_Divorced_Or_Seperated'}, axis=1,inplace=True)
    df_user_input.columns = map(str.upper, df_user_input.columns)

    # perform scaling
    df_user_input = df_user_input.drop(columns=['CHURNED'])
    minmaxscaler = joblib.load(path + 'models/minmaxscaler.jbl')
    df_userinput = minmaxscaler.transform(df_user_input)
    
    st.dataframe(df_user_input)
    
    model = joblib.load(path + 'models/xgb_churn_model.jbl')
    prediction = model.predict(df_user_input)
    st.write(prediction)
    
    
    # TODO - add customer segmentation, where does this customer belong?
    # TODO - add customer churn, what is the probability of this customer churning?
    # TODO - add chloropleth of sales by country
    
    