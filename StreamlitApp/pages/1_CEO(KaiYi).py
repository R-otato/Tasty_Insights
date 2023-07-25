#--Team--
# Tutorial Group: 	T01 Group 4 

#--Import statements--
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

import plotly.express as px
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
st.set_page_config(page_title="Churn Prediction", page_icon="💀")

st.markdown("# Churn Prediction")
tab1, tab2 = st.tabs(['Explore', 'Predict'])

with tab1:

    df = pd.read_csv('assets/testcoord.csv')
    df.columns = ['lat', 'lon', 'name', 'sum']

    st.write("""
    ## What is this?

    This is a map that represents the distribution of the churn based on the level at which the data is analysed. i.e. Country/City/Truck
    """)

    fig = px.scatter_mapbox(df, lat="lat", lon="lon", size = df['sum'])

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

with tab2:
    st.sidebar.header("Churn Prediction Demo")

    st.write("""
    ## How to use this tool?

    You only need to provide the parameters to the machine learning model at the sidebar on the left side of this page. And the predictions made by the model will be outputted right below.
    """)

    st.write("""
    ## Parameters Imputed:

    - Down below are the parameters setted up to the model by the inputs of the sidebar.
    """)
    test_data=pd.read_csv('assets/testdata.csv').drop(['CHURNED'],axis=1,errors='ignore')

    st.write(test_data)
    type='Example'
    customer_id = test_data.pop("CUSTOMER_ID")

    #--File Upload--
    st.markdown("## Multiple File Upload")
    uploaded_files = st.file_uploader('Upload your file', accept_multiple_files=True)


    if uploaded_files!=[]:
        type='Your'
        for f in uploaded_files:
            st.write(f)
        data_list = []
        for f in uploaded_files:
            temp_data = pd.read_csv(f)
            data_list.append(temp_data)

        data = pd.concat(data_list)

        st.dataframe(data)

        # #-- Prediction Result --
        # st.write('## Prediction Results:')

        # prediction = get_prediction(data)
        # predictionMsg = '***Not Churn***' if float(prediction['Churn'][0][:-1]) <= 50 else '***Churn***'
        # predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

        # st.write(f'The model predicted a percentage of **{predictionPercent}** that the custumer will {predictionMsg}!')
        # st.write(prediction)
    else:
        data=test_data.copy()
    #--Get Prediction--

    # get feature columns
    feature_cols = test_data.columns
    
    with st.spinner('Wait for it...'):
        # udf_score_xgboost_model_vec_cached  = session.udf.register(func=udf_proba_xgboost_model_vec_cached, 
        #                                                                 name="udf_score_xgboost_model", 
        #                                                                 stage_location='@MODEL_STAGE',
        #                                                                 input_types=[T.FloatType()]*len(feature_cols),
        #                                                                 return_type = T.FloatType(),
        #                                                                 replace=True, 
        #                                                                 is_permanent=True, 
        #                                                                 imports=['@MODEL_STAGE/xgboost_model.sav'],
        #                                                                 packages=[f'xgboost==1.7.3'
        #                                                                             ,f'joblib==1.1.1'
        #                                                                             ,f'cachetools==4.2.2'], 
        #                                                                 session=session)
        # data = pd.concat([customer_id, data], axis=1)
        # data=session.create_dataframe(data)
        # proba_data=udf_score_xgboost_model_vec_cached(*feature_cols)
        # pred=data.with_column('CHURN_PROBABILITY', proba_data)
        # st.markdown("# "+type+" Results")
        # st.write('*Tips: Click on column name to sort!')
        # st.dataframe(pred[['CUSTOMER_ID','CHURN_PROBABILITY']])  
        # st.success('Done!')  
        model = XGBClassifier()
        model.load_model("assets/model.json")
        predictions= pd.DataFrame(model.predict_proba(data),columns=['NotChurn','Churned'])
        
        st.dataframe(predictions)
        data=pd.concat([customer_id, predictions], axis=1)
        
        st.dataframe(data)

    st.button("Re-run")


import streamlit as st
import pandas as pd
import joblib 

# Load the pre-trained machine learning model
model = joblib.load('assets/churn-prediction-model.jbl')

# Load the customer data from a CSV file
customer_data = pd.read_csv('assets/testdata.csv')

# Page Title
st.title("Churn Prediction")

# Display a brief overview of the company
st.markdown("""
            ### What is Churn? 
            Churn is the percentage of customers that stopped using our service during a certain time frame. 
            
            For Tasty Bytes, that consititudes as customers who have not made a purchase in the last 7 days.
            
            We can calculate churn rate by dividing the number of customers we lost during 7 days by the number of customers we had at the beginning of that time period.
            """)

# Show basic statistics about the customer data
st.header("Customer Data Overview")
st.write("Total number of Transactions:", len(customer_data))
st.write("Total number of Transactions by Members:", len(customer_data))
st.write('')
st.write("Number of Members:", len(customer_data))
st.write("Number of Unique Data on Members:", len(customer_data.columns))

# Display a sample of the customer data
st.subheader("Customer Data")
st.dataframe(customer_data.head())

# Allow the CEO to select a specific customer ID to get more details
selected_customer_id = st.selectbox("Select a customer ID to view more details:", customer_data['CUSTOMER_ID'])

if selected_customer_id:
    selected_customer = customer_data[customer_data['CUSTOMER_ID'] == selected_customer_id]
    st.subheader("Selected Customer Details")
    st.dataframe(selected_customer)

    # Predict churn probability for the selected customer
    features = selected_customer.drop(['CUSTOMER_ID', 'Churn'], axis=1)
    churn_probability = model.predict_proba(features)[0][1]
    st.write(f"Churn Probability for Customer {selected_customer_id}: {churn_probability:.2f}")

# Show insights on customer churn
st.header("Customer Churn Analysis")

# Calculate and display the percentage of churned customers
total_churned = customer_data['Churn'].sum()
total_customers = len(customer_data)
churn_percentage = (total_churned / total_customers) * 100
st.write(f"Percentage of Churned Customers: {churn_percentage:.2f}%")

# Display a bar chart to visualize churn distribution
churn_distribution = customer_data['Churn'].value_counts()
st.bar_chart(churn_distribution)

# Provide recommendations and next steps based on the churn analysis
st.header("Recommendations")
st.markdown("""
- **Retention Strategies:** Implement customer retention strategies based on churn prediction.
- **Feedback Collection:** Gather feedback from churned customers to identify pain points.
- **Targeted Marketing:** Design targeted marketing campaigns to retain at-risk customers.
""")
