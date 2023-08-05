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

import snowflake.connector

def pipeline(data):
    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/models/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/models/windsorizer_gau.jbl")
    yjt = joblib.load("assets/models/yjt.jbl")
    ohe_enc = joblib.load("assets/models/ohe_enc.jbl")
    minMaxScaler = joblib.load("assets/models/minMaxScaler.jbl")

    # Apply the transformations to the data
    data = windsorizer_iqr.transform(data)  # Apply IQR Windsorization
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization
    data = yjt.transform(data)  # Apply Yeo-Johnson Transformation
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data.columns = data.columns.str.upper()
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    return data

@cached(cache={})
#Load model
def load_model(model_path: str) -> object:
    model = joblib.load(model_path)
    return model   

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# hide this using secrets
# my_cnx = snowflake.connector.connect(
#     user = "RLIAM",
#     password = "Cats2004",
#     account = "LGHJQKA-DJ92750",
#     role = "TASTY_BI",
#     warehouse = "TASTY_BI_WH",
#     database = "frostbyte_tasty_bytes",
#     schema = "analytics"
# )

# my_cur = my_cnx.cursor()
# my_cur.execute("select * from churn_to_sales")
# churn_to_sales = my_cur.fetchall()

# # not really efficient figure out a way to extract snowflake dataframe instead of tuple
# df = pd.DataFrame(churn_to_sales, columns=["YEAR", "MONTH", "CHURN RATE", "SALES"])

# df = pd.read_csv('assets/Churn_to_Sales.csv')

df_OTS = pd.read_csv('assets/datasets/latest_Order_Ts.csv')

# df_cts = df.sort_values(by=["YEAR", "MONTH"])

st.set_page_config(page_title="CFO-Sales Prediction",
                   page_icon="🍌")

st.markdown("# Sales Prediction")


# Commented code is previous iteration which included the churn link to sales
# which has been moved to home page.

# with tab1:

#     st.write("""
#     ## How does Churn link to Sales?

#     According to prelimnary research during the pitch phase we have established a link between Days to Next Order (target variable)
#     which is what we derive our churn prediction from, there is a clear link between the average days to next order in a month with 
#     that months sales.

#     Through the prediction of churn rate we will imply the average days to next orders and in turn predict the potential sales of
#     following month allowing for projection in sales and allow for informed steps to be taken in terms of operations and strategy.
#     """)

#     # insert table lining the average DTNO with the Churn rate of month with order totals and explain the percentage changes

#     # showing historical data linking the churn rate to the changes in sales
#     st.dataframe(
#         df_cts,
#         hide_index=True
#     )

# with tab2:
# page title
st.markdown("## CFO")
st.write("""This page shows a prediction of next months sales, the inital churn percent value is an underestimate
         of the months actual churn value. Here we can see what our churn value can hit and while still achieving our goal of
         25 percent growth in sales, as well as changing number of members to see the effect on the sales change.""")

    # page guide
with st.expander("Guide to Using Page"):
    st.write("""This page provides data to be used in the PowerBI visualiser.
                 By inputing data regarding our members based on their latest transactions we are able to make a
                 prediction of the churn of members (churn is a twin of member recursion).
                 The output data will contain the churn of each member as well as some categorising information
                 to assist in visualising and showing areas of churn. This information will help to predict the effct
                 of churn on sales in different regions.
            """)
with st.expander("Page Disclaimer"):
    st.write("""An issue with this page is the inital churn number is inaccurate
                Based on how the churn value is calculated (transactions that churned/transactions in month)
                the output of the initial prediction may not be accurate because of the churn %
                adjust the churn percent to see how many transactions can churn while still
                meeting KPI goals.""")
    

st.info("Using the last updated data of the members in United States (October and beyond).")
history_data = pd.read_csv("assets/datasets/last_month_sales.csv")
st.write("This is last months sales")
st.write(history_data)

#df=pd.read_csv('StreamlitApp/assets/without_transformation.csv')
df=pd.read_csv('assets/datasets/without_transformation.csv')
df = df.merge(df_OTS)
df = df[(df["MAX_ORDER_TS"]) > "2022-10-01"]
# df = df[(df["MAX_ORDER_TS"]) <= "2022-10-31"]
df = df.drop(columns="MAX_ORDER_TS")

# with st.expander("Raw Dataframe"):
#         st.write("This is the data set prior to any transformations")
#         st.write(df)
#         st.write(df.info())

    ## Removing Customer ID column
customer_id = df.pop("CUSTOMER_ID")
    # Get categoorical columns
demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

df=pipeline(df)

# with st.expander("Cleaned and Transformed Data"):
#         st.write("This is the data set after cleaning and transformation")
#         st.write(df)

model = load_model("assets/models/churn-prediction-model.jbl")
predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
demo_df = pd.concat([demo_df, predictions], axis=1)
beha_df = pd.concat([beha_df, predictions], axis=1)
data=pd.concat([customer_id, predictions], axis=1)

    # st.write(demo_df)
    # st.write(beha_df)
    # st.write(data)
    
output_data = pd.concat([data, demo_df[["GENDER", "CITY", "AGE"]]], axis=1)
output_data = output_data.dropna()
#output_data["AGE"] = output_data["AGE"].astype(int)    

with st.expander("Output Data"):
      st.write(output_data)

    # temporary test
output = convert_df(output_data)
st.download_button("Download Data", output, "file.csv", "text/csv", key='download-csv')

# see potential sales
# predicting next month sales and growth
model2 = load_model("assets/models/nextMonth.jbl")
# inputs: Churn rate, distinct members, month of year, average cust age

# create button to select which cities and change the dataset accordingly
city = st.selectbox(label="Select City", options=output_data["CITY"].unique())
using = output_data.loc[output_data["CITY"] == city]

# output_data

churn_rate = using[['CHURNED']].sum()/using[['CHURNED']].count()

# Presenting Churn Rate
value = round(churn_rate.iloc[0] * 100, 2)
st.metric('Existing members Predicted to Churn', f"{value}%")


input_data = pd.DataFrame()

input_data["CHURN_RATE"] = churn_rate
input_data["DISTINCT_CUSTOMER"] = using.count()
input_data["MONTH"] = 10
input_data["AVERAGE_AGE"] = using["AGE"].mean()


st.markdown("### Based on month initial information")
# data to input in the model
st.write(input_data)

pred_sales = model2.predict(input_data)
st.metric("Predicted Sales", f"${pred_sales[0]:.2f}")
# st.write(f"The predicted sales next month of {city} is {pred_sales[0]:.2f}")

# Change in Sales
lastMonthSales = history_data.loc[history_data["CITY"] == city]["SALES"].values  
change_sales = (pred_sales - lastMonthSales)/lastMonthSales * 100
st.metric("Change in sales", f"{change_sales[0]:.2f}%")

st.markdown("### Adjust churn and distinct members to see effect on sales")
new_churn = st.slider(label="Adjust Churn Rate", min_value=0, max_value=100)
new_members = st.slider(label="Adjust Distinct members", min_value=0, max_value=20000, value=len(using))

# show adjusted sales
input_data_new = input_data.copy()
input_data_new["CHURN_RATE"] = new_churn/100
input_data_new["DISTINCT_CUSTOMER"] = new_members

new_pred_sales = model2.predict(input_data_new)
st.metric("Adjusted Predicted Sales", f"${new_pred_sales[0]:.2f}")
# st.write(f"The adjust predicted sales next month of {city} is {new_pred_sales[0]:.2f}")

change_sales_new = (new_pred_sales - lastMonthSales)/lastMonthSales * 100
st.metric("Adjusted Change in Sales", f"{change_sales_new[0]:.2f}%")
