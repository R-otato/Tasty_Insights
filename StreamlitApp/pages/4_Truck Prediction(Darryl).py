#--Import statements--
import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib 
import time
import ast
import io
import zipfile
from snowflake.snowpark import Session
import json
from snowflake.snowpark.functions import call_udf, col
import snowflake.snowpark.types as T
from cachetools import cached

import snowflake.connector

# Function: pipeline
# the purpose of this function is to carry out the necessary transformations on the data provided by the
# user so that it can be fed into the machine learning model for prediction
def pipeline(data):
    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/windsorizer_gau.jbl")
    yjt = joblib.load("assets/yjt.jbl")
    ohe_enc = joblib.load("assets/ohe_enc.jbl")
    minMaxScaler = joblib.load("assets/minMaxScaler.jbl")

    # Apply the transformations to the data
    data = windsorizer_iqr.transform(data)  # Apply IQR Windsorization
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization
    data = yjt.transform(data)  # Apply Yeo-Johnson Transformation
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data.columns = data.columns.str.upper()
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    return data
        
# @cached(cache={})
# #Load model
# def load_model(model_path: str) -> object:
#     model = joblib.load(model_path)
#     return model   

# def convert_df(df):
#     return df.to_csv(index=False).encode('utf-8')

# Function: retrieve truck table
# the purpose of this function is to retrieve the truck table from snowflake containing all the details of the truck items
def retrieve_truck_table():
    # RETRIEVE TRUCK TABLE FROM SNOWFLAKE
    ## get connection to snowflake
    my_cnx = snowflake.connector.connect(
        user = "RLIAM",
        password = "Cats2004",
        account = "LGHJQKA-DJ92750",
        role = "TASTY_BI",
        warehouse = "TASTY_BI_WH",
        database = "frostbyte_tasty_bytes",
        schema = "raw_pos"
    )

    ## retrieve truck table from snowflake
    my_cur = my_cnx.cursor()
    # select united states
    my_cur.execute("select TRUCK_ID, PRIMARY_CITY, REGION, COUNTRY, FRANCHISE_ID, MENU_TYPE_ID from truck where COUNTRY = 'United States'")
    truck_table = my_cur.fetchall()
    
    ## create a DataFrame from the fetched result
    truck_table_df = pd.DataFrame(truck_table, columns=['TRUCK_ID', 'PRIMARY_CITY', 'REGION', 'COUNTRY', 'FRANCHISE_ID', 'MENU_TYPE_ID'])

    # Group the DataFrame by 'PRIMARY_CITY' and count the number of trucks in each city
    # truck_count_by_city = truck_table_df.groupBy("PRIMARY_CITY").count()

    return truck_table_df


# Function: retrive_menu_table
# the purpose of this function is to retrieve the menu table from snowflake containing all the details of the menu items which will then be merged with the transactions info to help the product team gain insight
def retrieve_menu_table():
    ## get connection to snowflake
    my_cnx = snowflake.connector.connect(
        user = "RLIAM",
        password = "Cats2004",
        account = "LGHJQKA-DJ92750",
        role = "TASTY_BI",
        warehouse = "TASTY_BI_WH",
        database = "frostbyte_tasty_bytes",
        schema = "raw_pos"
    )

    # retrieve menu table from snowflake
    my_cur = my_cnx.cursor()
    my_cur.execute("select MENU_TYPE_ID, MENU_ITEM_ID, TRUCK_BRAND_NAME, COST_OF_GOODS_USD, SALE_PRICE_USD from menu")
    menu_table = my_cur.fetchall()

    # create a DataFrame from the fetched result
    # remove cost of goods column due to irrelevance
    menu_table_df = pd.DataFrame(menu_table, columns=['MENU_TYPE_ID', 'MENU_ITEM_ID', 'TRUCK_BRAND_NAME', 'COST_OF_GOODS_USD', 
                                                    'SALE_PRICE_USD'])


    # rename the 'SALE_PRICE_USD' column to 'SALE_PRICE'
    menu_table_df = menu_table_df.rename(columns={'SALE_PRICE_USD': 'SALE_PRICE'})

    # rename the 'COST_OF_GOODS_USD' to 'COST_OF_GOODS'
    menu_table_df = menu_table_df.rename(columns={'COST_OF_GOODS_USD': 'COST_OF_GOODS'})
    
    
    # Add profit column
    ## calculate the profit column
    profit_column = menu_table_df['SALE_PRICE'] - menu_table_df['COST_OF_GOODS']

    ## get the index of the 'COST_OF_GOODS' column
    cost_of_goods_index = menu_table_df.columns.get_loc('COST_OF_GOODS')

    ## insert the 'profit' column to the right of the 'COST_OF_GOODS' column
    menu_table_df.insert(cost_of_goods_index + 1, 'SALE_PROFIT', profit_column)
    
    
    # Add gross profit margin column
    ## calculate gross profit margin
    gross_profit_margin = ((menu_table_df['SALE_PRICE'] - menu_table_df['COST_OF_GOODS']) / menu_table_df['SALE_PRICE']) * 100
    
    ## get the index of the 'SALE_PROFIT' column
    sale_profit_index = menu_table_df.columns.get_loc('SALE_PROFIT')
    
    ## insert the 'SALE_GROSS_PROFIT_MARGIN (%)' column to the right of the 'SALE_PROFIT' column
    menu_table_df.insert(sale_profit_index + 1, 'SALE_GROSS_PROFIT_MARGIN (%)', gross_profit_margin)
    
    
    # Add net profit margin column
    ## calculate net profit margin
    net_profit_margin = (menu_table_df['SALE_PROFIT'] / menu_table_df['SALE_PRICE']) * 100
    
    ## get the index of the 'SALE_GROSS_PROFIT_MARGIN (%)' column
    sale_gross_profit_margin_index = menu_table_df.columns.get_loc('SALE_GROSS_PROFIT_MARGIN (%)')
    
    ## insert the 'SALE_GROSS_PROFIT_MARGIN (%)' column to the right of the 'SALE_PROFIT' column
    menu_table_df.insert(sale_gross_profit_margin_index + 1, 'SALE_NET_PROFIT_MARGIN (%)', net_profit_margin)
    
    
    # round off sale price to 2dp
    menu_table_df['SALE_PRICE'] = menu_table_df['SALE_PRICE'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off cost of goods price to 2dp
    menu_table_df['COST_OF_GOODS'] = menu_table_df['COST_OF_GOODS'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off profit amount to 2dp
    menu_table_df['SALE_PROFIT'] = menu_table_df['SALE_PROFIT'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off gross profit margin to 2dp
    menu_table_df['SALE_GROSS_PROFIT_MARGIN (%)'] = menu_table_df['SALE_GROSS_PROFIT_MARGIN (%)'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off net profit margin to 2dp
    menu_table_df['SALE_NET_PROFIT_MARGIN (%)'] = menu_table_df['SALE_NET_PROFIT_MARGIN (%)'].apply(lambda x: '{:.2f}'.format(x))
    
    return menu_table_df

# Function: retrieve order_header table
# the purpose of this function is to retrieve the order_header table from snowflake containing all the details of the order_header items
def retrieve_order_details():
    # RETRIEVE order_header TABLE FROM SNOWFLAKE
    ## get connection to snowflake
    my_cnx = snowflake.connector.connect(
        user = "RLIAM",
        password = "Cats2004",
        account = "LGHJQKA-DJ92750",
        role = "TASTY_BI",
        warehouse = "TASTY_BI_WH",
        database = "frostbyte_tasty_bytes",
        schema = "raw_pos"
    )

    ## retrieve order_header table from snowflake
    my_cur = my_cnx.cursor()
    
    # Retrieve the list of customer IDs from the 'data' table
    customer_ids = data['CUSTOMER_ID'].tolist()

    # Split the list into smaller chunks of 1,000 customer IDs
    chunk_size = 1000
    customer_id_chunks = [customer_ids[i:i+chunk_size] for i in range(0, len(customer_ids), chunk_size)]

    # Execute queries for each customer ID chunk
    order_details = []
    for chunk in customer_id_chunks:
        # Create a comma-separated string of the customer IDs in the current chunk
        customer_ids_str = ','.join(map(str, chunk))

        # Construct the SQL query for the current chunk
        query = f"SELECT TRUCK_ID, CUSTOMER_ID, ORDER_AMOUNT, ORDER_TOTAL, ORDER_CURRENCY from order_header WHERE CUSTOMER_ID IN ({customer_ids_str})"

        # Execute the SQL query for the current chunk
        my_cur.execute(query)

        # Fetch the result for the current chunk
        chunk_result = my_cur.fetchall()

        # Append the chunk result to the overall result
        order_details.extend(chunk_result)

    # Create a DataFrame from the fetched result
    order_details_df = pd.DataFrame(order_details, columns=['TRUCK_ID', 'CUSTOMER_ID', 'ORDER_AMOUNT', 'ORDER_TOTAL', 'ORDER_CURRENCY'])

    # Convert ORDER_ID to string and then remove commas
    #order_details_df['ORDER_ID'] = order_details_df['ORDER_ID'].astype(str).str.replace(',', '')

    # Format ORDER_TOTAL and PRODUCT_TOTAL_PRICE columns to 2 decimal places
    order_details_df['ORDER_TOTAL'] = order_details_df['ORDER_TOTAL'].apply(lambda x: '{:.2f}'.format(x))

    order_details_df = order_details_df.sort_values(by='CUSTOMER_ID')

    return order_details_df


# Function: get_overall_truck_table
# the purpose of this function is to merge the truck and menu details table together to form an overall table
def get_overall_truck_table(truck_table_df, order_details_df):
    ## Merge the DataFrames based on 'MENU_ITEM_ID'
    overall_truck_df = pd.merge(truck_table_df, order_details_df, on='TRUCK_ID', how='left')

    ## Define the desired column order
    desired_columns = ['COUNTRY', 'REGION','PRIMARY_CITY', 'TRUCK_ID', 'CUSTOMER_ID', 'ORDER_TOTAL']

    ## Re-arrange the columns in the merged DataFrame
    overall_truck_df = overall_truck_df[desired_columns]
    
    ## Cast 'ORDER_TOTAL' to float
    overall_truck_df['ORDER_TOTAL'] = overall_truck_df['ORDER_TOTAL'].astype(float)
    
    ## Group by 'TRUCK_ID' and combine 'ORDER_TOTAL' for each truck
    overall_truck_df_grouped = overall_truck_df.groupby('TRUCK_ID').agg({
        'ORDER_TOTAL': 'sum'           # Sum the 'ORDER_TOTAL' amounts for each truck_id
    }).reset_index()
    
    return overall_truck_df_grouped

# # Function: retrieve order_header table
# # the purpose of this function is to retrieve the order_header table from snowflake containing all the details of the order_header items
# def retrieve_order_header_table():
#     # RETRIEVE order_header TABLE FROM SNOWFLAKE
#     ## get connection to snowflake
#     my_cnx = snowflake.connector.connect(
#         user = "RLIAM",
#         password = "Cats2004",
#         account = "LGHJQKA-DJ92750",
#         role = "TASTY_BI",
#         warehouse = "TASTY_BI_WH",
#         database = "frostbyte_tasty_bytes",
#         schema = "raw_pos"
#     )

#     ## retrieve order_header table from snowflake
#     my_cur = my_cnx.cursor()

#     my_cur.execute("select TRUCK_ID, ORDER_AMOUNT, ORDER_TAX_AMOUNT, ORDER_DISCOUNT_AMOUNT, ORDER_TOTAL from order_header where ORDER_CURRENCY = 'USD'")
#     order_header_table = my_cur.fetchall()
    
#     ## create a DataFrame from the fetched result
#     order_header_table_df = pd.DataFrame(order_header_table, columns=['TRUCK_ID', 'ORDER_AMOUNT', 'ORDER_TAX_AMOUNT', 'ORDER_DISCOUNT_AMOUNT', 'ORDER_TOTAL'])

#     return order_header_table_df


# Function: get_overall_table
# the purpose of this function is to merge the truck and menu details table together to form an overall table
def get_overall_table(truck_details_df, menu_table_df):
    ## Merge the DataFrames based on 'MENU_ITEM_ID'
    merged_df = pd.merge(truck_details_df, menu_table_df, on='MENU_TYPE_ID', how='left')

    ## Define the desired column order
    desired_columns = ['TRUCK_ID', 'TRUCK_BRAND_NAME', 'COUNTRY', 'REGION','PRIMARY_CITY', 'SALE_PRICE', 'COST_OF_GOODS', 'SALE_PROFIT']

    ## Re-arrange the columns in the merged DataFrame
    merged_df = merged_df[desired_columns]
    
    # merged_df['COST_OF_GOODS'] = merged_df['COST_OF_GOODS'].astype(float) * merged_df['QUANTITY'].astype(int)
    
    return merged_df


#Load model
def load_model(model_path: str) -> object:
    model = joblib.load(model_path)
    return model    

#Convert dataframe to csv 
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

#Filter dataframe
def filter(selected_options,column,data):
    # Filter the data based on selected clusters
    if 'All' in selected_options:
        filtered_data = data  # If 'All' is selected, show all data
    else:
        filtered_data = data[data[column].isin(selected_options)]
    return filtered_data

#####################
##### MAIN CODE #####
#####################

# Page Title
st.set_page_config(page_title="Truck Prediction", page_icon="📈")

st.markdown("# Truck Prediction")
tab1, tab2 = st.tabs(['Explore', 'Cluster'])

with tab1:
    st.markdown("##")
    
# Input data
    ## File Upload section
    st.markdown("## Input Data")
    uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
    df=''
    ### If uploaded file is not empty
    if uploaded_files:
        data_list = []
        #Append all uploaded files into the list
        for f in uploaded_files:
            st.write(f)
            temp_data = pd.read_csv(f)
            data_list.append(temp_data)
        st.success("Uploaded your file!")
        #concat the files together if there are more than one file uploaded
        df = pd.concat(data_list)
    else:
        st.info("Using the last updated data of the members in United States. Upload a file above to use your own data!")
        #df=pd.read_csv('StreamlitApp/assets/without_transformation.csv')
        df=pd.read_csv('assets/without_transformation.csv')

    # ## Display uploaded or default file
    # with st.expander("Raw Dataframe"):
    #     st.write("This is the data set prior to any transformations")
    #     st.write(df)
    
    # Preparing the data for prediction
    ## Removing Customer ID column
    customer_id = df.pop("CUSTOMER_ID")
    #Get categoorical columns
    demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
    beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

    df=pipeline(df)

    # with st.expander("Cleaned and Transformed Data"):
    #     st.write(df)
        
    # MODEL FOR PREDICTION
    model = joblib.load("assets/churn-prediction-model.jbl")
    predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
    demo_df = pd.concat([demo_df, predictions], axis=1)
    beha_df = pd.concat([beha_df, predictions], axis=1)

    ## ***table with custid and whether churn or not***
    data=pd.concat([customer_id, predictions], axis=1)
    
    ###TEST###
    import plotly.graph_objs as go
    # Map 0 n 1 to churn and not churn respectively
    data['CHURNED'] = data['CHURNED'].map({0: 'Not Churned', 1: 'Churned'})
    # Add some filters to filter by churn --> not churned
    churn_Options = ['All'] + pd.Series(data['CHURNED']).unique().tolist()
    selected_Churn = st.multiselect("Filter by Churn:", churn_Options, default=['All'])

    def filter_data(selected_values, column, data):
        if 'All' in selected_values:
            return data
        return data[data[column].isin(selected_values)]

    filtered_data = filter_data(selected_Churn, 'CHURNED', pd.DataFrame(data))

    # Number of members who churned and not churned
    churn_counts = filtered_data.groupby('CHURNED').size().reset_index(name='Number of Members')
    st.dataframe(churn_counts, hide_index=True)

    # Create a pie chart based on the churn counts
    fig = go.Figure(data=[go.Pie(labels=churn_counts['CHURNED'], values=churn_counts['Number of Members'])])
    st.plotly_chart(fig)
    
    
    
    # # replace '0' and '1' to 'not churned' and 'churned' respectively
    # data['CHURNED'] = data['CHURNED'].map({0: 'Not Churned', 1: 'Churned'})
    # ## Add some filters to filter by churn --> not churned
    # churn_Options = ['All'] + data['CHURNED'].unique().tolist()
    # selected_Churn= st.multiselect("Filter by Churn:",churn_Options, default=['All'])
    # filtered_data=filter(selected_Churn,'CHURNED',data)
    # # Number of members who churned and not churned
    # churn_counts = filtered_data.groupby('CHURNED').size().reset_index(name='Number of Members')
    # st.dataframe(churn_counts, hide_index=True)

    # TRUCK TABLE #
    ## retrieve truck table
    truck_table_df = retrieve_truck_table()
    
    ## Display header
    st.markdown("## Truck Table")

    ## Display the merged DataFrame
    st.dataframe(truck_table_df, width=0, hide_index=True)  
    
    
    # # ORDER HEADER TABLE #
    # ## retrieve order_header table
    # order_header_table_df = retrieve_order_header_table()

    # ## Display header
    # st.markdown("## Order Header Table")

    # ## Display the merged DataFrame
    # st.dataframe(order_header_table_df, width=0, hide_index=True)  
    
    
    order_details_df = retrieve_order_details()
    st.dataframe(order_details_df, width=0, hide_index=True)
    
    
    # OVERALL TRUCK TABLE #
    ## retrieve overall_truck_df table
    overall_truck_df_grouped = get_overall_truck_table(truck_table_df, order_details_df)

    ## Display header
    st.markdown("## Overall Truck Table")

    ## Display the merged DataFrame
    st.dataframe(overall_truck_df_grouped, width=0, hide_index=True) 
    
    # Page Instructions (Why we should identity low value food trucks)
    with st.expander("Why we should identity low value food trucks"):
        # List of steps
        st.write('Cutting back on low-sales food trucks aligns with Tasty Bytes goal of 25% YoY sales growth. By identifying and reducing underperforming trucks, they optimize resources, focus on high-potential markets, and boost profitability. Eliminating low-performing units improves resource allocation, maximizes ROI, and streamlines operations. It enables a data-driven approach to identify market trends, customer preferences, and expansion opportunities, resulting in increased sales and customer satisfaction')
    
    # MENU TABLE #
    ## retrieve menu table
    menu_table_df = retrieve_menu_table()

    ## Display header
    st.markdown("## Additional Information")

    ## Display the merged DataFrame
    st.dataframe(menu_table_df, width=0, hide_index=True)  
    
    
    # # OVERALL TABLE #
    # ## retrieve the merged table from truck and menu
    # merged_df = get_overall_table(truck_table_df, menu_table_df)

    # ## Display header
    # st.markdown("## Overall Table")

    # ## Display the merged DataFrame
    # st.dataframe(merged_df, width=0, hide_index=True)
    

    
with tab2:
    st.markdown("## Cluster")
    st.markdown("What is clustering? <br> Clustering is the task of dividing the population \
                or data points into a number of groups such that data") 
    
    st.markdown("## Clustering variables")