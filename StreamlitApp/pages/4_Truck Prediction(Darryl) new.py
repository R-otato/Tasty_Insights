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

# Function: retrieve location table
# the purpose of this function is to retrieve the location table from snowflake containing all the details of the location items
def retrieve_location_table():
    # RETRIEVE LOCATION TABLE FROM SNOWFLAKE
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

    ## retrieve location table from snowflake
    my_cur = my_cnx.cursor()
    # select united states
    my_cur.execute("select LOCATION_ID, LOCATION, CITY, COUNTRY from location where COUNTRY = 'United States'")
    location_table = my_cur.fetchall()
    
    ## create a DataFrame from the fetched result
    location_table_df = pd.DataFrame(location_table, columns=['LOCATION_ID', 'LOCATION', 'CITY', 'COUNTRY'])

    return location_table_df



# Function: retrieve order_header table
# the purpose of this function is to retrieve the order_header table from snowflake containing all the details of the order_header items
def retrieve_order_header():
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
        query = f"SELECT TRUCK_ID, LOCATION_ID, CUSTOMER_ID, ORDER_AMOUNT, ORDER_TOTAL, ORDER_CURRENCY from order_header WHERE CUSTOMER_ID IN ({customer_ids_str})"

        # Execute the SQL query for the current chunk
        my_cur.execute(query)

        # Fetch the result for the current chunk
        chunk_result = my_cur.fetchall()

        # Append the chunk result to the overall result
        order_details.extend(chunk_result)

    # Create a DataFrame from the fetched result
    order_header_df = pd.DataFrame(order_details, columns=['TRUCK_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'ORDER_AMOUNT', 'ORDER_TOTAL', 'ORDER_CURRENCY'])

    # Convert ORDER_ID to string and then remove commas
    #order_details_df['ORDER_ID'] = order_details_df['ORDER_ID'].astype(str).str.replace(',', '')

    # Format ORDER_TOTAL and PRODUCT_TOTAL_PRICE columns to 2 decimal places
    order_header_df['ORDER_TOTAL'] = order_header_df['ORDER_TOTAL'].apply(lambda x: '{:.2f}'.format(x))

    order_header_df = order_header_df.sort_values(by='CUSTOMER_ID')

    return order_header_df


# Function: get_overall_truck_table
# the purpose of this function is to merge the location and order header table together to form an overall table
def get_overall_truck_table(location_table_df, order_header_df):
    ## Merge the DataFrames based on 'MENU_ITEM_ID'
    overall_truck_df = pd.merge(location_table_df, order_header_df, on='LOCATION_ID', how='left')

    ## Define the desired column order
    desired_columns = ['COUNTRY', 'CITY','LOCATION', 'ORDER_TOTAL', 'TRUCK_ID', 'CUSTOMER_ID']

    ## Re-arrange the columns in the merged DataFrame
    overall_truck_df = overall_truck_df[desired_columns]
    
    ## Cast 'ORDER_TOTAL' to float
    overall_truck_df['ORDER_TOTAL'] = overall_truck_df['ORDER_TOTAL'].astype(float)
    
    ## Group by 'LOCATION' and combine 'ORDER_TOTAL' for each truck
    overall_truck_df_grouped = overall_truck_df.groupby('LOCATION').agg({
        'COUNTRY': 'first',            
        'CITY': 'first',               
        'CUSTOMER_ID': 'nunique',      # Count unique customers for each truck_id
        'ORDER_TOTAL': 'sum'           # Sum the 'ORDER_TOTAL' amounts for each location
    }).reset_index()
    
    # Change the column name to "Number of Customers"
    overall_truck_df_grouped = overall_truck_df_grouped.rename(columns={'CUSTOMER_ID': 'Number of Customers'})

    # Filter locations where the order total is more than 50000
    overall_truck_df_grouped = overall_truck_df_grouped[overall_truck_df_grouped['ORDER_TOTAL'] > 50000]

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


# Function: user_inputs()
# the purpose of this function is to get the user's input for the sales of a truck by location they would like to get a prediction for
def user_inputs(): 
    
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
    truck_table = pd.DataFrame(truck_table, columns=['TRUCK_ID', 'PRIMARY_CITY', 'REGION', 'COUNTRY', 'FRANCHISE_ID', 'MENU_TYPE_ID'])
    
    ## Option: primary city name
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    primary_city_options = np.sort(truck_table['PRIMARY_CITY'].unique())

    ## use the updated list of options for the selectbox
    selected_primary_city_name = st.selectbox("City Selected: ", [default_option] + list(primary_city_options))

    # Filter the truck_table to find the truck id for the selected trucks in that city
    truck_filter = truck_table['PRIMARY_CITY'] == selected_primary_city_name
    if truck_filter.any():
        selected_truck = truck_table.loc[truck_filter, 'TRUCK_ID'].values[0]
    else:
        selected_truck = None

    ## Option: truck id
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    truck_id_options = np.sort(truck_table['TRUCK_ID'].unique())

    ## use the updated list of options for the selectbox
    selected_truck_id = st.selectbox("Truck Id: ", [default_option] + list(truck_id_options))


    ## Option: menu type id
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    menu_type_options = np.sort(truck_table['MENU_TYPE_ID'].unique())

    ## use the updated list of options for the selectbox
    selected_menu_type = st.selectbox("Menu Type: ", [default_option] + list(menu_type_options))

    # Use current date for model predictions
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month

    user_input_full = {
        #"TRUCK": selected_truck,
        "PRIMARY_CITY": selected_primary_city_name,
        "TRUCK_ID": selected_truck_id, 
        "MENU_TYPE_ID": selected_menu_type,
        'YEAR': current_year,
        'MONTH': current_month
    }
    
    # # Create a dictionary with the current year and month
    # data = {
    #     'YEAR': current_year,
    #     'MONTH': current_month
    # }

    # # Convert the dictionary to a DataFrame
    # current_date_df = pd.DataFrame(data, index=[1])

    # create dataframe with all the user's inputs
    user_input_df = pd.DataFrame(user_input_full, index=[1])

    return user_input_df

# Function: prediction()
# the purpose of this function is to carry out certain data transformations and create the 2 tables shown after prediction
def prediction(user_input_df):
    # replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    user_input_df = user_input_df.replace({"Yes": 1, "No":0})
    
    # # MANUAL ENCODING
    # categorical_cols = ["PRIMARY_CITY"]
    
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
    truck_table = pd.DataFrame(truck_table, columns=['TRUCK_ID', 'PRIMARY_CITY', 'REGION', 'COUNTRY', 'FRANCHISE_ID', 'MENU_TYPE_ID'])

    # MANUAL ENCODING
    categorical_cols = ["PRIMARY_CITY"]
    
    # Loop through each categorical column
    for col in categorical_cols:
        # Get the unique values in the column
        unique_values = truck_table[col].unique()

        # Loop through unique values in the column
        for value in unique_values:
            # Check if the value in the truck_table table matches the corresponding value in user_input_df
            if value == user_input_df[col].values[0]:
                # Create a column with the name 'column_selected_value' and set its value to 1
                truck_table[f'{col}_{value}'] = 1

                # Add this column to the user_input_df
                user_input_df[f'{col}_{value}'] = 1
            else:
                # Create a column with the name 'column_unique_value' and set its value to 0
                truck_table[f'{col}_{value}'] = 0

                # Add this column to the user_input_df
                user_input_df[f'{col}_{value}'] = 0


    # Drop the original categorical columns from user_input_df
    user_input_df.drop(columns=categorical_cols, inplace=True)

    #user_input_df.drop(columns=["ITEM_SUBCATEGORY_Hot Option", "MENU_TYPE_Sandwiches", "TRUCK_BRAND_NAME_Better Off Bread", "ITEM_CATEGORY_Dessert"], inplace = True)

    desired_order = ['YEAR', 'MONTH', 'TRUCK_ID', 'MENU_TYPE_ID', 
                'PRIMARY_CITY_Boston', 'PRIMARY_CITY_Seattle', 'PRIMARY_CITY_Denver',
                'PRIMARY_CITY_New York City']

    user_input_df = user_input_df.reindex(columns=desired_order)
    
    
    # retrieve min max scaler
    min_max_scaler = joblib.load("assets/truck_min_max_scaler.joblib")
    
    min_max_scaler.transform(user_input_df)
    
    # retrieve regression model
    truck_sales_model = joblib.load("assets/truck_xgb_improved.joblib")
    
    final_prediction = truck_sales_model.predict(user_input_df)
    
    # # Round off the prediction to the nearest whole number
    # rounded_prediction = round(prediction[0])

    
    return final_prediction


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
    

    # LOCATION TABLE #
    ## retrieve location table
    location_table_df = retrieve_location_table()
    
    ## Display header
    st.markdown("## Location Table")

    ## Display the merged DataFrame
    st.dataframe(location_table_df, width=0, hide_index=True)  

    
    ########## TESTING USER INPUT ###############
    # Monthly Truck Sales Prediction
    st.markdown("## Monthly Truck Sales")
    
    user_input_df = user_inputs()
        
    # display dataframe
    st.dataframe(user_input_df, hide_index=True)


    # Check for null values in the user_input_df
    has_null_values = user_input_df.isnull().any().any()

    if has_null_values == False:
        # display message if no null values are found
        st.write("Proceed to make a prediction.")
        
        # Make a prediction
        if st.button("Predict"):
            final_prediction = prediction(user_input_df)
            
            st.markdown("### Prediction")
            ## display the rounded prediction
            st.markdown("##### Predicted Sales for Next Month: {}".format(final_prediction))
            
            st.write('')
            
            # st.markdown("##### Total Item Details:")
            # ## display the total_product_details_df DataFrame
            # st.dataframe(total_product_details_df, hide_index=True)
            
            # # display current menu items
            # with st.expander("Unit Item Details"):
            #     st.write("This table contains details specific to a single unit or item of the new product")
            #     ## display the new_product_details_df DataFrame
            #     st.dataframe(new_product_details_df, hide_index=True)
    else:
        st.error("Please fill in all required fields before proceeding with the prediction.")
    ###########################################################
    
    # ORDER HEADER TABLE #
    ## retrieve order_header table
    order_details_df = retrieve_order_header()
    
    # ## Display header
    # st.markdown("## Order Header Table")
    
    # ## Display the merged DataFrame
    # st.dataframe(order_details_df, width=0, hide_index=True)
    
    
    # # OVERALL TRUCK TABLE #
    # ## retrieve overall_truck_df table
    # overall_truck_df_grouped = get_overall_truck_table(location_table_df, order_details_df)

    # ## Display header
    # st.markdown("## Total Sales by Location")

    # ## Display the merged DataFrame
    # #st.dataframe(overall_truck_df_grouped, width=0, hide_index=True) 
    

    
with tab2:
    st.markdown("## Cluster")
    st.markdown("What is clustering? <br> Clustering is the task of dividing the population \
                or data points into a number of groups such that data") 
    
    st.markdown("## Clustering variables")