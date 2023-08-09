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
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

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



# # Function: retrieve order_header table
# # the purpose of this function is to retrieve the order_header table from snowflake containing all the details of the order_header items
# def retrieve_order_header():
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
    
#     # Retrieve the list of customer IDs from the 'data' table
#     customer_ids = data['CUSTOMER_ID'].tolist()

#     # Split the list into smaller chunks of 1,000 customer IDs
#     chunk_size = 1000
#     customer_id_chunks = [customer_ids[i:i+chunk_size] for i in range(0, len(customer_ids), chunk_size)]

#     # Execute queries for each customer ID chunk
#     order_details = []
#     for chunk in customer_id_chunks:
#         # Create a comma-separated string of the customer IDs in the current chunk
#         customer_ids_str = ','.join(map(str, chunk))

#         # Construct the SQL query for the current chunk
#         query = f"SELECT TRUCK_ID, LOCATION_ID, CUSTOMER_ID, ORDER_AMOUNT, ORDER_TOTAL, ORDER_CURRENCY from order_header WHERE CUSTOMER_ID IN ({customer_ids_str})"

#         # Execute the SQL query for the current chunk
#         my_cur.execute(query)

#         # Fetch the result for the current chunk
#         chunk_result = my_cur.fetchall()

#         # Append the chunk result to the overall result
#         order_details.extend(chunk_result)

#     # Create a DataFrame from the fetched result
#     order_header_df = pd.DataFrame(order_details, columns=['TRUCK_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'ORDER_AMOUNT', 'ORDER_TOTAL', 'ORDER_CURRENCY'])

#     # Convert ORDER_ID to string and then remove commas
#     #order_details_df['ORDER_ID'] = order_details_df['ORDER_ID'].astype(str).str.replace(',', '')

#     # Format ORDER_TOTAL and PRODUCT_TOTAL_PRICE columns to 2 decimal places
#     order_header_df['ORDER_TOTAL'] = order_header_df['ORDER_TOTAL'].apply(lambda x: '{:.2f}'.format(x))

#     order_header_df = order_header_df.sort_values(by='CUSTOMER_ID')

#     return order_header_df


# Function: retrieve_order_header_table()
# the purpose of this function is to retrieve the header details for USA from Snowflake to merge with the menu column to get total sales for a current menu type
def retrieve_order_header_table():
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
    my_cur.execute("select TRUCK_ID, ORDER_TS, ORDER_TOTAL, ORDER_CURRENCY from ORDER_HEADER where ORDER_CURRENCY = 'USD'")
    order_table = my_cur.fetchall()
    
    order_table_df = pd.DataFrame(order_table, columns=['TRUCK_ID', 'ORDER_TS', 'ORDER_TOTAL', 'ORDER_CURRENCY'])
    
    return order_table_df


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


# Function: get_overall_truck_sales_table
# the purpose of this function is to merge the truck and menu details table together to form an overall table
def get_overall_truck_sales_table(truck_table_df, order_header_df):
    ## Merge the DataFrames based on 'MENU_ITEM_ID'
    overall_truck_sales_df = pd.merge(truck_table_df, order_header_df, on='TRUCK_ID', how='left')

    ## Define the desired column order
    desired_columns = ['COUNTRY', 'REGION','PRIMARY_CITY', 'TRUCK_ID', 'CUSTOMER_ID', 'ORDER_TOTAL']

    ## Re-arrange the columns in the merged DataFrame
    overall_truck_sales_df = overall_truck_sales_df[desired_columns]
    
    ## Cast 'ORDER_TOTAL' to float
    overall_truck_sales_df['ORDER_TOTAL'] = overall_truck_sales_df['ORDER_TOTAL'].astype(float)
    
    ## Group by 'TRUCK_ID' and combine 'ORDER_TOTAL' for each truck
    overall_truck_sales_df_grouped = overall_truck_sales_df.groupby('TRUCK_ID').agg({
        'ORDER_TOTAL': 'sum'           # Sum the 'ORDER_TOTAL' amounts for each truck_id
    }).reset_index()
    
    return overall_truck_sales_df_grouped

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
    
    # retreive truck table
    truck_table = retrieve_truck_table()
    
    # ## Option: primary city name
    # ## add None as the default value (it won't be an actual selectable option)
    # default_option = None
    # primary_city_options = np.sort(truck_table['PRIMARY_CITY'].unique())

    # ## use the updated list of options for the selectbox
    # selected_primary_city_name = st.selectbox("City Selected: ", [default_option] + list(primary_city_options))

    # # Filter the truck_table to find the truck id for the selected trucks in that city
    # truck_filter = truck_table['PRIMARY_CITY'] == selected_primary_city_name
    # if truck_filter.any():
    #     selected_truck = truck_table.loc[truck_filter, 'TRUCK_ID'].values[0]
    # else:
    #     selected_truck = None

    # ## Option: truck id
    # ## add None as the default value (it won't be an actual selectable option)
    # default_option = None
    # truck_id_options = np.sort(truck_table['TRUCK_ID'].unique())

    # # ## use the updated list of options for the selectbox
    # selected_truck_id = st.selectbox("Truck Id: ", [default_option] + list(truck_id_options))


    
    ## Option: primary city name
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    primary_city_options = np.sort(truck_table['PRIMARY_CITY'].unique())

    ## use the updated list of options for the selectbox
    selected_primary_city_name = st.selectbox("City Selected: ", [default_option] + list(primary_city_options))

    # Filter the truck_table to find the truck ids for the selected city
    if selected_primary_city_name is not None:
        truck_filter = truck_table['PRIMARY_CITY'] == selected_primary_city_name
        selected_truck_ids = truck_table.loc[truck_filter, 'TRUCK_ID'].values
    else:
        selected_truck_ids = []

    ## Option: truck id
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None

    ## use the updated list of options for the selectbox
    selected_truck_id = st.selectbox("Truck Id: ", [default_option] + list(selected_truck_ids))

    
    # user inputs (year n month can be hardcoded as data is static)
    user_input_full = {
        #"TRUCK": selected_truck,
        "PRIMARY_CITY": selected_primary_city_name,
        "TRUCK_ID": selected_truck_id, 
        'YEAR': 2022,
        'MONTH': 12
    }
    
    # # Create a dictionary with the current year and month
    # data = {
    #     'YEAR': current_year,
    #     'MONTH': current_month
    # }

    # # Convert the dictionary to a DataFrame
    # current_date_df = pd.DataFrame(data, index=[1])

    # create dataframe with all the user's inputs
    user_input_df = pd.DataFrame(user_input_full, index=[0])

    return user_input_df

# Function: prediction()
# the purpose of this function is to carry out certain data transformations and create the 2 tables shown after prediction
def prediction(user_input_df):
    # replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    user_input_df = user_input_df.replace({"Yes": 1, "No":0})
    
    # retreive truck table
    truck_table = retrieve_truck_table()
    
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

    desired_order = ['TRUCK_ID', 'YEAR', 'MONTH', 
                'PRIMARY_CITY_Seattle', 'PRIMARY_CITY_New York City', 'PRIMARY_CITY_Denver',
                'PRIMARY_CITY_San Mateo']

    user_input_df = user_input_df.reindex(columns=desired_order)
    
    # retrieve min max scaler
    min_max_scaler = joblib.load("assets/truck_min_max_scaler.joblib")
    
    user_input_df = min_max_scaler.transform(user_input_df)
    
    # retrieve regression model
    truck_sales_model = joblib.load("assets/models/truck_xgb_improved.joblib")
    
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
tab1, tab2, tab3 = st.tabs(['About', 'Model Prediction', 'TEST'])

# Explore page 
with tab1:
    # High level goals
    st.markdown("# High Level Goals")
    st.markdown("This Streamlit page plays a crucial role in contributing to Tasty Bytes' goal of achieving a 25% YoY sales increase. \
                By leveraging the power of data analysis and machine learning, the page predicts next month's sales for each food truck based on historical data. \
                This accurate forecasting enables Tasty Bytes to make informed decisions and implement targeted strategies to optimize sales for individual trucks and menu items. \
                \n\n With a clear understanding of upcoming sales trends, Tasty Bytes can proactively adjust inventory, marketing efforts, and operational aspects, maximizing revenue potential. \
                As a result, this page empowers Tasty Bytes to make data-driven decisions, improve overall sales performance, and work towards their ambitious high-level sales target.") 
    
    # Benefits of the sales predictions
    st.markdown("## Benefits of this Prediction")
    st.markdown("Tasty Bytes can utilize the predictions of next month's sales for each truck ID to strategically boost their sales and achieve their growth target. \
                With accurate foresight into future sales trends, Tasty Bytes can optimize their inventory management, ensuring that popular menu items are well-stocked, \
                reducing waste, and minimizing stockouts. \n\nMoreover, they can tailor their marketing efforts, focusing on promoting specific menu items or trucks that are \
                projected to perform exceptionally well. By allocating resources efficiently and proactively adapting their operations, Tasty Bytes can provide a more \
                personalized and customer-centric experience, fostering customer loyalty and attracting new patrons.")
    st.write(
            """
            Using this prediction, you can implement the following strategies:

            - Demand Forecasting: By analyzing historical sales data, the company can identify patterns and trends in customer behavior. This allows you to anticipate fluctuations in demand, helping you be better prepared \
                for busy periods and ensuring you can meet customer expectations. On the other hand, during slower times, you can adjust your operations to reduce costs while maintaining service quality.

            - Sales Target Setting: Incorporating the predictions into sales target setting will enable Tasty Bytes to set realistic and achievable goals for each truck. This will ensure that performance expectations are aligned with market trends and growth potentials.

            - Real-time Monitoring: Integrating the predictions into a real-time monitoring system will enable Tasty Bytes to stay agile and respond promptly to changing sales trends, ensuring optimal performance.
            
            By adopting these strategies and actively utilizing the sales predictions, Tasty Bytes can unlock new growth opportunities, optimize their operations, and stay ahead in the competitive market, ultimately propelling them towards their ambitious 25% YoY sales increase goal.
            """
        )
    
    # Limitations
    st.markdown("## Limitations of the Model")
    st.markdown("While my predictive model offers valuable insights, it has some limitations. It doesn't account for seasonal variations in sales, leading to inaccurate resource allocation and inventory management during specific periods. \
                Additionally, the model might not account for extreme weather events that can disrupt customer behavior and sales. It relies solely on historical data and internal factors, overlooking external variables like economic conditions or competitor activities.")
    
    
# Prediction and Analysis
with tab2:
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

    with st.expander("Cleaned and Transformed Data"):
        st.write(df)
    
    
    ########## USER INPUT ###############
    # Monthly Truck Sales Prediction
    st.markdown("## Monthly Truck Sales")
    
    user_input_df = user_inputs()
        
    # display dataframe
    st.dataframe(user_input_df, hide_index=True)


    # Check for null values in the user_input_df
    has_null_values = user_input_df.isnull().any().any()

    if has_null_values == False:
        # display message if no null values are found
        st.write("Proceed to make a prediction!")
        
        # Make a prediction
        if st.button("Predict"):
            final_prediction = prediction(user_input_df)
            final_prediction = int(final_prediction)
            
            # Get estimated sales prediction for next year
            sales_next_year = final_prediction * 12
            
            ## display the rounded prediction
            st.markdown("### Estimated sales next month: ${:.2f}".format(final_prediction))
            #st.markdown("### Estimated sales next year: ${:.2f}".format(sales_next_year))
            st.markdown("### Estimated sales next year: ${:.2f} million".format(sales_next_year / 10**6))

            
            
    else:
        st.error("Please fill in all required fields before proceeding with the prediction.")
    
    # Benefits
    st.markdown("## How it helps Tasty Bytes towards its high level goal of 25% YoY Sales")
    st.markdown("My ultimate goal in predicting future sales of food trucks is to increase Tasty Bytes sales by 5 percent, \
                contributing significantly to achieving a remarkable 25% YoY sales increase.  \
                By accurately forecasting sales for each food truck, \
                Tasty Bytes can optimize their inventory, and streamline their operations.")
    




    ############# TESTING ###########
with tab3:
    st.markdown("## Next Year Sales Prediction by Truck")
    history_data = pd.read_csv("assets/datasets/truck_last_month_sales.csv")
    st.write("This is last months sales")
    st.write(history_data)

    # retreive truck table
    truck_table = retrieve_truck_table()
        
        
    # # set default option to none   
    default_opt = None

    # get truck id options for users to choose
    truck_id_opt = [
    f"{row['TRUCK_ID']}"
    for _, row in truck_table.iterrows()
    ]

    ## use the updated list of options for the selectbox
    selected_truck = st.selectbox("Truck Id: ", [default_opt] + list(truck_id_opt), key="unique_truck_selector")

    if selected_truck == None:
        st.error("Please fill in the required field to get a prediction")
    
    else:
        # extract TRUCK_ID from the option string
        truck_id_selection = int(selected_truck)
        
        # retrieve regression model
        model2 = joblib.load("assets/models/truck_xgb_improved.joblib")
        input_data = pd.DataFrame()
        input_data = truck_table[truck_table["TRUCK_ID"] == truck_id_selection]
        input_data = input_data.drop(["PRIMARY_CITY", "REGION", "COUNTRY", "FRANCHISE_ID", "MENU_TYPE_ID"], axis=1)
        input_data["YEAR"] = 2022
        input_data["MONTH"] = 12
        
        st.markdown("### Based on month initial information")
        # data to input in the model
        st.write(input_data)

        pred_sales = model2.predict(input_data)
        
        st.metric("Predicted Sales", f"${pred_sales[0]:.2f}")

        
        
    #     item_info_df = truck_table[truck_table["TRUCK_ID"] == truck_id_selection]
        
        
    #     # retrieve year from order timestamp
    #     order_header_df = pd.read_csv('assets/total_sales_by_truck.csv')
        
    #      # Convert the 'YEAR' column to numeric values
    #     order_header_df['YEAR'] = order_header_df['YEAR'].astype(str).replace(',', '').astype(int)
        
        
    #     # get the total sales by truck over the years
    #     total_sales_by_truck_over_time = order_header_df[order_header_df["TRUCK_ID"]==truck_id_selection]
        
        
    #     # Plotly Line Chart
    #     ## create the line chart
    #     fig = go.Figure(data=go.Line(x=total_sales_by_truck_over_time['YEAR'], y=total_sales_by_truck_over_time['TOTAL_SALES_PER_YEAR'], mode='lines+markers'))

    #     ## update the layout
    #     fig.update_layout(title='Total Sales by Truck',
    #                     xaxis_title='Year',
    #                     yaxis_title='Total Sales')



    #     # get one year after the latest year provided in the data
    #     year = total_sales_by_truck_over_time["YEAR"].max() + 1
        
        
    # #     # order_header_df = retrieve_order_header_table()
    # #     # order_header_df['YEAR'] = order_header_df['ORDER_TS'].dt.year
    # #     # order_header_df['MONTH'] = order_header_df['ORDER_TS'].dt.month
        
    # #     # # Group order total to truck id
    # #     # SUM_SALES_CITY = order_header_df.groupby(['YEAR', 'MONTH', 'TRUCK_ID'])['ORDER_TOTAL'].sum().reset_index()

    # #     # # Renaming the 'ORDER_TOTAL' column to 'TOTAL_SALES_PER_MONTH'
    # #     # SUM_SALES_CITY = SUM_SALES_CITY.rename(columns={'ORDER_TOTAL': 'TOTAL_SALES_PER_MONTH'})

    # #     # # Convert the 'YEAR' column to numeric values
    # #     # SUM_SALES_CITY['YEAR'] = SUM_SALES_CITY['YEAR'].astype(str).replace(',', '').astype(int)
        
        
    # #     # # get the highest year and month
    # #     # max_year_month = SUM_SALES_CITY.groupby('TRUCK_ID')[['YEAR', 'MONTH']].max().reset_index()

    # #     # truck_max_year_month = max_year_month[max_year_month["TRUCK_ID"]==truck_id]

    # #     # total_sales_by_truck = SUM_SALES_CITY[SUM_SALES_CITY["TRUCK_ID"]==truck_id]
        
    # #     # # Plotly Line Chart
    # #     # ## create the line chart
    # #     # fig = go.Figure(data=go.Line(x=total_sales_by_truck['MONTH'], y=total_sales_by_truck['TOTAL_SALES_PER_MONTH'], mode='lines+markers'))

    # #     # ## update the layout
    # #     # fig.update_layout(title='Monthly Sales by Truck',
    # #     #                 xaxis_title='Month',
    # #     #                 yaxis_title='Total Sales')

    # #     # ## show the plot in the Streamlit app 
    # #     # st.plotly_chart(fig)


    # #     # # Form month and year column for prediction
    # #     # ## if month is less than or equal to 11 then plus 1
    # #     # if int(truck_max_year_month["MONTH"])<=11:
    # #     #     month = int(truck_max_year_month["MONTH"]) + 1
    # #     #     year = int(truck_max_year_month["YEAR"])
    # #     # ## if month is equal to 12 then month will be 1 and year plus 1
    # #     # elif int(truck_max_year_month["MONTH"])== 12:
    # #     #     month = 1
    # #     #     year = int(truck_max_year_month["YEAR"]) + 1
        
        
        
    #     # Replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    #     item_info_df = item_info_df.replace({'Yes': 1, 'No': 0})
        
    #     ###############################################################################
        
    #     # MANUAL ENCODING
    #     categorical_cols = ["PRIMARY_CITY"]
        
    #     # Loop through each categorical column
    #     for col in categorical_cols:
    #         # Get the unique values in the column
    #         unique_values = truck_table[col].unique()

    #         # Loop through unique values in the column
    #         for value in unique_values:
    #             # Check if the value in the truck_table table matches the corresponding value in user_input_df
    #             if value == item_info_df[col].values[0]:
    #                 # Create a column with the name 'column_selected_value' and set its value to 1
    #                 truck_table[f'{col}_{value}'] = 1

    #                 # Add this column to the item_info_df
    #                 item_info_df[f'{col}_{value}'] = 1
    #             else:
    #                 # Create a column with the name 'column_unique_value' and set its value to 0
    #                 truck_table[f'{col}_{value}'] = 0

    #                 # Add this column to the item_info_df
    #                 item_info_df[f'{col}_{value}'] = 0


    #     # Drop the original categorical columns from user_input_df
    #     item_info_df.drop(columns=categorical_cols, inplace=True)

    #     ## assign the columns YEAR with their respective values
    #     item_info_df['YEAR'] = year
        
        
    #     desired_order = ['TRUCK_ID', 'YEAR', 
    #                 'PRIMARY_CITY_Denver', 'PRIMARY_CITY_San Mateo', 'PRIMARY_CITY_Boston',
    #                 'PRIMARY_CITY_New York City']
        
    #     # drop columns not in the desired column list
    #     item_info_df = item_info_df[desired_order]

    #     item_info_df = item_info_df.reindex(columns=desired_order)
            

        
        
    #     # retrieve min max scaler
    #     min_max_scaler = joblib.load("assets/truck_min_max_scaler.joblib")
        
    #     min_max_scaler.fit(item_info_df)
        
    #     min_max_scaler.transform(item_info_df)
        
        
    #     # retrieve regression model
    #     truck_sales_per_year_model = joblib.load("assets/truck_xgb_improved.joblib")
        
    #     model_prediction = truck_sales_per_year_model.predict(item_info_df)
        
    #     # Assuming model_prediction is a numpy ndarray with only one element
    #     #model_prediction = model_prediction.item()
        
    #     sales_next_year = float(model_prediction)
        
        
        
    #     # # Replace 'SELECTED_TRUCK_ID' with the ID of the specific food truck you want to predict for
    #     # selected_truck_id = truck_id

    #     # # Filter item_info_df for the selected truck ID
    #     # selected_truck_df = item_info_df[item_info_df['TRUCK_ID'] == selected_truck_id]

    #     # # Predict sales for the selected truck using the model
    #     # model_prediction = truck_sales_per_month_model.predict(selected_truck_df)

    #     # # Assuming model_prediction is a numpy ndarray with only one element
    #     # model_prediction = model_prediction.item()

    #     # # Convert the prediction to a float
    #     # sales_prediction = float(model_prediction)
                
        
        
    #     # # Round off the prediction to the nearest whole number
    #     # rounded_prediction = round(model_prediction[0])
        
    #     # unit_price = menu_table.loc[menu_table['MENU_ITEM_ID'] == menu_item_id, 'UNIT_PRICE'].values[0]
    #     # sales_next_month = float(unit_price) * int(rounded_prediction)
        
        
        
    #     ############# Input data ##############
    #     # # retrieve regression model
    #     # model2 = joblib.load("assets/truck_xgb_improved.joblib")
    #     # input_data = pd.DataFrame()
    #     # input_data["TRUCK_ID"] = selected_truck
    #     # input_data["YEAR"] = 2022
    #     # input_data["MONTH"] = 10
    #     # pred_sales = model2.predict(input_data)
        
    #     # st.markdown("### Based on month initial information")
    #     # # data to input in the model
    #     # st.write(input_data)

    #     # pred_sales = model2.predict(input_data)
    #     # st.metric("Predicted Sales", f"${pred_sales[0]:.2f}")
    #     #############################################################

    #     # Get previous month sales
    #     ## sort the DataFrame by 'Year' in descending order
    #     total_sales_by_truck_sorted = total_sales_by_truck_over_time.sort_values(by='YEAR', ascending=False)

    #     ## keep only the first row for each 'TRUCK_ID' which is the latest
    #     total_sales_by_truck_over_time = total_sales_by_truck_sorted.groupby('TRUCK_ID').first().reset_index()
    
    #     ## get the total sales for the latest year
    #     sales_last_year = float(total_sales_by_truck_over_time["TOTAL_SALES_PER_YEAR"])

    #     # chnage in sales by year
    #     sales_change = (float(model_prediction)) - sales_last_year
        
    #     percent_change = ((sales_change / sales_last_year)*100)

    #     # DISPLAY
    #     st.markdown("## Prediction:")
        
    #     # show the plot in the Streamlit app 
    #     st.plotly_chart(fig)
        
    #     st.markdown("### Estimated sales next year: ${:.2f}".format(sales_next_year))
    #     st.markdown("### Percentage change from last year: {:.2f}%".format(percent_change))

    #     st.dataframe(item_info_df)
        
    #     # show historical qty sold over the years
    #     total_sales_by_truck_sorted = total_sales_by_truck_sorted.sort_values(by='YEAR', ascending=True)
        
    #     # convert the 'YEAR' column to string to remove ','
    #     total_sales_by_truck_sorted['YEAR'] = total_sales_by_truck_sorted['YEAR'].astype(str).replace(',', '').astype(str)

    #     # display table
    #     st.dataframe(total_sales_by_truck_sorted, hide_index=True)