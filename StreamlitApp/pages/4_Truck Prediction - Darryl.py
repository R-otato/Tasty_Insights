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
tab1, tab2 = st.tabs(['About', 'Model Prediction'])

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
            
            # Assuming user_input_df['MONTH'] contains the selected month as an integer (e.g., 1 for January, 2 for February, etc.)
            selected_month = user_input_df['MONTH'].iloc[0]

            # Calculate previous month
            if selected_month == 1:  # If January, set previous_month to December
                previous_month = 12
            else:
                previous_month = selected_month - 1
                
            # get monthly sales
            history_data = pd.read_csv("assets/datasets/truck_last_month_sales.csv")
            
            # Filter history_data to get sales data for the previous month, selected city, and truck ID
            previous_month_sales_data = history_data[
                (history_data['MONTH'] == previous_month) &
                (history_data['TRUCK_ID'] == user_input_df['TRUCK_ID'].iloc[0])  # Replace with the selected truck ID
            ]

            # display dataframe
            sales_last_month = previous_month_sales_data.iloc[:, 3]
            # this is because last order TS is only pon the 1st on November (Hence, we have to get the estimated sales throughout the whole month)
            sales_last_month = float(sales_last_month) * 30
            
            # chnage in sales by month
            sales_change = (float(final_prediction)) - sales_last_month
            sales_change = float(sales_change)
            
            month_percent_change = ((sales_change / sales_last_month)*100)
            
            
            
            # get yearly sales
            history_data_year = pd.read_csv("assets/datasets/total_sales_by_truck_city.csv")
            
            selected_year = user_input_df['YEAR'].iloc[0]
            
            # Filter history_data to get sales data for the previous month, selected city, and truck ID
            previous_year_sales_data = history_data_year[
                (history_data_year['YEAR'] == selected_year) &
                (history_data_year['TRUCK_ID'] == user_input_df['TRUCK_ID'].iloc[0])  # Replace with the selected truck ID
            ]

            # display dataframe
            sales_last_year = previous_year_sales_data.iloc[:, 3]
            # convert to float type
            sales_last_year = float(sales_last_year)
            
            # chnage in sales by year
            sales_change = (float(sales_next_year)) - sales_last_year
            sales_change = float(sales_change)
            
            year_percent_change = ((sales_change / sales_last_year)*100)
            
            
            
            
            # Display metrics
            col1,col2=st.columns(2)
            
            col1.metric('Estimated sales next month', f"${format(round(final_prediction, 2), ',')}", help="This is the estimated sales predicted for next month")
            col2.metric('Estimated sales next year', f"${format(round(sales_next_year, 2), ',')}", help="This is the estimated sales predicted for next year")
            
            # col1.metric('Estimated sales next month', f"${format(round(final_prediction, 2), ',')}", f"${round(final_prediction, 2)-sales_last_month}", delta_color="normal", help="This is the estimated sales predicted for next month")
            # col2.metric('Estimated sales next year', f"${format(round(sales_next_year, 2), ',')}", f"${round(sales_next_year, 2)-sales_last_year}", delta_color="normal", help="This is the estimated sales predicted for next year")
            
            # Display percentage change for each month
            ## Check if more than 0, no change, or less than 0 percentage increase
            if month_percent_change > 0:
                col1.metric('Estimated MOM sales growth', f"↑ {format(round(month_percent_change, 2), ',')}%")
            elif month_percent_change == 0:
                col1.metric('Estimated MOM sales growth', f"↔ {format(round(month_percent_change, 2), ',')}%")
            else:
                col1.metric('Estimated MOM sales growth', f"↓ {format(round(month_percent_change, 2), ',')}%")
            
            
            # Display percentage change for each year
            
            ## Check if more than 0, no change, or less than 0 percentage increase
            if year_percent_change > 0:
                col2.metric('Estimated YoY sales growth', f"↑ {format(round(year_percent_change, 2), ',')}%")
            elif year_percent_change == 0:
                col2.metric('Estimated YoY sales growth', f"↔ {format(round(year_percent_change, 2), ',')}%")
            else:
                col2.metric('Estimated YoY sales growth', f"↓ {format(round(year_percent_change, 2), ',')}%")
            
            
            # Summary
            if year_percent_change > 0:
                st.markdown("""Based on your selections, we have identified that the truck you have selected has shown **:green[positive sales growth]** over the past year. 
                            The analysis indicates **:green[an increase of]** **:green[{:,.2f}%]** **:green[in sales]** compared to the previous year. This indicates an upward trend in sales 
                            for this specific truck. From this prediction, Tasty Bytes can expect a continuation of this **:green[positive trend]** into the upcoming year. This projection 
                            suggests that the sales performance for this truck is likely to continue improving in the next year.""" .format(year_percent_change))
            else:
                st.markdown("""Based on your selections, we have identified that the truck you have selected has shown **:red[negative sales growth]** over the past year. 
                            The analysis indicates **:red[a decrease of]** **:red[{:,.2f}%]** **:red[in sales]** compared to the previous year. This indicates a downward trend in sales 
                            for this specific truck. From this prediction, Tasty Bytes can anticipate a continuation of this **:red[negative trend]** into the upcoming year. This projection 
                            suggests that the sales performance for this truck is likely to continue declining in the next year.""" .format(year_percent_change))

                     
            
    else:
        st.error("Please fill in all required fields before proceeding with the prediction.")
    
            
    # Benefits
    st.markdown("## How it helps Tasty Bytes towards its high level goal of 25% YoY Sales")
    
    st.markdown("""
        Predicting the future sales of food trucks, whether they are **:blue[increasing]** or **:blue[decreasing]**, is a crucial strategy that aligns directly with Tasty Bytes' high-level goal of achieving a **:blue[25% Year-over-Year (YoY) sales growth]**. By leveraging advanced data analysis and predictive models, Tasty Bytes can gain valuable insights into the upcoming trends and shifts in their customer preferences.

        By accurately anticipating sales growth, Tasty Bytes can proactively allocate resources, plan inventory, and optimize operations to accommodate **:blue[increasing demand]**. This enables them to provide a seamless customer experience, prevent stockouts, and maintain high customer satisfaction levels. Moreover, the ability to cater to **:blue[rising sales]** effectively positions Tasty Bytes to capitalize on their projected growth, thereby helping them work towards their high level goal of a **:blue[25% YoY sales target]**.

        Conversely, if a **:blue[sales decrease]** were ot be predicted, Tasty Bytes can make informed decisions to mitigate the impact by exploring and adjusting their marketing strategies, introducing new offerings, or identifying potential factors causing the decline. Addressing these challenges head-on allows Tasty Bytes to **:blue[strategize and adapt]** swiftly, minimizing any adverse effects on their sales figures.

        In both scenarios, **:blue[accurate predictions]** empower Tasty Bytes to align its resources and efforts more effectively with the expected market conditions which ultimately helps them in reaching their high level goal of a **:blue[25% Year-over-Year (YoY) sales growth]**.
        """)
    
    
    # st.markdown("My ultimate goal in predicting future sales of food trucks is to increase Tasty Bytes sales by 5 percent, \
    #             contributing significantly to achieving a remarkable 25% YoY sales increase.  \
    #             By accurately forecasting sales for each food truck, \
    #             Tasty Bytes can optimize their inventory, and streamline their operations.")
    




    