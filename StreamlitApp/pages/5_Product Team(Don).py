# Import statements--#
import streamlit as st
import pandas as pd
import joblib
import snowflake.connector
import ast
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from snowflake.snowpark import Session

#################
### FUNCTIONS ###
#################

# Function: my_encode_units(x)
# the purpose of this function is to convert all positive values to 1 and everything else to 0
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

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
    my_cur.execute("select MENU_TYPE, TRUCK_BRAND_NAME, MENU_ITEM_ID, MENU_ITEM_NAME, ITEM_CATEGORY, ITEM_SUBCATEGORY, SALE_PRICE_USD, COST_OF_GOODS_USD, MENU_ITEM_HEALTH_METRICS_OBJ from menu")
    menu_table = my_cur.fetchall()

    # create a DataFrame from the fetched result
    # remove cost of goods column due to irrelevance
    menu_table_df = pd.DataFrame(menu_table, columns=['MENU_TYPE', 'TRUCK_BRAND_NAME', 'MENU_ITEM_ID', 'MENU_ITEM_NAME', 
                                                    'ITEM_CATEGORY', 'ITEM_SUBCATEGORY', 'SALE_PRICE_USD', 'COST_OF_GOODS_USD', 'MENU_ITEM_HEALTH_METRICS_OBJ'])


    # rename the 'SALE_PRICE_USD' column to 'UNIT_PRICE'
    menu_table_df = menu_table_df.rename(columns={'SALE_PRICE_USD': 'UNIT_PRICE'})

    # rename the 'COST_OF_GOODS_USED' to 'COST_OF_GOODS'
    menu_table_df = menu_table_df.rename(columns={'COST_OF_GOODS_USD': 'COST_OF_GOODS'})
    
    
    # Add profit column
    ## calculate the profit column
    profit_column = menu_table_df['UNIT_PRICE'] - menu_table_df['COST_OF_GOODS']

    ## get the index of the 'COST_OF_GOODS' column
    cost_of_goods_index = menu_table_df.columns.get_loc('COST_OF_GOODS')

    ## insert the 'profit' column to the right of the 'COST_OF_GOODS' column
    menu_table_df.insert(cost_of_goods_index + 1, 'UNIT_PROFIT', profit_column)
    
    
    # Add gross profit margin column
    ## calculate gross profit margin
    gross_profit_margin = ((menu_table_df['UNIT_PRICE'] - menu_table_df['COST_OF_GOODS']) / menu_table_df['UNIT_PRICE']) * 100
    
    ## get the index of the 'UNIT_PROFIT' column
    unit_profit_index = menu_table_df.columns.get_loc('UNIT_PROFIT')
    
    ## insert the 'UNIT_GROSS_PROFIT_MARGIN (%)' column to the right of the 'UNIT_PROFIT' column
    menu_table_df.insert(unit_profit_index + 1, 'UNIT_GROSS_PROFIT_MARGIN (%)', gross_profit_margin)
    
    
    # Add net profit margin column
    ## calculate net profit margin
    net_profit_margin = (menu_table_df['UNIT_PROFIT'] / menu_table_df['UNIT_PRICE']) * 100
    
    ## get the index of the 'UNIT_GROSS_PROFIT_MARGIN (%)' column
    unit_gross_profit_margin_index = menu_table_df.columns.get_loc('UNIT_GROSS_PROFIT_MARGIN (%)')
    
    ## insert the 'UNIT_GROSS_PROFIT_MARGIN (%)' column to the right of the 'UNIT_PROFIT' column
    menu_table_df.insert(unit_gross_profit_margin_index + 1, 'UNIT_NET_PROFIT_MARGIN (%)', net_profit_margin)
    
    
    # round off sale price to 2dp
    menu_table_df['UNIT_PRICE'] = menu_table_df['UNIT_PRICE'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off cost of goods price to 2dp
    menu_table_df['COST_OF_GOODS'] = menu_table_df['COST_OF_GOODS'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off profit amount to 2dp
    menu_table_df['UNIT_PROFIT'] = menu_table_df['UNIT_PROFIT'].apply(lambda x: '{:.2f}'.format(x))
    
    # round off gross profit margin to 1dp
    menu_table_df['UNIT_GROSS_PROFIT_MARGIN (%)'] = menu_table_df['UNIT_GROSS_PROFIT_MARGIN (%)'].apply(lambda x: '{:.1f}'.format(x))
    
    # round off net profit margin to 1dp
    menu_table_df['UNIT_NET_PROFIT_MARGIN (%)'] = menu_table_df['UNIT_NET_PROFIT_MARGIN (%)'].apply(lambda x: '{:.1f}'.format(x))
    
    return menu_table_df

# Function: get_health_metrics_menu_table
# the purpose of this function is to manipulate the data in the 'MENU_ITEM_HEALTH_METRICS_OBJ' to get only the health metrics info with its corresponding column values bring Yes or No
def get_health_metrics_menu_table(menu_table_df):
    # Convert the string JSON data to a nested dictionary
    menu_table_df['MENU_ITEM_HEALTH_METRICS_OBJ'] = menu_table_df['MENU_ITEM_HEALTH_METRICS_OBJ'].apply(ast.literal_eval)

    # Use json_normalize to flatten the nested JSON data
    menu_item_metrics = pd.json_normalize(menu_table_df['MENU_ITEM_HEALTH_METRICS_OBJ'], record_path='menu_item_health_metrics')

    # Rename the columns
    menu_item_metrics = menu_item_metrics.rename(columns={
        'is_dairy_free_flag': 'DAIRY_FREE',
        'is_gluten_free_flag': 'GLUTEN_FREE',
        'is_healthy_flag': 'HEALTHY',
        'is_nut_free_flag': 'NUT_FREE'
    })

    # Replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    menu_item_metrics = menu_item_metrics.replace({'Y': 'Yes', 'N': 'No'})

    # Concatenate the flattened DataFrame with the original DataFrame
    menu_table_df = pd.concat([menu_table_df, menu_item_metrics], axis=1)

    # Drop the original 'MENU_ITEM_HEALTH_METRICS_OBJ' and 'ingredients' column 
    menu_table_df = menu_table_df.drop(columns=['MENU_ITEM_HEALTH_METRICS_OBJ', 'ingredients'])
    
    return menu_table_df

# Function: user_inputs()
# the purpose of this function is to get the user's input for the new product item they would like to predict the total quantity sold for
def user_inputs(): 
    ## Option: truck brand name
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    truck_brand_name_options = np.sort(menu_table['TRUCK_BRAND_NAME'].unique())

    ## use the updated list of options for the selectbox
    selected_truck_brand_name = st.selectbox("Truck Brand Name: ", [default_option] + list(truck_brand_name_options))

    # Filter the menu_table to find the menu type for the selected truck brand name
    menu_type_filter = menu_table['TRUCK_BRAND_NAME'] == selected_truck_brand_name
    if menu_type_filter.any():
        selected_menu_type = menu_table.loc[menu_type_filter, 'MENU_TYPE'].values[0]
    else:
        selected_menu_type = None

    ## Option: item category
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    item_cat_options = np.sort(menu_table['ITEM_CATEGORY'].unique())

    ## use the updated list of options for the selectbox
    selected_item_cat = st.selectbox("Item Category: ", [default_option] + list(item_cat_options))


    ## Option: item subcategory
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    item_subcat_options = np.sort(menu_table['ITEM_SUBCATEGORY'].unique())

    ## use the updated list of options for the selectbox
    selected_item_subcat = st.selectbox("Item Subcategory: ", [default_option] + list(item_subcat_options))


    ## Option: cost of goods
    cost_of_goods = st.text_input("Enter cost of goods:")


    ## Option: sale price
    sale_price = st.text_input("Enter sale price:")


    ## Option: healthy
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    healthy_options = np.sort(menu_table['HEALTHY'].unique())

    ## use the updated list of options for the selectbox
    selected_is_healthy = st.selectbox("Healthy: ", [default_option] + list(healthy_options))


    ## Option: dairy free
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    dairy_free_options = np.sort(menu_table['DAIRY_FREE'].unique())

    ## use the updated list of options for the selectbox
    selected_is_dairy_free = st.selectbox("Dairy Free: ", [default_option] + list(dairy_free_options))


    ## Option: gluten free
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    gluten_free_options = np.sort(menu_table['GLUTEN_FREE'].unique())

    ## use the updated list of options for the selectbox
    selected_is_gluten_free = st.selectbox("Gluten Free: ", [default_option] + list(gluten_free_options))


    ## Option: nut free
    ## add None as the default value (it won't be an actual selectable option)
    default_option = None
    nut_free_options = np.sort(menu_table['NUT_FREE'].unique())

    ## use the updated list of options for the selectbox
    selected_is_nut_free = st.selectbox("Nut Free: ", [default_option] + list(nut_free_options))

    user_input_full = {
        "MENU_TYPE": selected_menu_type, 
        "TRUCK_BRAND_NAME": selected_truck_brand_name, 
        "ITEM_CATEGORY": selected_item_cat, 
        "ITEM_SUBCATEGORY": selected_item_subcat, 
        "SALE_PRICE_USD": sale_price,
        "HEALTHY": selected_is_healthy,
        "DAIRY_FREE": selected_is_dairy_free, 
        "GLUTEN_FREE": selected_is_gluten_free, 
        "NUT_FREE": selected_is_nut_free
    }

    # create dataframe with all the user's inputs
    user_input_df = pd.DataFrame(user_input_full, index=[1])

    return user_input_df, sale_price, cost_of_goods, selected_menu_type

# Function: prediction()
# the purpose of this function is to carry out certain data transformations and create the 2 tables shown after prediction
def prediction(user_input_df):
    # replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    user_input_df = user_input_df.replace({"Yes": 1, "No":0})
    
    # MANUAL ENCODING
    categorical_cols = ["MENU_TYPE", "TRUCK_BRAND_NAME", "ITEM_CATEGORY", "ITEM_SUBCATEGORY"]

    # Loop through each categorical column
    for col in categorical_cols:
        # Get the unique values in the column
        unique_values = menu_table[col].unique()

        # Loop through unique values in the column
        for value in unique_values:
            # Check if the value in the menu_table matches the corresponding value in user_input_df
            if value == user_input_df[col].values[0]:
                # Create a column with the name 'column_selected_value' and set its value to 1
                menu_table[f'{col}_{value}'] = 1

                # Add this column to the user_input_df
                user_input_df[f'{col}_{value}'] = 1
            else:
                # Create a column with the name 'column_unique_value' and set its value to 0
                menu_table[f'{col}_{value}'] = 0

                # Add this column to the user_input_df
                user_input_df[f'{col}_{value}'] = 0


    # Drop the original categorical columns from user_input_df
    user_input_df.drop(columns=categorical_cols, inplace=True)

    user_input_df.drop(columns=["ITEM_SUBCATEGORY_Hot Option", "MENU_TYPE_Sandwiches", "TRUCK_BRAND_NAME_Better Off Bread", "ITEM_CATEGORY_Dessert"], inplace = True)

    desired_order = ['SALE_PRICE_USD', 'DAIRY_FREE', 'GLUTEN_FREE', 'HEALTHY', 'NUT_FREE',
                'MENU_TYPE_Ethiopian', 'MENU_TYPE_Gyros', 'MENU_TYPE_Indian',
                'MENU_TYPE_Hot Dogs', 'MENU_TYPE_Vegetarian', 'MENU_TYPE_Tacos',
                'MENU_TYPE_BBQ', 'MENU_TYPE_Crepes', 'MENU_TYPE_Poutine',
                'MENU_TYPE_Ice Cream', 'MENU_TYPE_Grilled Cheese', 'MENU_TYPE_Ramen',
                'MENU_TYPE_Mac & Cheese', 'MENU_TYPE_Chinese',
                'TRUCK_BRAND_NAME_Tasty Tibs', 'TRUCK_BRAND_NAME_Cheeky Greek',
                'TRUCK_BRAND_NAME_Nani\'s Kitchen', 'TRUCK_BRAND_NAME_Amped Up Franks',
                'TRUCK_BRAND_NAME_Plant Palace', 'TRUCK_BRAND_NAME_Guac n\' Roll',
                'TRUCK_BRAND_NAME_Smoky BBQ', 'TRUCK_BRAND_NAME_Le Coin des Crêpes',
                'TRUCK_BRAND_NAME_Revenge of the Curds',
                'TRUCK_BRAND_NAME_Freezing Point', 'TRUCK_BRAND_NAME_The Mega Melt',
                'TRUCK_BRAND_NAME_Kitakata Ramen Bar', 'TRUCK_BRAND_NAME_The Mac Shack',
                'TRUCK_BRAND_NAME_Peking Truck', 'ITEM_CATEGORY_Main',
                'ITEM_CATEGORY_Beverage', 'ITEM_CATEGORY_Snack',
                'ITEM_SUBCATEGORY_Warm Option', 'ITEM_SUBCATEGORY_Cold Option']

    user_input_df = user_input_df.reindex(columns=desired_order)
    
    # Convert 'SALE_PRICE_USD' column to numeric type
    user_input_df['SALE_PRICE_USD'] = pd.to_numeric(user_input_df['SALE_PRICE_USD'])
    
    # retrieve min max scaler
    min_max_scaler = joblib.load("assets/product_team_min_max_scaler.joblib")
    
    min_max_scaler.transform(user_input_df)
    
    # retrieve regression model
    product_qty_model = joblib.load("assets/product_qty_regression.joblib")
    
    prediction = product_qty_model.predict(user_input_df)
    
    # Round off the prediction to the nearest whole number
    rounded_prediction = round(prediction[0])

    
    # Show New Product Details Table
    ## calculate UNIT_SALE_PRICE, UNIT_COST_PRICE, UNIT_PROFIT
    unit_sale_price = float(sale_price)
    unit_cost_price = float(cost_of_goods)
    unit_profit = unit_sale_price - unit_cost_price

    ## calculate UNIT_GROSS_PROFIT_MARGIN (%) and UNIT_NET_PROFIT_MARGIN (%)
    unit_gross_profit_margin = (unit_profit / unit_sale_price) * 100
    unit_net_profit_margin = (unit_profit / unit_sale_price) * 100

    # Round the profit margin values to the nearest whole number
    unit_gross_profit_margin = round(unit_gross_profit_margin)
    unit_net_profit_margin = round(unit_net_profit_margin)

    ## create the new_product_details_df DataFrame
    data = {
        'UNIT_SALE_PRICE': [unit_sale_price],
        'UNIT_COST_PRICE': [unit_cost_price],
        'UNIT_PROFIT': [unit_profit],
        'UNIT_GROSS_PROFIT_MARGIN (%)': [unit_gross_profit_margin],
        'UNIT_NET_PROFIT_MARGIN (%)': [unit_net_profit_margin]
    }
    ## convert to dataframe
    new_product_details_df = pd.DataFrame(data)
    
    
    # Prediction Total Details Table
    ## calculate TOTAL_SALE_PRICE, TOTAL_COST_PRICE, TOTAL_PROFIT
    total_sale_price = float(sale_price) * rounded_prediction
    total_cost_price = float(cost_of_goods) * rounded_prediction
    total_profit = total_sale_price - total_cost_price

    ## calculate TOTAL_GROSS_PROFIT_MARGIN (%) and TOTAL_NET_PROFIT_MARGIN (%)
    total_gross_profit_margin = (total_profit / total_sale_price) * 100
    total_net_profit_margin = (total_profit / total_sale_price) * 100

    ## round the profit margin values to the nearest whole number
    total_gross_profit_margin = round(total_gross_profit_margin)
    total_net_profit_margin = round(total_net_profit_margin)

    ## create the total_product_details_df DataFrame
    data = {
        'TOTAL_SALES': [total_sale_price],
        'TOTAL_COSTS': [total_cost_price],
        'TOTAL_PROFIT': [total_profit],
        'GROSS_PROFIT_MARGIN (%)': [total_gross_profit_margin],
        'NET_PROFIT_MARGIN (%)': [total_net_profit_margin]
    }

    total_product_details_df = pd.DataFrame(data)
    
    return total_product_details_df, new_product_details_df, rounded_prediction


#####################
##### MAIN CODE #####
#####################

st.markdown("# Product Team")
tab1, tab2 = st.tabs(['About', 'Model Prediction'])

# TAB 1: About
with tab1:
    # High Level Goals Explanations
    st.markdown("# High Level Goals")
    st.write("""This page is dedicated to helping Tasty Bytes reach its goal of achieving a 25% YoY sales increase, from \$105M to \$320M over 5 years with 
             the sucess metric being the increase of average quantity sold of all products by 25% YoY to achieve the high level goal.

Combining the power of data analysis and machine learning, the page predicts the quantity sold for a specific menu item based on historical
                data. Accurately forecasting the next month's quantity sold of a specific menu item can help Tasty Bytes can make informed decisions and 
                implement targeted strategies to optimise sales for menu items. With a clear understanding of upcoming sales trends, Tasty Bytes can 
                proactively adjust inventory, marketing efforts, and operational aspects, maximizing revenue potential.

The data-driven insights provided by this page offer Tasty Bytes a competitive advantage, enabling Tasty Bytes to stay ahead in the highly
                dynamic food truck industry. By leveraging the personalized predictions for each menu item, the product team can focus on strategic 
                decision-making tailored to the performance of individual products. This personalized approach enhances customer satisfaction, fosters 
                loyalty, and boosts overall profitability.""")



    # Utilisation of model's prediction
    st.markdown("# How to utilise predictions?")
    
    st.write("""The model prediction tab will allow you to gain insights from the next month quantity sold prediction model. You can select the different
             menu items from the dropdown and view the model's predicted quantities sold for the next month.""")
    
    st.write("""
             The model's prediction can provide insights and support the following:
             - Inventory Management: Enable the product team to optimise inventory levels for each menu item. Avoid overstocking or understocking, reducing
             waste and minimising carrying costs.
             - Marketing Strategies: Tailor marketing efforts and promotions to maximise the impact. Focus marketing campaigns on menu items that are 
             predicted to perform well and to amplify marketing efforts for underperforming menu items, driving customer interest and boosting sales.
             - Faciliate Franchisees' Overall Planning: Aid franchisees' overall planning in terms of inventory management, logistic management, and budgeting purposes
             """)

    
    # Limitations and Assumptions the model makes
    st.markdown("# Limitations and assumptions the model makes")
    st.write("""The limitation to my model is that it assumes that all the time business is as usual. It does not take into account external factors such as
             changes in customer preferences, economic conditions, or marketing campaigns that could significantly impact sales which can lead to inaccurate
             insights and data driven decisions such as menu optimisation, marketing strategies and inventory management.
             
The assumption that my model makes is that the December 2022 quantity sold is the average of the past 11 months of 2022. The year on year percentage
             increase could be slightly larger or smaller. However, this is only one month of assumption therefore, it should not impact the model to a large
             extent.""")
        

# TAB 2: Model Prediction
with tab2:
    st.markdown("## Menu Item Next Month Sales Prediction")
    
    default_option = None
    
    menu_table_df = retrieve_menu_table()
    menu_table = get_health_metrics_menu_table(menu_table_df)
    # get menu item options for users to choose
    menu_item_options = [
    f"({row['MENU_ITEM_ID']}) {row['MENU_ITEM_NAME']}"
    for _, row in menu_table.iterrows()
    ]

    # use the updated list of options for the selectbox
    # user can select menu item they want to predict next month quantity sold for
    selected_item_cat = st.selectbox("Menu Item: ", [default_option] + list(menu_item_options))
    
    if selected_item_cat == None:
        st.error("Please fill in the required field to get a prediction")
    
    else:
        # extract MENU_ITEM_ID from the option string
        menu_item_id = int(selected_item_cat.split(")")[0][1:])

        menu_item_name = selected_item_cat.split(") ")[1]
        
        item_info_df = menu_table[menu_table["MENU_ITEM_ID"] == menu_item_id]
        
        item_info_df = item_info_df.drop(["MENU_ITEM_NAME", "COST_OF_GOODS", "UNIT_PROFIT", "UNIT_GROSS_PROFIT_MARGIN (%)", "UNIT_NET_PROFIT_MARGIN (%)"], axis=1)
        
        item_info_df = item_info_df.rename(columns={'UNIT_PRICE': 'SALE_PRICE_USD'})
        
        # retrieve year and month from order timestamp
        order_df = pd.read_csv('assets/total_qty_by_item.csv')
        

        # Convert the 'YEAR' column to numeric values
        order_df['YEAR'] = order_df['YEAR'].astype(str).replace(',', '').astype(int)
        
        
        # get the total qty sold over the years for a particular menu item
        total_qty_by_item_over_time = order_df[order_df["MENU_ITEM_ID"]==menu_item_id]
        
        
        # Plotly Line Chart
        ## create the line chart
        fig = go.Figure(data=go.Line(x=total_qty_by_item_over_time['YEAR'], y=total_qty_by_item_over_time['TOTAL_QTY_SOLD_PER_YEAR'], mode='lines+markers'))

        ## update the layout
        fig.update_layout(title='Total Quantity Sold per Year',
                        xaxis_title='Year',
                        yaxis_title='Total Qty Sold')



        # get one year after the latest year provided in the data
        year = total_qty_by_item_over_time["YEAR"].max() + 1
        
        
        # Replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
        item_info_df = item_info_df.replace({'Yes': 1, 'No': 0})
        
        
        
        # MANUAL ONT HOT ENCODING
        
        # Define the mapping dictionary
        temperature_mapping = {'Cold Option': 0, 'Warm Option': 1, 'Hot Option': 2}

        # Apply the mapping to the 'ITEM_SUBCATEGORY' column in item_info_df
        item_info_df['ITEM_SUBCATEGORY'] = item_info_df['ITEM_SUBCATEGORY'].map(temperature_mapping)

        item_info_df.head()
        
        ## state cat cols to carry out manual encoding on
        categorical_cols = ["MENU_TYPE", "TRUCK_BRAND_NAME", "ITEM_CATEGORY"]
        
        ## loop through each categorical column
        for col in categorical_cols:
            ## get the unique values in the column
            unique_values = menu_table[col].unique()

            ## loop through unique values in the column
            for value in unique_values:
                ## check if the value in the menu_table table matches the corresponding value in item_info_df
                if value == item_info_df[col].values[0]:
                    ## create a column with the name 'column_selected_value' and set its value to 1
                    menu_table[f'{col}_{value}'] = 1

                    ## add this column to the item_info_df
                    item_info_df[f'{col}_{value}'] = 1
                else:
                    ## create a column with the name 'column_unique_value' and set its value to 0
                    menu_table[f'{col}_{value}'] = 0

                    ## add this column to the item_info_df
                    item_info_df[f'{col}_{value}'] = 0

        ## drop the original categorical columns from item_info_df
        item_info_df.drop(columns=categorical_cols, inplace=True)
        
        
        
        ## assign the columsn YEAR with their respective values
        item_info_df['YEAR'] = year
        
        # define the desired column order
        desired_columns = ['MENU_ITEM_ID', 'ITEM_SUBCATEGORY', 'SALE_PRICE_USD', 'YEAR',
       'DAIRY_FREE', 'GLUTEN_FREE', 'HEALTHY', 'NUT_FREE',
       'MENU_TYPE_Ethiopian', 'MENU_TYPE_Ice Cream', 'MENU_TYPE_Chinese',
       'MENU_TYPE_BBQ', 'MENU_TYPE_Hot Dogs', 'MENU_TYPE_Gyros',
       'MENU_TYPE_Tacos', 'MENU_TYPE_Vegetarian', 'MENU_TYPE_Ramen',
       'MENU_TYPE_Crepes', 'MENU_TYPE_Poutine', 'MENU_TYPE_Sandwiches',
       'MENU_TYPE_Grilled Cheese', 'MENU_TYPE_Mac & Cheese',
       'TRUCK_BRAND_NAME_Tasty Tibs', 'TRUCK_BRAND_NAME_Freezing Point',
       'TRUCK_BRAND_NAME_Peking Truck', 'TRUCK_BRAND_NAME_Smoky BBQ',
       'TRUCK_BRAND_NAME_Amped Up Franks', 'TRUCK_BRAND_NAME_Cheeky Greek',
       'TRUCK_BRAND_NAME_Guac n\' Roll', 'TRUCK_BRAND_NAME_Plant Palace',
       'TRUCK_BRAND_NAME_Kitakata Ramen Bar',
       'TRUCK_BRAND_NAME_Le Coin des Crêpes',
       'TRUCK_BRAND_NAME_Revenge of the Curds',
       'TRUCK_BRAND_NAME_Better Off Bread', 'TRUCK_BRAND_NAME_The Mega Melt',
       'TRUCK_BRAND_NAME_The Mac Shack', 'ITEM_CATEGORY_Beverage',
       'ITEM_CATEGORY_Dessert', 'ITEM_CATEGORY_Main']

        # drop columns not in the desired column list
        item_info_df = item_info_df[desired_columns]

        # convert SALE_PRICE_USD column value to float
        item_info_df["SALE_PRICE_USD"] = item_info_df["SALE_PRICE_USD"].astype(float)

        
        # retrieve min max scaler
        min_max_scaler = joblib.load("assets/product_qty_year_min_max_scaler.joblib")
        
        min_max_scaler.fit(item_info_df)
        
        min_max_scaler.transform(item_info_df)
        
        
        # retrieve regression model
        product_qty_per_year_model = joblib.load("assets/product_qty_year_xgb_model.joblib")
        
        model_prediction = product_qty_per_year_model.predict(item_info_df)
        
        
        # Round off the prediction to the nearest whole number
        rounded_prediction = round(model_prediction[0])
        
        # retrieve the unit price for selected item
        unit_price = menu_table.loc[menu_table['MENU_ITEM_ID'] == menu_item_id, 'UNIT_PRICE'].values[0]
        
        # get the total sales for the next year
        sales_next_year = float(unit_price) * int(rounded_prediction)
        
        
        # Get previous year sales
        ## sort the DataFrame by 'YEAR' in descending order
        total_qty_by_item_over_time_sorted = total_qty_by_item_over_time.sort_values(by='YEAR', ascending=False)
        
        ## keep only the first row for each 'MENU_ITEM_ID' which is the latest
        total_qty_by_item_over_time = total_qty_by_item_over_time_sorted.groupby('MENU_ITEM_ID').first().reset_index()
        
        ## get the quantity sold for the latest year
        qty_sold_last_year = int(total_qty_by_item_over_time["TOTAL_QTY_SOLD_PER_YEAR"])
        
        ## get the total sales for the latest year
        sales_last_year = int(total_qty_by_item_over_time["TOTAL_QTY_SOLD_PER_YEAR"]) * float(unit_price)

        
        # Get the percentage month to month change (latest to predicted)
        sales_change = (rounded_prediction*float(unit_price)) - sales_last_year
        
        # get the percentage change in sales comparing previous year and predicted year
        percent_change = ((sales_change / sales_last_year)*100)



        # DISPLAY
        st.markdown("## Prediction:")

        st.markdown("### No. of {} sold next year: {}".format(menu_item_name, rounded_prediction))
        st.markdown("### Estimated sales next year: ${:.2f}".format(sales_next_year))
        st.markdown("### Estimated Year on Year sales growth: {:.2f}%".format(percent_change))
        
        with st.expander("{} Historical Sales Data".format(menu_item_name)):
            # show historical qty sold over the years
            total_qty_by_item_over_time_sorted = total_qty_by_item_over_time_sorted.sort_values(by='YEAR', ascending=True)
            qty_sold_historic_year = total_qty_by_item_over_time_sorted.drop("MENU_ITEM_ID", axis=1)
            
            # Calculate the 'TOTAL_SALES' column by multiplying 'TOTAL_QTY_SOLD_PER_YEAR' with 'unit_price'
            qty_sold_historic_year['TOTAL_SALES'] = qty_sold_historic_year['TOTAL_QTY_SOLD_PER_YEAR'].astype(float) * float(unit_price)
            
            # convert the 'YEAR' column to string to remove ','
            qty_sold_historic_year['YEAR'] = qty_sold_historic_year['YEAR'].astype(str).replace(',', '').astype(str)

            # display table
            st.dataframe(qty_sold_historic_year, hide_index=True)
        
        
        
            # show the plot in the Streamlit app 
            st.plotly_chart(fig)
        