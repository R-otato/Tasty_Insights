# Import statements--#
import streamlit as st
import pandas as pd
import joblib
import snowflake.connector
import ast
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# FUNCTIONS #

# Function: my_encode_units(x)
# the purpose of this function is to convert all positive values to 1 and everything else to 0
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# Function: get_item_bundles_unformatted(df)
# the purpose of this function is to search for bundles within the provided dataset
def get_item_bundles_unformatted(df):
    # Data preparation for apriori
    ## get table format for apriori algo
    mybasket = df.groupby(["ORDER_ID", "MENU_ITEM_ID"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("ORDER_ID")

    ## convert all positive values to 1 and everything else to 0
    my_basket_sets = mybasket.applymap(my_encode_units)

    # Apriori Model in action
    ## load apriori model
    apriori_model = joblib.load("assets/apriori_algo_product.joblib")

    ## generating rules
    my_rules = association_rules(apriori_model, metric="lift", min_threshold=14)

    # Fix formatting issue
    ## Convert frozensets to strings in the antecedents and consequents columns
    my_rules['antecedents'] = my_rules['antecedents'].apply(lambda x: tuple(x))
    my_rules['consequents'] = my_rules['consequents'].apply(lambda x: tuple(x))

    ## Convert 'frozenset' objects to strings in the 'antecedents' and 'consequents' columns
    my_rules['antecedents'] = my_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
    my_rules['consequents'] = my_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))

    # Remove duplicate bundles
    ## Create a new column with unique itemsets
    my_rules['itemset'] = my_rules[['antecedents', 'consequents']].apply(lambda x: frozenset(x), axis=1)

    ## Drop duplicates based on the 'itemset' column
    my_rules.drop_duplicates(subset='itemset', keep='first', inplace=True)

    ## Drop the 'itemset' column (optional)
    my_rules.drop(columns='itemset', inplace=True)

    ## Reset the index if needed
    my_rules.reset_index(drop=True, inplace=True)

    return my_rules


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
    my_cur.execute("select * from menu")
    menu_table = my_cur.fetchall()

    # create a DataFrame from the fetched result
    # remove cost of goods column due to irrelevance
    menu_table_df = pd.DataFrame(menu_table, columns=['MENU_ITEM_ID', 'MENU_ITEM_NAME'])

    return menu_table_df

# Function: display_bundles(my_rules)
# the purpose of this function is to manipulate the table such that the identified bundles are displayed in a specific format
def display_bundles(my_rules):
    ## get only the antecedents and consequents from the my_rules dataframe
    bundles_df = my_rules[['antecedents', 'consequents']]
        
    ## retrive menu table
    menu_df = retrieve_menu_table()

    ## convert both columns to int type
    bundles_df["antecedents"] = bundles_df["antecedents"].astype(int)
    bundles_df["consequents"] = bundles_df["consequents"].astype(int)

    ## merge with 'antecedents' to get antecedents_menu_name
    bundles_df = pd.merge(bundles_df, menu_df.rename(columns={'MENU_ITEM_ID': 'antecedents', 'MENU_ITEM_NAME': 'antecedents_item_name'}),
                        on='antecedents', how='left')

    ## merge with 'consequents' to get consequents_menu_name
    bundles_df = pd.merge(bundles_df, menu_df.rename(columns={'MENU_ITEM_ID': 'consequents', 'MENU_ITEM_NAME': 'consequents_item_name'}),
                        on='consequents', how='left')

    ## re-arrange columns
    bundles_df = bundles_df[["antecedents_item_name", "antecedents", "consequents_item_name", "consequents"]]


    ## convert both columns to str type
    bundles_df["antecedents"] = bundles_df["antecedents"].astype(str)
    bundles_df["consequents"] = bundles_df["consequents"].astype(str)

    ## create a new DataFrame with the rearranged columns and the new BundleNo, Item1, and Item2 columns
    final_bundles_df = bundles_df[["antecedents_item_name", "antecedents", "consequents_item_name", "consequents"]].copy()

    ## create the BundleNo column as an index starting from 1
    final_bundles_df.insert(0, 'BundleNo', range(1, 1 + len(final_bundles_df)))

    ## merge the 'antecedents_item_name' and 'antecedents' columns into 'Item1'
    final_bundles_df['Item1'] = final_bundles_df['antecedents_item_name'] + ' (' + final_bundles_df['antecedents'] + ')'

    ## merge the 'consequents_item_name' and 'consequents' columns into 'Item2'
    final_bundles_df['Item2'] = final_bundles_df['consequents_item_name'] + ' (' + final_bundles_df['consequents'] + ')'

    ## drop the individual 'antecedents_item_name', 'antecedents', 'consequents_item_name', and 'consequents' columns
    final_bundles_df.drop(columns=["antecedents_item_name", "antecedents", "consequents_item_name", "consequents"], inplace=True)

    return final_bundles_df

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


    #hide this using secrets
    my_cnx = snowflake.connector.connect(
        user = "RLIAM",
        password = "Cats2004",
        account = "LGHJQKA-DJ92750",
        role = "TASTY_BI",
        warehouse = "TASTY_BI_WH",
        database = "frostbyte_tasty_bytes",
        schema = "analytics"
    )

    my_cur = my_cnx.cursor()

    # Construct the SQL query for the current chunk
    query = f"SELECT MENU_ITEM_ID, TRUCK_ID FROM order_details_usa_matched"
    
    my_cur.execute(query)
    
    order_details_df = my_cur.fetchall()
    
    return order_details_df

#####################
##### MAIN CODE #####
#####################

# Page Title
st.markdown("# Product Team")

# Page Instructions (How to Use This Page)
with st.expander("How to Use This Page"):
    # List of steps
    st.write('1. Load you own dataset or Use the provided dataset')
    st.write('2. View the model\'s predictions')
    st.write('3. Analyse the visualisations below to gain insights on how to reduce customer churn from a product standpoint')

# # SEARCH FOR BUNDLES
# st.markdown("## Search for Bundles")

# # SECTION: INPUT DATA 
# ## File Upload section
# st.markdown("### Input Data")
# uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
# df=''
# ### If uploaded file is not empty
# if uploaded_files!=[]:
#     data_list = []
#     #Append all uploaded files into the list
#     for f in uploaded_files:
#         st.write(f)
#         temp_data = pd.read_csv(f)
#         data_list.append(temp_data)
#     st.success("Uploaded your file!")
#     #concat the files together if there are more than one file uploaded
#     df = pd.concat(data_list)
# else:
#     st.info("Using the last updated data of all the United States transactions. Upload a file above to use your own data!")
#     df=pd.read_csv('assets/apriori_dataset.csv')

# st.write(df)

## retrieve menu table with health metrics in different columns
menu_table_df = retrieve_menu_table()
menu_table = get_health_metrics_menu_table(menu_table_df)

# display current menu items
with st.expander("Current Menu Items"):
    st.dataframe(menu_table, hide_index=True)

# PRODUCT PERFORMANCE PREDICTION
st.markdown("## Product Performance")

## Option: menu type option
## add None as the default value (it won't be an actual selectable option)
default_option = None
menu_type_options = np.sort(menu_table['MENU_TYPE'].unique())

## use the updated list of options for the selectbox
selected_menu_type = st.selectbox("Menu Type: ", [default_option] + list(menu_type_options))


## Option: truck brand name
## add None as the default value (it won't be an actual selectable option)
default_option = None
truck_brand_name_options = np.sort(menu_table['TRUCK_BRAND_NAME'].unique())

## use the updated list of options for the selectbox
selected_truck_brand_name = st.selectbox("Truck Brand Name: ", [default_option] + list(truck_brand_name_options))


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

# display dataframe
st.dataframe(user_input_df, hide_index=True)


# Check for null values in the user_input_df
has_null_values = user_input_df.isnull().any().any()

if has_null_values == False:
    # display message if no null values are found
    st.write("Proceed to make a prediction.")
    
    # Make a prediction
    if st.button("Predict"):
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
            'TOTAL_COSTSE': [total_cost_price],
            'TOTAL_PROFIT': [total_profit],
            'GROSS_PROFIT_MARGIN (%)': [total_gross_profit_margin],
            'NET_PROFIT_MARGIN (%)': [total_net_profit_margin]
        }

        total_product_details_df = pd.DataFrame(data)
        
        st.markdown("### Prediction")
        
        ## display the rounded prediction
        st.markdown("##### Predicted Total Quantity Sold: {}".format(rounded_prediction))
        
        st.write('')
        
        st.markdown("##### Total Item Details:")
        ## display the total_product_details_df DataFrame
        st.dataframe(total_product_details_df, hide_index=True)
        
        # display current menu items
        with st.expander("Unit Item Details"):
            st.write("This label indicates that the table contains details specific to a single unit or item")
            ## display the new_product_details_df DataFrame
            st.dataframe(new_product_details_df, hide_index=True)
else:
    # display message if null values are found
    st.write("<span style='color:red'>Make sure all options have an input.</span>", unsafe_allow_html=True)















# # Retrieve bundles found by apriori
# my_rules = get_item_bundles_unformatted(df)


# # Show identified bundles
# st.markdown("### Identified Bundles")

# ## retrieve bundle dataframe
# final_bundles_df = display_bundles(my_rules)

# ## print the new DataFrame with the desired structure
# st.dataframe(final_bundles_df, hide_index=True)