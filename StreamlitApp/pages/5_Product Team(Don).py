#--Import statements--
import streamlit as st
import pandas as pd
import joblib 
import snowflake.connector
import ast

#--Functions--#

# Function: pipline
# the purpose of this function is to carry out the necessary transformations on the data provided by the user so that it can be fed into the machine learning model for prediction
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

# Function: multi_select_custid_individual
# the purpose of this function is to allow the user to multi-select individual customers to check the model's prediction on each customer (churned or not churned)
def multi_select_custid_individual(data):
    
    # Get all unique customer IDs
    all_customer_ids = data['CUSTOMER_ID'].unique()
    
    # Individual customer's churned status 
    st.markdown("## Model Output: Customer's Churned Status")
    
    # Create a multiselect dropdown with checkboxes for customer IDs
    selected_customer_ids = st.multiselect("Select Customer IDs:", all_customer_ids)

    # Filter the DataFrame based on selected customer IDs
    if selected_customer_ids:
        filtered_data = data[data['CUSTOMER_ID'].isin(selected_customer_ids)]
        st.dataframe(filtered_data, hide_index=True)
        
        # Update churn counts based on the filtered data
        churn_counts = filtered_data['CHURNED'].value_counts().reset_index()
        churn_counts = churn_counts.rename(columns={'count': 'Number of Customers'})
        st.dataframe(churn_counts, hide_index=True)
    
    else:
        ## show model result for churned customers only
        st.dataframe(data, hide_index = True)
        
        # Number of customers by churn group
        churn_counts = data['CHURNED'].value_counts().reset_index()
        churn_counts = churn_counts.rename(columns={'count': 'Number of Customers'})
        st.dataframe(churn_counts, hide_index=True)

# Function: retrive_menu_table
# the purpose of this function is to retrieve the menu table from snowflake containing all the details of the menu items which will then be merged with the transactions info to help the product team gain insight
def retrieve_menu_table():
    # RETRIEVE MENU TABLE FROM SNOWFLAKE
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

    ## retrieve menu table from snowflake
    my_cur = my_cnx.cursor()
    my_cur.execute("select MENU_TYPE, TRUCK_BRAND_NAME, MENU_ITEM_ID, MENU_ITEM_NAME, ITEM_CATEGORY, ITEM_SUBCATEGORY, SALE_PRICE_USD, MENU_ITEM_HEALTH_METRICS_OBJ from menu")
    menu_table = my_cur.fetchall()

    ## create a DataFrame from the fetched result
    ## remove cost of goods column due to irrelevance
    menu_table_df = pd.DataFrame(menu_table, columns=['MENU_TYPE', 'TRUCK_BRAND_NAME', 'MENU_ITEM_ID', 'MENU_ITEM_NAME', 
                                                    'ITEM_CATEGORY', 'ITEM_SUBCATEGORY', 'SALE_PRICE_USD', 'MENU_ITEM_HEALTH_METRICS_OBJ'])

    # Rename the 'SALE_PRICE_USD' column to 'UNIT_PRICE'
    menu_table_df = menu_table_df.rename(columns={'SALE_PRICE_USD': 'UNIT_PRICE'})

    # round off sale price to 2dp
    menu_table_df['UNIT_PRICE'] = menu_table_df['UNIT_PRICE'].apply(lambda x: '{:.2f}'.format(x))
    
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
        'is_dairy_free_flag': 'IS_DAIRY_FREE',
        'is_gluten_free_flag': 'IS_GLUTEN_FREE',
        'is_healthy_flag': 'IS_HEALTHY',
        'is_nut_free_flag': 'IS_NUT_FREE'
    })

    # Replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
    menu_item_metrics = menu_item_metrics.replace({'Y': 'Yes', 'N': 'No'})

    # Concatenate the flattened DataFrame with the original DataFrame
    menu_table_df = pd.concat([menu_table_df, menu_item_metrics], axis=1)

    # Drop the original 'MENU_ITEM_HEALTH_METRICS_OBJ' and 'ingredients' column 
    menu_table_df = menu_table_df.drop(columns=['MENU_ITEM_HEALTH_METRICS_OBJ', 'ingredients'])
    
    return menu_table_df

# Function: retrieve_order_details
# the purpose of this function is to retrieve order info from Snowflake that is to be merged with the menu table to allow the product team to gain further insight about products and orders
def retrieve_order_details():
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
        query = f"SELECT ORDER_ID, CUSTOMER_ID, MENU_ITEM_ID, QUANTITY, PRICE, ORDER_TOTAL FROM order_details_usa_matched WHERE CUSTOMER_ID IN ({customer_ids_str})"

        # Execute the SQL query for the current chunk
        my_cur.execute(query)

        # Fetch the result for the current chunk
        chunk_result = my_cur.fetchall()

        # Append the chunk result to the overall result
        order_details.extend(chunk_result)

    # Create a DataFrame from the fetched result
    order_details_df = pd.DataFrame(order_details, columns=['ORDER_ID', 'CUSTOMER_ID', 'MENU_ITEM_ID', 'QUANTITY', 'PRODUCT_TOTAL', 'ORDER_TOTAL'])

    # Convert ORDER_ID to string and then remove commas
    order_details_df['ORDER_ID'] = order_details_df['ORDER_ID'].astype(str).str.replace(',', '')

    # Format ORDER_TOTAL and PRODUCT_TOTAL_PRICE columns to 2 decimal places
    order_details_df['ORDER_TOTAL'] = order_details_df['ORDER_TOTAL'].apply(lambda x: '{:.2f}'.format(x))
    order_details_df['PRODUCT_TOTAL'] = order_details_df['PRODUCT_TOTAL'].apply(lambda x: '{:.2f}'.format(x))

    order_details_df = order_details_df.sort_values(by='CUSTOMER_ID')
    
    return order_details_df

# Function: convert_df_to_csv
# the purpose of this function is to convert the pandas dataframe to csv so that the user can export the data for further visualisation, exploration, or analysis
def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

# Function: get_overall_table
# the purpose of this function is to merge the menu and order details table together to form an overall table
def get_overall_table(order_details_df, menu_table_df):
    ## Merge the DataFrames based on 'MENU_ITEM_ID'
    merged_df = pd.merge(order_details_df, menu_table_df, on='MENU_ITEM_ID', how='left')

    ## Define the desired column order
    desired_columns = ['ORDER_ID', 'CUSTOMER_ID', 'MENU_TYPE','TRUCK_BRAND_NAME',  'MENU_ITEM_ID', 'MENU_ITEM_NAME', 'ITEM_CATEGORY', 'ITEM_SUBCATEGORY',
                    'IS_DAIRY_FREE', 'IS_GLUTEN_FREE', 'IS_HEALTHY', 'IS_NUT_FREE', 'QUANTITY', 'UNIT_PRICE', 'PRODUCT_TOTAL', 'ORDER_TOTAL']

    ## Re-arrange the columns in the merged DataFrame
    merged_df = merged_df[desired_columns]
    
    return merged_df

# Function: menu_item_table
# the purpose of this function is to carry out data manipulation to retrieve additional information for each menu item to form a separate table for the menu items
def menu_item_table():

    # create initial final product table
    final_product_df = menu_table_df[['MENU_ITEM_ID', 'MENU_ITEM_NAME', 'UNIT_PRICE']].sort_values(by='MENU_ITEM_ID')

    # Get the total quantity sold for each menu item 
    ## group by 'MENU_ITEM_ID' and calculate the total quantity sold
    total_qty_sold_per_item = merged_df.groupby('MENU_ITEM_ID')['QUANTITY'].sum().reset_index()

    ## rename the 'QUANTITY' column to 'AVERAGE_QUANTITY'
    total_qty_sold_per_item = total_qty_sold_per_item.rename(columns={'QUANTITY': 'TOTAL_QTY_SOLD'})

    ## merge total_qty_sold_per_item with final_product_df
    final_product_df = pd.merge(final_product_df, total_qty_sold_per_item, on='MENU_ITEM_ID')


    # Get the total sales for each menu item
    ## convert 'PRODUCT_TOTAL' column to numeric
    merged_df['PRODUCT_TOTAL'] = merged_df['PRODUCT_TOTAL'].astype(float)

    ## group by 'MENU_ITEM_ID' and calculate the average 'PRODUCT_TOTAL'
    total_sales_per_item = merged_df.groupby('MENU_ITEM_ID')['PRODUCT_TOTAL'].sum().reset_index()

    ## rename the column for clarity
    total_sales_per_item.rename(columns={'PRODUCT_TOTAL': 'TOTAL_SALES'}, inplace=True)

    ## merge total_qty_sold_per_item with final_product_df
    final_product_df = pd.merge(final_product_df, total_sales_per_item, on='MENU_ITEM_ID')


    # Get the total number of transactions for each menu item
    ## get the count of each menu_item_id in merged_df
    menu_item_counts = merged_df['MENU_ITEM_ID'].value_counts().reset_index()

    ## rename the columns for clarity
    menu_item_counts.columns = ['MENU_ITEM_ID', 'TOTAL NO. OF TRANSACTIONS']

    ## sort the results by menu_item_id (optional)
    menu_item_counts = menu_item_counts.sort_values(by='MENU_ITEM_ID')

    ## merge menu_item_counts with final_product_df
    final_product_df = pd.merge(final_product_df, menu_item_counts, on='MENU_ITEM_ID')


    # round off sale price to 2dp
    final_product_df['TOTAL_SALES'] = final_product_df['TOTAL_SALES'].apply(lambda x: '{:.2f}'.format(x))

    return final_product_df


#####################
##### MAIN CODE #####
#####################

# Page Title
st.markdown("# Product")

# Page Description
st.markdown("This page will provide insight on how to reduce customer churn from a product standpoint using the prediction of the customer churn machine learning model which predicts if a customer will churn.")

# Page Instructions (How to Use This Page)
with st.expander("How to Use This Page"):
    # List of steps
    st.write('1. Load you own dataset or Use the provided dataset')
    st.write('2. View the model\'s predictions')
    st.write('3. Analyse the visualisations below to gain insights on how to reduce customer churn from a product standpoint')


# SECTION: INPUT DATA 
## File Upload section
st.markdown("## Input Data")
uploaded_files = st.file_uploader('Upload your file(s)', accept_multiple_files=True)
df=''
### If uploaded file is not empty
if uploaded_files!=[]:
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
    
# Preparing the data for prediction
## Removing Customer ID column
customer_id = df.pop("CUSTOMER_ID")
#Get categoorical columns
demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

df=pipeline(df)

with st.expander("Cleaned and Transformed Data"):
    st.write(df)
    
# MODEL FOR PREDICTION
model = joblib.load("assets/churn-prediction-model.jbl")
predictions= pd.DataFrame(model.predict(df),columns=['CHURNED'])
demo_df = pd.concat([demo_df, predictions], axis=1)
beha_df = pd.concat([beha_df, predictions], axis=1)

## ***table with custid and whether churn or not***
data=pd.concat([customer_id, predictions], axis=1) 

# replace '0' and '1' to 'not churned' and 'churned' respectively
data['CHURNED'] = data['CHURNED'].map({0: 'Not Churned', 1: 'Churned'})

# Set the display options to right-align the columns and adjust column width
pd.set_option('colheader_justify', 'right')
pd.set_option('display.max_colwidth', None)



# Display individual customer churned status and number of customers by churned status
# Multi-select feature also enabled for these 2 tables
multi_select_custid_individual(data)



# MENU TABLE #
## retrieve menu table
## manipulation to retrieve health metrics for each product
menu_table_df = retrieve_menu_table()
menu_table_df = get_health_metrics_menu_table(menu_table_df)
st.dataframe(menu_table_df, hide_index = True)


# ORDER DETAILS TABLE #
## retrieve order details table from Snowflake
## manipulation to retrieve desired layout for table
order_details_df = retrieve_order_details()
st.dataframe(order_details_df, hide_index = True)


# OVERALL TABLE #
## merge tables to get overall table
## re-arrange columns to get desired table
merged_df = get_overall_table(order_details_df, menu_table_df)

## Display header
st.markdown("## Overall Table")

## Display the merged DataFrame
st.dataframe(merged_df, width=0, hide_index=True)


# MENU ITEM TABLE #

final_product_df = menu_item_table()

## Display header
st.markdown("## Menu Item Table")

## Display the merged DataFrame
st.dataframe(final_product_df, hide_index=True)





# MENU ITEM CATEGORY TABLE #

## Display header
st.markdown("## Menu Item Category Table")

## Group by ITEM_CATEGORY and calculate the number of unique MENU_ITEM_ID
unique_products_per_category_df = menu_table_df.groupby('ITEM_CATEGORY')['MENU_ITEM_ID'].nunique().reset_index()

## Rename the columns for clarity
unique_products_per_category_df.columns = ['ITEM_CATEGORY', 'NO_OF_MENU_ITEMS']



## Pivot the DataFrame to get the count of each unique ITEM_SUBCATEGORY within each ITEM_CATEGORY
no_of_subcategory_within_category_df = pd.pivot_table(menu_table_df, index='ITEM_CATEGORY', columns='ITEM_SUBCATEGORY', values='MENU_ITEM_ID', aggfunc='count', fill_value=0)

## Reset the index to make 'ITEM_CATEGORY' a regular column
no_of_subcategory_within_category_df.reset_index(inplace=True)

## Rename the columns for clarity
no_of_subcategory_within_category_df.columns = ['ITEM_CATEGORY', 'NO_OF_COLD_OPTIONS', 'NO_OF_HOT_OPTIONS', 'NO_OF_WARM_OPTIONS']



##  Convert 'UNIT_PRICE' column to a numeric type
menu_table_df['UNIT_PRICE'] = menu_table_df['UNIT_PRICE'].astype(float)
    
## Group by ITEM_CATEGORY to get the average unit price for each category
avg_unit_price_per_category = menu_table_df.groupby('ITEM_CATEGORY')['UNIT_PRICE'].mean().reset_index()

## Rename the column for clarity
avg_unit_price_per_category.rename(columns={'UNIT_PRICE': 'AVG_UNIT_PRICE'}, inplace=True)

# round off sale price to 2dp
avg_unit_price_per_category['AVG_UNIT_PRICE'] = avg_unit_price_per_category['AVG_UNIT_PRICE'].apply(lambda x: '{:.2f}'.format(x))



## Group by ITEM_CATEGORY to get the average quantity sold for each category
avg_qty_sold_per_category = merged_df.groupby('ITEM_CATEGORY')['QUANTITY'].mean().reset_index()

## Rename the column for clarity
avg_qty_sold_per_category.rename(columns={'QUANTITY': 'AVG_QTY_SOLD'}, inplace=True)



## Group by 'MENU_ITEM_ID' and calculate the average 'PRODUCT_TOTAL'
avg_spending_per_category = merged_df.groupby('ITEM_CATEGORY')['PRODUCT_TOTAL'].mean().reset_index()

## Rename the column for clarity
avg_spending_per_category.rename(columns={'PRODUCT_TOTAL': 'AVG_SPENDING'}, inplace=True)



## Get the count of each menu_item_id in merged_df
menu_item_category_counts = merged_df['ITEM_CATEGORY'].value_counts().reset_index()

## Rename the columns for clarity
menu_item_category_counts.columns = ['ITEM_CATEGORY', 'TOTAL NO. OF TRANSACTIONS']





## Merge the two DataFrames on 'ITEM_CATEGORY'
menu_item_cat_merged_df = pd.merge(unique_products_per_category_df, no_of_subcategory_within_category_df, on='ITEM_CATEGORY', how='outer')

## Merge avg_unit_price_per_category with menu_item_cat_merged_df
menu_item_cat_merged_df = pd.merge(menu_item_cat_merged_df, avg_unit_price_per_category, on='ITEM_CATEGORY', how='outer')

## Merge avg_qty_sold_per_category with menu_item_cat_merged_df
menu_item_cat_merged_df = pd.merge(menu_item_cat_merged_df, avg_qty_sold_per_category, on='ITEM_CATEGORY', how='outer')

## Merge avg_spending_per_category with menu_item_cat_merged_df
menu_item_cat_merged_df = pd.merge(menu_item_cat_merged_df, avg_spending_per_category, on='ITEM_CATEGORY', how='outer')

## Merge menu_item_category_counts with menu_item_cat_merged_df
menu_item_cat_merged_df = pd.merge(menu_item_cat_merged_df, menu_item_category_counts, on='ITEM_CATEGORY', how='outer')

## round off sale price to 2dp
menu_item_cat_merged_df['AVG_SPENDING'] = menu_item_cat_merged_df['AVG_SPENDING'].apply(lambda x: '{:.2f}'.format(x))

## Display the merged DataFrame
st.dataframe(menu_item_cat_merged_df, width=0, hide_index=True)










# EXPORT DATA OPTION #
st.header('Export data to .csv')
st.write("Click the button below to export the overall table to csv format")
csv = convert_df_to_csv(merged_df)
st.download_button(
"Download",
csv,
"Tasty Insights - Product Team.csv",
"text/csv",
key='download-csv'
)