# Import statements--#
import streamlit as st
import pandas as pd
import joblib
import snowflake.connector
import ast
import numpy as np
from PIL import Image
import plotly.graph_objects as go

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

# Function: retrieve_order_detail_table()
# the purpose of this function is to retrieve the order details for USA from Snowflake to merge with the menu column to get total sales for a current menu type
def retrieve_order_detail_table():
    ## get connection to snowflake
    my_cnx = snowflake.connector.connect(
        user = "RLIAM",
        password = "Cats2004",
        account = "LGHJQKA-DJ92750",
        role = "TASTY_BI",
        warehouse = "TASTY_BI_WH",
        database = "frostbyte_tasty_bytes",
        schema = "analytics"
    )

    # retrieve menu table from snowflake
    my_cur = my_cnx.cursor()
    my_cur.execute("select MENU_ITEM_ID, PRICE, ORDER_TS, QUANTITY from ORDER_DETAILS_USA_MATCHED")
    order_table = my_cur.fetchall()
    
    order_table_pandas = pd.DataFrame(order_table, columns=['MENU_ITEM_ID', 'PRICE', 'ORDER_TS', 'QUANTITY'])
    
    return order_table_pandas


#####################
##### MAIN CODE #####
#####################

st.markdown("# Product Team")
tab1, tab2, tab3 = st.tabs(['About', 'Model Prediction', 'New Model'])

# TAB 1: About
with tab1:
    # High Level Goals Explanations
    st.markdown("# High Level Goals")
    st.write("""Our solution, Tasty Insights, is dedicated to assisting Tasty Bytes in achieving its high-level goals over the next 5 years. 
             More specifically, to remarkable 25% Year-Over-Year sales increase, from annual sales of \$105M to \$320M. This page is designed 
             to meet the needs of the product team, to help them with menu optimisation. """)



    # Utilisation of model's prediction
    st.markdown("# How to utilise predictions?")
    st.write("""The model's output will show the predicted total quantity sold for the new menu item as well as its respective sales, profits and gross
             and net profit margins. In addition, it will show the unit item details in a expander. The image below shows what the model prediction will be
             displayed.""")
    st.write('')
    st.image(Image.open('assets/Product Qty Prediction Outcome.png'))
    st.caption("Prediction display after model makes prediction")

    st.write("")
    
    st.write("""
             The model's prediction can provide insights on:
             - Risk Assessment: Assess the potential risk associated with introducing a new menu item. Understand the level of uncertainty and take 
             proactive measures to mitigate risks if the predicted quantity is lower than desired
             - Demand Forecasting: Estimate the expected demand for the new menu item. Plan inventory levels, production schedules, and supply chain 
             logistics more effectively, minimising the risk of overstocking or understocking
             - Optimised Menu Planning: Assess the potential success of introducing new menu items. Focus on items with projected to have higher sales and 
             ensure that the menu aligns with customer preferences and demands.
             - Pricing Strategies: Determine appropriate pricing strategies. Optimise prices to achieve a balance between maximizing revenue and 
             maintaining customer satisfaction and loyalty.
             """)



    # Interpretability of the model
    st.markdown("# Interpreting the model")
    st.write("""The model's feature importance analysis provides valuable insights into the factors that significantly influence the prediction of total 
             quantity sold for new menu items.""")
    st.image(Image.open('assets/Product Qty Model Feature Importance.png'))
    st.caption("The image shows the feature importance in order of its importance, starting with the most important feature.")
    st.write("""Among the features considered, "ITEM_CATEGORY_Beverage" emerges as the most influential, with a high importance value of 0.626. This suggests
             that the choice of menu item category, particularly beverages, plays a crucial role in determining the sales performance of new products. 
             Additionally, "SALE_PRICE_USD" follows as the second most important feature with an importance value of 0.224, indicating that the sale price
             of the item strongly affects its quantity sold. Further down the list, we observe other menu types, such as "MENU_TYPE_Tacos" and 
             "MENU_TYPE_BBQ," showing moderate importance values, suggesting they also have some impact on sales performance. Conversely, some features 
             like "TRUCK_BRAND_NAME_Revenge of the Curds" and "ITEM_CATEGORY_Snack" have negligible importance, implying that they have minimal influence 
             on the total quantity sold. Overall, this feature importance analysis aids the product team in understanding which attributes have the most 
             significant impact on product sales, enabling them to make data-driven decisions to optimize menu planning and maximize revenue.""")
    
    
    # Limitations and Assumptions the model makes
    st.markdown("# Limitations and assumptions the model makes")
    st.markdown("## Assumptions")
    st.write("""The first assumption the model makes is that the total quantity sold from the order details data as the total lifetime quantity 
             sold of a product which may not be the case for every product considering the fact that the launch date and removal date is not provided in 
             the dataset.""")
    
    st.write("""The second assumption the model makes is the independence of the observations in the training data. In other words, the model assumes that 
             each menu item's sales data is not influenced by other menu items' sales or external factors.""")
    
    st.write("""The last assumption the model might make is the linear relationships between certain features and the target variable (quantity sold). This
             assumption may not hold in some cases, especially if there are complex interactions or non-linear patterns in the data.""")
    
    st.markdown("## Limitations")
    st.write("""The first limitation of the model is that the model does not account for seasonality or temporal trends in sales. The model assumes 
             that sales patterns are consistent over time, which might not hold true for certain menu items affected by seasonal demand or changing trends.""")
    
    st.write("""The second limitation of the model is that the model does not consider external factors such as changes in customer preferences, market 
             trends, economic conditions, or marketing efforts that might influence sales. These factors can significantly impact the quantity sold but 
             are not explicitly captured in the model.""")
    
    
    
    # Model's Confidence Level
    st.markdown("# Model's confidence level")
    st.image(Image.open('assets/Product Qty Model Performance.png'))
    st.caption("The image below shows the performance of the model.")
    st.write("""The root mean squared error (RMSE) measures the average prediction error, where a lower value indicates better predictive performance. 
             In this case, the training RMSE is 42.18, and the testing RMSE is 96.78. The relatively low RMSE values demonstrate that the model's 
             predictions are close to the actual values, both during training and when dealing with unseen data.""")
    st.write("""The mean squared error (MSE) provides another perspective on prediction accuracy, and lower values are preferable. The training MSE is 
             1779.03, and the testing MSE is 9366.61, reinforcing the notion that the model achieves good accuracy on both training and testing datasets.""")
    st.write("""The accuracy of the regression model is another critical metric. While accuracy is more commonly associated with classification tasks, it 
             can be interpreted here as a measure of how well the model captures the variance in the data. The training accuracy is 99.88%, and the testing
             accuracy is 99.33%. These high accuracy values indicate that the model captures a significant portion of the data's variability, demonstrating
             its effectiveness in predicting the total quantity sold.""")
    st.write("""In summary, the regression model performs exceptionally well in making predictions for new menu items' total quantity sold. The low RMSE 
             and MSE values indicate that the model's predictions are close to the actual values, while the high accuracy values suggest that the model 
             captures a substantial portion of the data's variability. The product team can be confident in relying on this model to make data driven 
             decisions about menu planning and optimising sales performance.""")
    
    
    
# TAB 2: Model Prediction   
with tab2:
    # Page Instructions (How to Use This Page)
    with st.expander("How to Use This Page"):
        # List of steps
        st.write('1. Load you own dataset or Use the provided dataset')
        st.write('2. View the model\'s predictions')
        st.write('3. Analyse the visualisations below to gain insights on how to reduce customer churn from a product standpoint')

    ## retrieve menu table with health metrics in different columns
    menu_table_df = retrieve_menu_table()
    menu_table = get_health_metrics_menu_table(menu_table_df)

    # display current menu items
    with st.expander("Current Menu Items"):
        st.dataframe(menu_table, hide_index=True)


    # PRODUCT PERFORMANCE PREDICTION
    st.markdown("## Product Performance")
    
    user_input_df, sale_price, cost_of_goods, menu_type = user_inputs()
        
    # display dataframe
    st.dataframe(user_input_df, hide_index=True)


    # Check for null values in the user_input_df
    has_null_values = user_input_df.isnull().any().any()

    if has_null_values == False:
        # display message if no null values are found
        st.write("Proceed to make a prediction.")
        
        # Make a prediction
        if st.button("Predict"):
            total_product_details_df, new_product_details_df, rounded_prediction = prediction(user_input_df)
            
            order_data = retrieve_order_detail_table()
            menu_table_required = menu_table[["MENU_ITEM_ID", "MENU_TYPE"]]
            
            # Merge the two DataFrames based on the 'MENU_ITEM_ID' column
            required_data_df = pd.merge(order_data, menu_table_required, on='MENU_ITEM_ID')

            # Group the data by 'MENU_TYPE' and calculate the total sales (sum of 'PRICE') for each menu type
            total_sales_by_menu_type = required_data_df.groupby('MENU_TYPE')['PRICE'].sum().reset_index()
            
            total_sales_for_menu_type = total_sales_by_menu_type[total_sales_by_menu_type["MENU_TYPE"] == menu_type]

            total_sales_for_menu_type = float(total_sales_for_menu_type['PRICE'])
            
            new_item_sales = total_product_details_df["TOTAL_SALES"]
            new_item_sales = float(new_item_sales)

            # Calculate the percentage increase in sales
            percentage_increase = (new_item_sales / total_sales_for_menu_type) * 100
                
            st.markdown("### Prediction")
            ## display the rounded prediction
            st.markdown("##### Predicted Total Quantity Sold: {}".format(rounded_prediction))
            st.markdown(f"##### Percentage Increase in Sales: {percentage_increase:.2f}%")
            
            # calculate the total sales after adding the new menu item
            total_sales_for_menu_type_after = total_sales_for_menu_type + new_item_sales

            # create a DataFrame to hold the data for the bar chart
            data = {
                'Sales': ['Without New Item', 'With New Item'],
                'Total Sales': [total_sales_for_menu_type, total_sales_for_menu_type_after]
            }
            df = pd.DataFrame(data)

            # create the Plotly bar chart
            fig = go.Figure(data=[go.Bar(x=df['Sales'], y=df['Total Sales'])])

            # set the title
            fig.update_layout(title_text='Total Sales Before and After Adding New Menu Item', title_x=0.23)

            # remove y-axis ticks and labels
            fig.update_yaxes(showticklabels=False, showgrid=False)

            # add data labels to the bars
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside', textfont=dict(size=12))

            # show the plot using Streamlit's st.plotly_chart function
            st.plotly_chart(fig)
            
            
            st.write('')
            
            st.markdown("##### Total Item Details:")
            ## display the total_product_details_df DataFrame
            st.dataframe(total_product_details_df, hide_index=True)
            
            # display current menu items
            with st.expander("Unit Item Details"):
                st.write("This table contains details specific to a single unit or item of the new product")
                ## display the new_product_details_df DataFrame
                st.dataframe(new_product_details_df, hide_index=True)
    else:
        st.error("Please fill in all required fields before proceeding with the prediction.")
        

with tab3:
    st.markdown("## Menu Item Next Month Sales Prediction")
    
    default_option = None
    
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
        order_df = retrieve_order_detail_table()
        order_df['YEAR'] = order_df['ORDER_TS'].dt.year
        order_df['MONTH'] = order_df['ORDER_TS'].dt.month
        
        # Group order total to truck id
        total_qty_by_item = order_df.groupby(['YEAR', 'MONTH', 'MENU_ITEM_ID'])['QUANTITY'].sum().reset_index()

        # Renaming the 'ORDER_TOTAL' column to 'TOTAL_SALES_PER_MONTH'
        total_qty_by_item = total_qty_by_item.rename(columns={'QUANTITY': 'TOTAL_QTY_SOLD_PER_MONTH'})

        # Convert the 'YEAR' column to numeric values
        total_qty_by_item['YEAR'] = total_qty_by_item['YEAR'].astype(str).replace(',', '').astype(int)
        
        
        
        # get the highest year and month
        max_year_month = total_qty_by_item.groupby('MENU_ITEM_ID')[['YEAR', 'MONTH']].max().reset_index()

        menu_item_max_year_month = max_year_month[max_year_month["MENU_ITEM_ID"]==menu_item_id]

        total_qty_by_item_over_time = total_qty_by_item[total_qty_by_item["MENU_ITEM_ID"]==menu_item_id]

        # Plotly Line Chart
        ## create the line chart
        fig = go.Figure(data=go.Line(x=total_qty_by_item_over_time['MONTH'], y=total_qty_by_item_over_time['TOTAL_QTY_SOLD_PER_MONTH'], mode='lines+markers'))

        ## update the layout
        fig.update_layout(title='Total Quantity Sold per Month',
                        xaxis_title='Month',
                        yaxis_title='Total Qty Sold')

        ## show the plot in the Streamlit app 
        st.plotly_chart(fig)


        # Form month and year column for prediction
        ## if month is less than or equal to 11 then plus 1
        if int(menu_item_max_year_month["MONTH"])<=11:
            month = int(menu_item_max_year_month["MONTH"]) + 1
            year = int(menu_item_max_year_month["YEAR"])
        ## if month is equal to 12 then month will be 1 and year plus 1
        elif int(menu_item_max_year_month["MONTH"])== 12:
            month = 1
            year = int(menu_item_max_year_month["YEAR"]) + 1
        
        
        
        # Replace 'Y' with 'Yes' and 'N' with 'No' in the DataFrame
        item_info_df = item_info_df.replace({'Yes': 1, 'No': 0})
        
        
        
        # MANUAL ONT HOT ENCODING
        
        ## state cat cols to carry out manual encoding on
        categorical_cols = ["MENU_TYPE", "TRUCK_BRAND_NAME", "ITEM_CATEGORY", "ITEM_SUBCATEGORY"]
        
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
        
        
        
        ## assign the columsn YEAR and MONTH with their respective values
        item_info_df['YEAR'] = year
        item_info_df['MONTH'] = month
        
        # define the desired column order
        desired_columns = ['MENU_ITEM_ID', 'SALE_PRICE_USD', 'YEAR', 'MONTH', 'DAIRY_FREE',
                        'GLUTEN_FREE', 'HEALTHY', 'NUT_FREE', 'MENU_TYPE_BBQ',
                        'MENU_TYPE_Ramen', 'MENU_TYPE_Grilled Cheese', 'MENU_TYPE_Poutine',
                        'MENU_TYPE_Ethiopian', 'MENU_TYPE_Mac & Cheese', 'MENU_TYPE_Sandwiches',
                        'MENU_TYPE_Indian', 'MENU_TYPE_Gyros', 'MENU_TYPE_Hot Dogs',
                        'MENU_TYPE_Tacos', 'MENU_TYPE_Chinese', 'MENU_TYPE_Crepes',
                        'MENU_TYPE_Ice Cream', 'TRUCK_BRAND_NAME_Smoky BBQ',
                        'TRUCK_BRAND_NAME_Kitakata Ramen Bar', 'TRUCK_BRAND_NAME_The Mega Melt',
                        'TRUCK_BRAND_NAME_Revenge of the Curds', 'TRUCK_BRAND_NAME_Tasty Tibs',
                        'TRUCK_BRAND_NAME_The Mac Shack', 'TRUCK_BRAND_NAME_Better Off Bread',
                        'TRUCK_BRAND_NAME_Nani\'s Kitchen', 'TRUCK_BRAND_NAME_Cheeky Greek',
                        'TRUCK_BRAND_NAME_Amped Up Franks', 'TRUCK_BRAND_NAME_Guac n\' Roll',
                        'TRUCK_BRAND_NAME_Peking Truck', 'TRUCK_BRAND_NAME_Le Coin des Crêpes',
                        'TRUCK_BRAND_NAME_Freezing Point', 'ITEM_CATEGORY_Beverage',
                        'ITEM_CATEGORY_Main', 'ITEM_CATEGORY_Snack',
                        'ITEM_SUBCATEGORY_Cold Option', 'ITEM_SUBCATEGORY_Hot Option']

        # drop columns not in the desired column list
        item_info_df = item_info_df[desired_columns]

        # convert SALE_PRICE_USD column value to float
        item_info_df["SALE_PRICE_USD"] = item_info_df["SALE_PRICE_USD"].astype(float)

        
        
        # retrieve min max scaler
        min_max_scaler = joblib.load("assets/product_team_min_max_scaler.joblib")
        
        min_max_scaler.fit(item_info_df)
        
        min_max_scaler.transform(item_info_df)
        
        
        # retrieve regression model
        product_qty_per_month_model = joblib.load("assets/product_qty_per_month_model.joblib")
        
        model_prediction = product_qty_per_month_model.predict(item_info_df)
        
        
        # Round off the prediction to the nearest whole number
        rounded_prediction = round(model_prediction[0])
        
        unit_price = menu_table.loc[menu_table['MENU_ITEM_ID'] == menu_item_id, 'UNIT_PRICE'].values[0]
        sales_next_month = float(unit_price) * int(rounded_prediction)
        
        st.markdown("## Prediction:")
        st.markdown("### No. of {} sold next month: {}".format(menu_item_name, rounded_prediction))
        st.markdown("### Estimated sales next month: ${:.2f}".format(sales_next_month))
        
        
        st.write(menu_table_df)
        st.write(unit_price)