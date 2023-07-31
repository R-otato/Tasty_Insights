# Import statements--#
import streamlit as st
import pandas as pd
import joblib
import snowflake.connector

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
    my_cur.execute("select MENU_ITEM_ID, MENU_ITEM_NAME from menu")
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

# SEARCH FOR BUNDLES
st.markdown("## Search for Bundles")

# SECTION: INPUT DATA 
## File Upload section
st.markdown("### Input Data")
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
    st.info("Using the last updated data of all the United States transactions. Upload a file above to use your own data!")
    df=pd.read_csv('assets/apriori_dataset.csv')

st.write(df)

# Retrieve bundles found by apriori
my_rules = get_item_bundles_unformatted(df)


# Show identified bundles
st.markdown("### Identified Bundles")

## retrieve bundle dataframe
final_bundles_df = display_bundles(my_rules)

## print the new DataFrame with the desired structure
st.dataframe(final_bundles_df, hide_index=True)







# PRODUCT PERFORMANCE PREDICTION
st.markdown("## Product Performance")

# load product qty regression model
product_qty_model = joblib.load("assets/product_qty_regression.joblib")