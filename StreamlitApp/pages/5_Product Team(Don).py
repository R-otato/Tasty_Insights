# Import statements--#
import streamlit as st
import pandas as pd
import joblib


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
    st.info("Using the last updated data of all the United States transactions. Upload a file above to use your own data!")
    df=pd.read_csv('assets/apriori_dataset.csv')

st.write(df)


# retrieve bundles found by apriori
my_rules = get_item_bundles_unformatted(df)

st.write(my_rules)






# PRODUCT PERFORMANCE PREDICTION
st.markdown("## Product Performance")

# load product qty regression model
product_qty_model = joblib.load("assets/product_qty_regression.joblib")