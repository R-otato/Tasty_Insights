#--Import statements--
import streamlit as st
import pandas as pd
import joblib 
import snowflake.connector

#--Functions--

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





#################
### MAIN CODE ### 
#################

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

# ***table with custid and whether churn or not***
data=pd.concat([customer_id, predictions], axis=1) 

# filter data for only those who churn
data = data[data['CHURNED'] == 1]

# show model result for churned customers only
st.dataframe(data, hide_index = True)


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
    query = f"SELECT MENU_ITEM_ID, CUSTOMER_ID, FAVOURITE_BRAND, UNIT_PRICE, ORDER_AMOUNT FROM order_details_usa_matched WHERE CUSTOMER_ID IN ({customer_ids_str})"

    # Execute the SQL query for the current chunk
    my_cur.execute(query)

    # Fetch the result for the current chunk
    chunk_result = my_cur.fetchall()

    # Append the chunk result to the overall result
    order_details.extend(chunk_result)

# Create a DataFrame from the fetched result
order_details_df = pd.DataFrame(order_details, columns=['MENU_ITEM_ID', 'CUSTOMER_ID', 'FAVOURITE_BRAND', 'UNIT_PRICE', 'ORDER_AMOUNT'])

st.write(order_details_df)




# # convert SQL code to break down based on input
# my_cur.execute(f"SELECT * FROM order_details_usa_matched WHERE CUSTOMER_ID IN ({customer_ids_str})")
# order_details = my_cur.fetchall()

# st.dataframe(order_details)
# order_details_df = pd.DataFrame(order_details, columns = ['MENU_ID', 'MENU_ITEM_ID', 
#                                                           'CUSTOMER_ID', 'FAVOURITE_BRAND', 
#                                                           'UNIT PRICE', 'ORDER_AMOUNT'])
# st.dataframe(order_details_df, hide_index = True)
# st.write(order_details_df)