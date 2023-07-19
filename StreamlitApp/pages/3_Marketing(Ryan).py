#--Import statements--
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from cachetools import cached
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


#################
### Functions ### 
#################

#Note to teammates: Copy the functions below if you want to transform the data to perform Kmeans clustering and Churn prediction
#Else if you want to perform Churn prediction just edit the code by removing data_kmeans.

def pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Removing Customer ID column
    customer_id = data.pop("CUSTOMER_ID")

    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/windsorizer_gau.jbl")
    yjt = joblib.load("assets/yjt.jbl")
    ohe_enc = joblib.load("assets/ohe_enc.jbl")
    minMaxScaler = joblib.load("assets/minMaxScaler.jbl")
    kmeansMinMaxScaler=joblib.load("assets/kmeans_scaling.jbl")

    # Apply the transformations to the data
    #Both models table transformation
    data = windsorizer_iqr.transform(data)  # Apply IQR Windsorization
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization

    #KMeans table tranformation
    cols_to_scale=['RECENCY','FREQUENCY','MONETARY','AGE','AVG_DAYS_BETWEEN_PURCHASE', 'LENGTH_OF_RELATIONSHIP']
    data_kmeans=data[cols_to_scale].copy() # For our Kmeans model, it does not include any yeo johnson transformation
    data_kmeans[cols_to_scale] = kmeansMinMaxScaler.transform(data_kmeans[cols_to_scale])  # Apply Min-Max Scaling for Kmeans

    #Churn prediction table transformation
    data = yjt.transform(data)  # Apply Yeo-Johnson Transformation
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data.columns = data.columns.str.upper() #Normalize naming convention
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    #Concat Customer ID back
    data=pd.concat([customer_id, data], axis=1)
    data_kmeans=pd.concat([customer_id, data_kmeans], axis=1)

    return data,data_kmeans

#Load model - To teammates: you can just copy this entirely
def load_model(model_path: str) -> object:
    model = joblib.load(model_path)
    return model    

#Convert dataframe to csv 
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

#################
### MAIN CODE ### 
#################
#To teammates: Try not to copy my format entirely 

def main() -> None:
    # Page title
    st.markdown("# Marketing")

    # How to use this page
    with st.expander("How to Use This Page"):
        #Going to add some stuff here 
        st.write('How to Use This Page')

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

    ## Display uploaded or defaul file
    with st.expander("Raw Dataframe"):
        st.write(df.head(10))

    # #Get categoorical columns
    # demo_df=df[['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE']]
    # beha_df=df.loc[:, ~df.columns.isin(['GENDER','MARITAL_STATUS','CITY','CHILDREN_COUNT','AGE'])]

    clean_df,kmeans_df=pipeline(df)

    with st.expander("Cleaned and Transformed Data"):
        st.write(clean_df.head(10))

    # Setup: Model loading, predictions and combining the data
    churn_model = load_model("assets/churn-prediction-model.jbl")
    seg_model = load_model("assets/kmeans.jbl")
    kmeans_pred=pd.DataFrame(seg_model.predict(kmeans_df.drop('CUSTOMER_ID',axis=1)),columns=['CLUSTER'])
    churn_pred= pd.DataFrame(churn_model.predict(clean_df.drop('CUSTOMER_ID',axis=1)),columns=['CHURNED'])
    # demo_df = pd.concat([demo_df, predictions], axis=1)
    # beha_df = pd.concat([beha_df, predictions], axis=1)

    data=pd.concat([df,kmeans_pred],axis=1)
    data=pd.concat([data, churn_pred], axis=1)

    # Display predictions
    st.markdown("## Customer Segmentation")
    st.dataframe(data.value_counts('CLUSTER'))

    st.markdown("## Churn Prediction")
    st.dataframe(data.value_counts('CHURNED'))

    st.markdown("## Overall Table")
    filtered_data=filter_dataframe(data)
    st.dataframe(filtered_data)

    #Allow user to download dataframe for further analysis
    st.header('**Export results ✨**')
    st.write("_Finally you can export the resulting table after Clustering and Churn Prediction._")
    csv = convert_df(filtered_data)
    st.download_button(
    "Press to Download",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

 
###########################
### Page Configurations ### 
###########################
if __name__ == "__main__":
    # Setting page configuration
    st.set_page_config(
        "Tasty Bytes Marketing by Ryan Liam",
        "📊",
        #initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
