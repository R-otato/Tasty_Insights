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
from PIL import Image
import math
from datetime import datetime


#################
### Functions ### 
#################

#Note to teammates: Copy the functions below if you want to transform the data to perform Kmeans clustering and Churn prediction
#Else if you want to perform Churn prediction just edit the code by removing data_kmeans.
def kmeans_pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Removing columns not used in transformation
    cols_Not_Involved=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)','AVG_SALES_ORDER','TENURE_MONTHS']
    not_Involved=data[cols_Not_Involved]
    data.drop(cols_Not_Involved,axis=1,inplace=True,errors='ignore')

    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/windsorizer_gau.jbl")
    kmeansMinMaxScaler=joblib.load("assets/kmeans_scaling.jbl")

    # Apply the transformations to the data
    #Both models table transformation
    data = windsorizer_iqr.transform(data)  # Apply IQR Windsorization
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization
    cols_to_scale=['RECENCY','FREQUENCY','MONETARY']
    data[cols_to_scale] = kmeansMinMaxScaler.transform(data[cols_to_scale])  # Apply Min-Max Scaling for Kmeans

    #Concat Customer ID back
    data=pd.concat([not_Involved, data], axis=1)

    return data


def churn_pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Removing columns not used in transformation
    cols_Not_Involved=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)','AVG_SALES_ORDER','TENURE_MONTHS']
    not_Involved=data[cols_Not_Involved]
    data.drop(cols_Not_Involved,axis=1,inplace=True,errors='ignore')

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
    data.columns = data.columns.str.upper() #Normalize naming convention
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    #Concat Customer ID back
    data=pd.concat([not_Involved, data], axis=1)

    return data

def sales_pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Filter columns
    not_Involved=data[['CUSTOMER_ID','MONETARY']]
    data=data[['CUSTOMER_ID','MONETARY','FREQUENCY','AVG_SALES_ORDER','TENURE_MONTHS']]

    # Load the necessary transformations
    windsorizer_gau = joblib.load("assets/memb_sales_win_gau.jbl")
    minMaxScaler = joblib.load("assets/memb_sales_scale.jbl")

    # Apply the transformations to the data
    data = windsorizer_gau.transform(data)  # Apply Gaussian Windsorization
    data[['FREQUENCY','AVG_SALES_ORDER','TENURE_MONTHS']] = minMaxScaler.transform(data[['FREQUENCY','AVG_SALES_ORDER','TENURE_MONTHS']])  # Apply Min-Max Scaling

    return data


#Load model - To teammates: you can just copy this entirely
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

# Define a function to validate if the input is an integer
def validate_integer_input(input_str):
    try:
        return int(input_str)
    except ValueError:
        return None

#################
### MAIN CODE ### 
#################
#To teammates: Try not to copy my format entirely 

def main() -> None:
    # Page title
    st.markdown("# Marketing") 
    tab1, tab2 = st.tabs(['About', 'Model Prediction'])
    with tab1:
        # High level goals
        st.markdown("## High Level Goals")
        st.write("""As stated in our homepage, our team is dedicated to assisting Tasty Bytes in achieving its ambitious goals over the next 5 years. 
                 In particular, we aim to help Tasty Bytes achieve a remarkable 25% Year-Over-Year increase in sales. This page is exclusively focused 
                 on churn prediction, which is a twin concept of member retention. It is designed to empower and elevate your marketing strategies with 
                 our data-driven approach, ultimately driving significant sales growth by retaining valuable customers and understanding their likelihood of churning.""")
        # How to use predictions
        st.markdown('## How to Utilize the Predictions')
        st.write(
            """
            In the Model Prediction tab, you will have access to valuable insights derived from both the Segmentation and Churn prediction models. 
            Additionally, we have incorporated a Sales Prediction model that predicts customer sales based on Frequency, Average Order Sales, and 
            Tenure Month. Once you filter the segments and churn predictions, you can input your own frequency and months data to get personalized sales predictions.

            With these powerful predictions, you can unlock various opportunities:

            - Explore Customer Segments: Dive into the different segments within your member base, understanding their purchasing behavior.

            - Identify Churn Likelihood: Gain visibility into which members are likely to churn or remain engaged, predicting their purchase behavior in the next 14 days.

            - Targeted Marketing Strategies: Armed with these predictions, you can design targeted marketing schemes tailored to specific segments or groups of customers, maximizing the impact of your campaigns.

            - Personalized Sales Predictions: Input the desired number of months and the expected number of purchases for each member in that period to assume the impact of your marketing strategy and generate forecasted sales.

            Leveraging these data-driven insights, your marketing team can make informed decisions, optimize marketing efforts, and drive sales growth for Tasty Bytes. Let's unlock the full potential of your marketing strategies together!
            """
        )

        # Models confidence
        st.markdown("## Model's Confidence")


    with tab2:
        with st.expander("How to Use This Page"):
            #Going to add some stuff here 
            st.write("""
            1. Default Data: The page displays default data for members in the United States. To update member information,
            upload a new Excel file in the Input Data section.
            2. Predictions: After uploading your file, the predictions will be automatically generated and shown.
            3. Filter Data: Use filters to explore specific segments or refine the data for analysis.
            4. Download Predictions: Download the predictions along with relevant data for further analysis, if desired.
            """)

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
            df=pd.read_csv('assets/marketing.csv')

        ## Display uploaded or defaul file
        with st.expander("Uploaded/Default Data"):
            st.write(df)

        # Run pipeline
        clean_df=churn_pipeline(df)
        kmeans_df=kmeans_pipeline(df)

        # Setup: Model loading
        churn_model = load_model("assets/churn-prediction-model.jbl")
        seg_clf_model = load_model("assets/models/segment_classifier.jbl")
        sales_model=load_model("assets/models/memb_sales_pred.jbl")

        # Setup: Get predictions
        cols_to_ignore=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)','AVG_SALES_ORDER','TENURE_MONTHS']
        kmeans_cols=['RECENCY','FREQUENCY','MONETARY']
        churn_pred= pd.DataFrame(churn_model.predict(clean_df.drop(cols_to_ignore,axis=1,errors='ignore')),columns=['CHURNED'])
        cluster_pred=pd.DataFrame(seg_clf_model.predict(kmeans_df[kmeans_cols]),columns=['CLUSTER'])
        
        # Setup: Map predictions to understandable insights
        churn_pred['CHURNED'] = churn_pred['CHURNED'].map({0: 'Not Churned', 1: 'Churned'})
        cluster_pred['CLUSTER'] =cluster_pred['CLUSTER'].map({
        0: "Active Moderate-Value Members",
        1: "Inactive Low-Spending Members",
        2: "High-Value Loyal Members",
        3: "Engaged Moderate-Value Members",
        4: "Active Low-Spending Members"})

        # Setup:Combine tables with predictions
        data=pd.concat([df,cluster_pred],axis=1)
        data=pd.concat([data, churn_pred], axis=1)
        
        # Display predictions
        st.markdown("## Member Segmentation and Churn Prediction Results")

        # Display a filter for selecting clusters
        cluster_Options = ['All'] + data['CLUSTER'].unique().tolist()
        selected_Cluster = st.multiselect("Filter by Member's Segment:", cluster_Options, default=['All'])
        filtered_data=filter(selected_Cluster,'CLUSTER',data)

        churn_Options = ['All'] + data['CHURNED'].unique().tolist()
        selected_Churn= st.multiselect("Filter by Churn:",churn_Options, default=['All'])
        filtered_data=filter(selected_Churn,'CHURNED',filtered_data)
        
        # Number of members of each cluster
        cluster_counts = filtered_data.groupby('CLUSTER').size().reset_index(name='Number of Members')
        st.dataframe(cluster_counts, hide_index=True)

        # Number of members who churned and not churned
        churn_counts = filtered_data.groupby('CHURNED').size().reset_index(name='Number of Members')
        st.dataframe(churn_counts, hide_index=True)

        # Display forecasted sales
        st.write('### Member Forecasted Sales')
        #Get user input
        forecast_months = st.slider('Months for forecast:', 1, 12, 1)

        # Get the input from the user
        expected_purchases_input = st.text_input("Expected purchases in period:", "1")

        # Validate the input
        estimated_frequency = validate_integer_input(expected_purchases_input)

        # Display error message if input is not an integer
        if estimated_frequency is None:
            st.error("Please enter a valid integer for the expected purchases.")
        else:
            #Setup data
            sales_model_input = filtered_data[['CUSTOMER_ID', 'FREQUENCY', 'AVG_SALES_ORDER', 'TENURE_MONTHS','MONETARY']]
            sales_model_input['FREQUENCY'] = sales_model_input['FREQUENCY'] + estimated_frequency
            sales_model_input['TENURE_MONTHS'] = sales_model_input['TENURE_MONTHS'] + forecast_months
            #Transform data
            sales_clean=sales_pipeline(sales_model_input)
            monetary_pred= pd.DataFrame(sales_model.predict(sales_clean.drop(['CUSTOMER_ID','MONETARY'],axis=1,errors='ignore')),columns=['FORECAST_MONETARY'])
            #Prep output data
            sales_model_input['FORECAST_MONETARY']=round(monetary_pred['FORECAST_MONETARY'],2)
            sales_model_input['FORECAST_SALES']=round(sales_model_input['FORECAST_MONETARY']-sales_model_input['MONETARY'],2)
            # Display output data
            st.write(sales_model_input)
            st.metric('Forecasted Sales', f"${round(sales_model_input['FORECAST_SALES'].sum(),2)}")


    # # Metrics table
    # st.markdown("### Member's Metric")
    # ## Display table
    # metric_df=filtered_data[['CUSTOMER_ID','RECENCY','FREQUENCY','MONETARY','LENGTH_OF_RELATIONSHIP','PROFIT','PROFIT_MARGIN(%)']]
    # st.dataframe(metric_df, hide_index=True)


    # # Demographic table
    # st.markdown("### Member's Demographic")
    # ## Clean up columns
    # filtered_data['CHILDREN_COUNT'] = filtered_data['CHILDREN_COUNT'].map({
    # '0': "No",
    # '1': "Yes",
    # '2': "Yes",
    # '3': "Yes",
    # '4': "Yes",
    # '5+': "Yes",
    # 'Undisclosed':'Undisclosed'})
    # filtered_data.rename({'CHILDREN_COUNT':'HAVE_CHILDREN'},inplace=True,errors='ignore',axis=1)
    # ## Display table
    # demo_df=filtered_data[['CUSTOMER_ID','GENDER','MARITAL_STATUS','CITY','HAVE_CHILDREN','AGE']]
    # st.dataframe(demo_df, hide_index=True)

    # # Behavioral table
    # st.markdown("### Member's Behaviour")
    
    # ## Clean up columns
    # # # Convert 'LENGTH_OF_RELATIONSHIP' from days to years (with decimal point)
    # # filtered_data['LENGTH_OF_RELATIONSHIP_YEARS'] = filtered_data['LENGTH_OF_RELATIONSHIP'] / 365.25

    # # # Rename the new column to indicate it represents relationship duration in decimal years
    # # filtered_data.rename(columns={'LENGTH_OF_RELATIONSHIP': 'LENGTH_OF_RELATIONSHIP_DAYS'}, inplace=True)

    # ## Display table
    # beha_df=filtered_data[['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY']]
    # st.dataframe(beha_df, hide_index=True)

    # # Overall Table
    # st.markdown("### Overall Table")
    # st.dataframe(filtered_data, hide_index=True)

    # #Allow user to download dataframe for further analysis
    # st.header('**Export results ✨**')
    # st.write("_Finally you can export the resulting table after Clustering and Churn Prediction._")
    # csv = convert_df(filtered_data)
    # st.download_button(
    # "Press to Download",
    # csv,
    # "marketing.csv",
    # "text/csv",
    # key='download-csv'
    # )

 
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
