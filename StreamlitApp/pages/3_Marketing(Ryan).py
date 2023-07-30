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

def pipeline(data):
    ## Make a copy first
    data=data.copy()

    ## Removing columns not used in transformation
    cols_Not_Involved=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)']
    not_Involved=data[cols_Not_Involved]
    data.drop(cols_Not_Involved,axis=1,inplace=True)

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
    cols_to_scale=['RECENCY','FREQUENCY','MONETARY']
    data_kmeans=data[cols_to_scale].copy() # For our Kmeans model, it does not include any yeo johnson transformation
    data_kmeans[cols_to_scale] = kmeansMinMaxScaler.transform(data_kmeans[cols_to_scale])  # Apply Min-Max Scaling for Kmeans

    #Churn prediction table transformation
    data = yjt.transform(data)  # Apply Yeo-Johnson Transformation
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data.columns = data.columns.str.upper() #Normalize naming convention
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling
    
    #Concat Customer ID back
    data=pd.concat([not_Involved, data], axis=1)
    data_kmeans=pd.concat([not_Involved, data_kmeans], axis=1)

    return data,data_kmeans

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

#################
### MAIN CODE ### 
#################
#To teammates: Try not to copy my format entirely 

def main() -> None:
    # Page title
    st.markdown("# Marketing")
    tab1, tab2 = st.tabs(["High Level Goals", "Model Prediction"])
    with tab1:
        # High Level Goals
        st.markdown("## High Level Goals")
        st.write("""As stated in our homepage, our team is dedicated to assisting Tasty Bytes in achieving its ambitious goals over the next 5 years. 
                 In particular, we aim to assist helping Tasty Bytes achieve a remarkable 25% Year-Over-Year increase, with the objective of elevating 
                 annual sales from \$105M to an impressive \$320M. This page is designed exclusively for the marketing team, showcasing our data-driven 
                 approach to empower and elevate your marketing strategies, ultimately driving significant sales growth.""")
        
        st.markdown('### How to Utilize the Predictions')
        st.write(
        """
        In the Model Prediction tab, you will have access to valuable insights derived from both the Segmentation and Churn prediction models. These predictions empower you to:

        - Explore Customer Segments: Dive into the different segments within your member base, understanding their unique behaviors and preferences.
        - Identify Churn Likelihood: Gain visibility into which members are likely to churn or remain engaged, predicting their purchase behavior in the next 14 days.
        - Targeted Marketing Strategies: Armed with these predictions, you can design targeted marketing schemes tailored to specific segments or groups of customers, 
        maximizing the impact of your campaigns.

        Leveraging these data-driven insights, your marketing team can make informed decisions, optimize marketing efforts, and drive sales growth for Tasty Bytes. 
        Let's unlock the full potential of your marketing strategies together!
        """
        )

        with st.expander('Data Limitations'):
            st.write(
            """
            - Lack of Customer ID for non-members, limiting individual tracking and analysis for this group.
            - Absence of data on marketing campaigns, hindering the assessment of campaign effectiveness.
            - Missing information on Discounts and Order Channels, limiting insights into pricing strategies and sales channels.

            Despite these limitations, we will leverage available data and employ advanced analytics techniques to drive actionable insights and optimize marketing strategies. 
            Our data-driven approach aims to address the challenges and still provide valuable recommendations to support Tasty Bytes' growth and sales goals. 
            Through creativity and resourcefulness, we are confident in our ability to make a significant impact and help Tasty Bytes succeed in its endeavors.
            """
            )
        
        with st.expander("Value of Membership"):
            #Going to add some stuff here 
            st.image(Image.open('assets/beforeConversionToMember.png'))
            st.write(
            """
            The box plot provides valuable insights into the impact of membership. After customers become members, they exhibit a 
            remarkable increase in their purchase frequency from Tasty Bytes. This finding underscores the significance of membership 
            in fostering customer loyalty and driving repeat purchases. By focusing on members and offering tailored experiences, we 
            can further nurture customer engagement and strengthen our relationship with them. These efforts, in turn, contribute to 
            sustained sales growth and enhanced customer satisfaction.
            """
            )
            
        st.markdown('### Sales Forecasting for End of Year - All Members')
        st.write('*Estimated Sales=Average Spending of Customer * (Remaining Time Period/Selected number of days between regular purchase)')

        regular_purchase_days = st.slider('Select the number of days between regular purchases', 1, 30, 14)
        avg_Spending=pd.read_csv('assets/datasets/average_spending_members.csv')
        last_date_datetime = datetime.strptime('2022-11-02', "%Y-%m-%d")
        # Get the end of the year for the same year (2022)
        end_of_year_datetime = datetime(last_date_datetime.year, 12, 31)

        # Calculate the difference (timedelta) between the two dates
        time_difference = end_of_year_datetime - last_date_datetime

        # Extract the number of days from the timedelta
        days_to_end_of_year = time_difference.days
        # Calculate the estimated number of times customers will purchase
        estimated_freq=math.floor(days_to_end_of_year/regular_purchase_days)
        estimated_sales=round(sum(avg_Spending['AVERAGE_SPENDING']*estimated_freq),2)
        st.metric ('Days to End of Year',days_to_end_of_year)
        st.metric('Estimated Frequency',estimated_freq)
        st.metric('Estimated Sales', f"${estimated_sales}")
        st.write('In the next ',days_to_end_of_year,' days, if we get each member just to purchase ',estimated_freq,' time. The total estimated sales we can generate is $',estimated_sales,'.')
        

    with tab2:
    # How to use this page
        st.markdown("## High Level Goals")
        st.write("""As stated in our homepage, our team is dedicated to assisting Tasty Bytes in achieving its ambitious goals over the next 5 years. 
                 In particular, we aim to assist helping Tasty Bytes achieve a remarkable 25% Year-Over-Year increase, with the objective of elevating 
                 annual sales from \$105M to an impressive \$320M. This page is designed exclusively for the marketing team, showcasing our data-driven 
                 approach to empower and elevate your marketing strategies, ultimately driving significant sales growth.""")
        
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
        with st.expander("Raw Dataframe"):
            st.write(df.head(10))

        # Run pipeline
        clean_df,kmeans_df=pipeline(df)

        with st.expander("Cleaned and Transformed Data"):
            st.write(clean_df.head(10))

        # Setup: Model loading
        churn_model = load_model("assets/churn-prediction-model.jbl")
        seg_clf_model = load_model("assets/models/segment_classifier.jbl")

        # Setup: Get predictions
        cols_to_ignore=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)']
        churn_pred= pd.DataFrame(churn_model.predict(clean_df.drop(cols_to_ignore,axis=1)),columns=['CHURNED'])
        cluster_pred=pd.DataFrame(seg_clf_model.predict(kmeans_df.drop(cols_to_ignore,axis=1)),columns=['CLUSTER'])
        
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
        
        #Summary
        st.markdown("### Summary")
        # Number of members of each cluster
        cluster_counts = filtered_data.groupby('CLUSTER').size().reset_index(name='Number of Members')
        st.dataframe(cluster_counts, hide_index=True)

        # Number of members who churned and not churned
        churn_counts = filtered_data.groupby('CHURNED').size().reset_index(name='Number of Members')
        st.dataframe(churn_counts, hide_index=True)

        # Metrics table
        st.markdown("### Member's Metric")
        ## Display table
        metric_df=filtered_data[['CUSTOMER_ID','RECENCY','FREQUENCY','MONETARY','LENGTH_OF_RELATIONSHIP','PROFIT','PROFIT_MARGIN(%)']]
        st.dataframe(metric_df, hide_index=True)


        # Demographic table
        st.markdown("### Member's Demographic")
        ## Clean up columns
        filtered_data['CHILDREN_COUNT'] = filtered_data['CHILDREN_COUNT'].map({
        '0': "No",
        '1': "Yes",
        '2': "Yes",
        '3': "Yes",
        '4': "Yes",
        '5+': "Yes",
        'Undisclosed':'Undisclosed'})
        filtered_data.rename({'CHILDREN_COUNT':'HAVE_CHILDREN'},inplace=True,errors='ignore',axis=1)
        ## Display table
        demo_df=filtered_data[['CUSTOMER_ID','GENDER','MARITAL_STATUS','CITY','HAVE_CHILDREN','AGE']]
        st.dataframe(demo_df, hide_index=True)

        # Behavioral table
        st.markdown("### Member's Behaviour")
        
        ## Clean up columns
        # # Convert 'LENGTH_OF_RELATIONSHIP' from days to years (with decimal point)
        # filtered_data['LENGTH_OF_RELATIONSHIP_YEARS'] = filtered_data['LENGTH_OF_RELATIONSHIP'] / 365.25

        # # Rename the new column to indicate it represents relationship duration in decimal years
        # filtered_data.rename(columns={'LENGTH_OF_RELATIONSHIP': 'LENGTH_OF_RELATIONSHIP_DAYS'}, inplace=True)

        ## Display table
        beha_df=filtered_data[['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY']]
        st.dataframe(beha_df, hide_index=True)

        # Overall Table
        st.markdown("### Overall Table")
        st.dataframe(filtered_data, hide_index=True)

        #Allow user to download dataframe for further analysis
        st.header('**Export results ✨**')
        st.write("_Finally you can export the resulting table after Clustering and Churn Prediction._")
        csv = convert_df(filtered_data)
        st.download_button(
        "Press to Download",
        csv,
        "marketing.csv",
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
