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
    cols_Not_Involved=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)']
    not_Involved=data[cols_Not_Involved]
    data.drop(cols_Not_Involved,axis=1,inplace=True,errors='ignore')

    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/models/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/models/windsorizer_gau.jbl")
    kmeansMinMaxScaler=joblib.load("assets/models/kmeans_scaling.jbl")

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
    cols_Not_Involved=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)']
    not_Involved=data[cols_Not_Involved]
    data.drop(cols_Not_Involved,axis=1,inplace=True,errors='ignore')

    # Load the necessary transformations
    windsorizer_iqr = joblib.load("assets/models/windsorizer_iqr.jbl")
    windsorizer_gau = joblib.load("assets/models/windsorizer_gau.jbl")
    yjt = joblib.load("assets/models/yjt.jbl")
    ohe_enc = joblib.load("assets/models/ohe_enc.jbl")
    minMaxScaler = joblib.load("assets/models/minmaxscaler.jbl")

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

    # Load the necessary transformations
    ohe_enc = joblib.load("assets/models/memb_sales_ohe.jbl")
    minMaxScaler = joblib.load("assets/models/memb_sales_scale.jbl")
    # Apply the transformations to the data
    data = ohe_enc.transform(data)  # Apply One-Hot Encoding
    data[data.columns] = minMaxScaler.transform(data[data.columns])  # Apply Min-Max Scaling

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


def automate_sales_pred(current_date,data,sales_model):
    #Get dates
    next_month_date = current_date + pd.DateOffset(months=1)
    next_quarter_date = current_date + pd.DateOffset(months=3)
    next_year_date = current_date + pd.DateOffset(years=1)

    # Prepare input data for next month, quarter, and year predictions
    next_month_pred=data.copy()
    next_quarter_pred = data.copy()
    next_year_pred = data.copy()
    #Next Month
    next_month_pred['DATE']=next_quarter_date
    next_month_pred['YEAR'] = next_month_pred['DATE'].dt.year
    next_month_pred['MONTH'] = next_month_pred['DATE'].dt.month
    next_month_pred.drop('DATE',axis=1,inplace=True)
    next_month_pred_df=sales_pipeline(next_month_pred)
    next_month_pred['NEXT_MONTH_SALES']= pd.DataFrame(sales_model.predict(next_month_pred_df),columns=['NEXT_MONTH_SALES'])
    
    #Next Quarter
    next_quarter_pred = pd.DataFrame(pd.date_range(current_date, next_quarter_date-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
    next_quarter_pred['YEAR'] = next_quarter_pred['DATE'].dt.year
    next_quarter_pred['MONTH'] = next_quarter_pred['DATE'].dt.month
    next_quarter_pred.drop('DATE', axis=1, inplace=True)
    # Combine data and next_year_pred
    df_list = []
    for year, month in zip(next_quarter_pred['YEAR'], next_quarter_pred['MONTH']):
        for cluster, city in zip(data['CLUSTER'], data['CITY']):
            row = {
                'CLUSTER': cluster,
                'CITY': city,
                'NUMBER OF MEMBERS': data[data['CLUSTER'] == cluster]['NUMBER OF MEMBERS'].values[0],
                'FREQUENCY': data[data['CLUSTER'] == cluster]['FREQUENCY'].values[0],
                'YEAR': year,
                'MONTH': month
            }
            df_list.append(row)

    # Convert the list of rows to a DataFrame
    next_quarter_df= pd.DataFrame(df_list)
    next_quarter_pred_df=sales_pipeline(next_quarter_df)
    next_quarter_df['NEXT_QUARTER_SALES']= pd.DataFrame(sales_model.predict(next_quarter_pred_df),columns=['NEXT_QUARTER_SALES'])

    #Next Year
    next_year_pred = pd.DataFrame(pd.date_range(current_date, next_year_date-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
    next_year_pred['YEAR'] = next_year_pred['DATE'].dt.year
    next_year_pred['MONTH'] = next_year_pred['DATE'].dt.month
    next_year_pred.drop('DATE', axis=1, inplace=True)

    # Combine data and next_year_pred
    df_list = []
    for year, month in zip(next_year_pred['YEAR'], next_year_pred['MONTH']):
        for cluster, city in zip(data['CLUSTER'], data['CITY']):
            row = {
                'CLUSTER': cluster,
                'CITY': city,
                'NUMBER OF MEMBERS': data[data['CLUSTER'] == cluster]['NUMBER OF MEMBERS'].values[0],
                'FREQUENCY': data[data['CLUSTER'] == cluster]['FREQUENCY'].values[0],
                'YEAR': year,
                'MONTH': month
            }
            df_list.append(row)
    
    # Convert the list of rows to a DataFrame
    next_year_df= pd.DataFrame(df_list)
    next_year_pred_df=sales_pipeline(next_year_df)
    next_year_df['NEXT_YEAR_SALES']= pd.DataFrame(sales_model.predict(next_year_pred_df),columns=['NEXT_YEAR_SALES'])
    return next_month_pred, next_quarter_df, next_year_df

def get_sales_growth(sales_model_input,current_date,seg_Sales):
    #Get dates
    prev_month_date = current_date - pd.DateOffset(months=1)
    prev_quarter_date = current_date - pd.DateOffset(months=3)
    prev_year_date = current_date - pd.DateOffset(years=1)

    #Merge the segment sales with sales model input to get 
    seg_Sales=pd.merge(seg_Sales,right=sales_model_input[['CLUSTER','CITY']],on=['CLUSTER','CITY'],how='inner')

    # Get the prev month, quarter, year from dataframe
    #Prev Month
    prev_month = pd.DataFrame(pd.date_range(prev_month_date, current_date-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
    prev_month['YEAR'] = prev_month['DATE'].dt.year
    prev_month['MONTH'] = prev_month['DATE'].dt.month
    prev_month.drop('DATE',axis=1,inplace=True)
    prev_month_sales=seg_Sales.copy()
    prev_month_sales=pd.merge(prev_month_sales,right=prev_month,on=['YEAR','MONTH'],how='inner')
   
    #Prev Quarter
    prev_quarter = pd.DataFrame(pd.date_range(prev_quarter_date, current_date-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
    prev_quarter['YEAR'] = prev_quarter['DATE'].dt.year
    prev_quarter['MONTH'] = prev_quarter['DATE'].dt.month
    prev_quarter.drop('DATE',axis=1,inplace=True)
    prev_quarter_sales=seg_Sales.copy()
    prev_quarter_sales=pd.merge(prev_quarter_sales,right=prev_quarter,on=['YEAR','MONTH'],how='inner')

    #Prev Year
    prev_year = pd.DataFrame(pd.date_range(prev_year_date, current_date-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
    prev_year['YEAR'] = prev_year['DATE'].dt.year
    prev_year['MONTH'] = prev_year['DATE'].dt.month
    prev_year.drop('DATE',axis=1,inplace=True)
    prev_year_sales=seg_Sales.copy()
    prev_year_sales=pd.merge(prev_year_sales,right=prev_year,on=['YEAR','MONTH'],how='inner')
    return prev_month_sales,prev_quarter_sales,prev_year_sales



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
        st.markdown("""As stated in our homepage, our team is dedicated to assisting Tasty Bytes in achieving its ambitious goals over the next 5 years. 
                 In particular, we aim to help Tasty Bytes achieve a remarkable **:blue[25% Year-Over-Year increase in sales]**. This page is exclusively focused 
                 on churn prediction, which is a twin concept of member retention. It is designed to empower and elevate your marketing strategies with 
                 our data-driven approach, ultimately driving significant sales growth by retaining valuable customers and understanding their likelihood of churning.""")
        st.markdown("## Success Metrics")
        st.write("To assess if the marketing page is a success, I have established a key success metric.")
        st.markdown("Success Metric: Achieve a **:blue[25% Year-over-Year (YoY) Member Sales Growth]** for 2022")
        st.write('''The achievement of at least 25% Year on Year Member Sales Growth for 2022, serves as a tangible step 
                 towards realizing the high-level goal of achieving 25% year on year sales growth for non-member 
                 and member purchases over the next 5 years.''')
        
        # How to use predictions
        st.markdown('## How to Utilize the Predictions')
        st.write(
            """
            In the Model Prediction tab, you will have access to valuable insights derived from both the Segmentation and Churn prediction models. 
            Additionally, we have incorporated a Sales Prediction model that predicts the segment's sales for the next month. Once you filter the segments and churn 
            predictions, you can input the estimated frequency for each month to get personalized sales predictions.

            With these powerful predictions, you can unlock various opportunities:

            - Explore Customer Segments: Dive into the different segments within your member base, understanding their purchasing behavior.

            - Identify Churn Likelihood: Gain visibility into which members are likely to churn or remain engaged, predicting their purchase behavior in the next 14 days.

            - Targeted Marketing Strategies: Armed with these predictions, you can design targeted marketing schemes tailored to specific segments or groups of customers, maximizing the impact of your campaigns.

            - Personalized Sales Predictions: Asssume the impact of your marketing strategy to get the forecasted sales

            Leveraging these data-driven insights, your marketing team can make informed decisions, optimize marketing efforts, and drive sales growth for Tasty Bytes. Let's unlock the full potential of your marketing strategies together!
            """
        )



    with tab2:
        with st.expander("How to Use This Page"):
            #Going to add some stuff here 
            st.write("""
            1. Default Data: The page displays default data for members in the United States. To update member information,
            upload a new Excel file in the Input Data section.
            2. Predictions: After uploading your file, the predictions will be automatically generated and shown.
            3. Filter Data: Use filters to explore specific segments or refine the data for analysis.
            4. Sales Forecast: Input the expected number of purchases for all members in each month to assume the impact of your marketing strategy and generate forecasted sales.
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
            df=pd.read_csv('assets/datasets/marketing.csv')

        ## Display uploaded or defaul file
        with st.expander("Uploaded/Default Data"):
            st.write(df)

        # Run pipeline
        clean_df=churn_pipeline(df)
        kmeans_df=kmeans_pipeline(df)

        # Setup: Model loading
        churn_model = load_model("assets/models/churn-prediction-model.jbl")
        seg_clf_model = load_model("assets/models/segment_classifier.jbl")
        sales_model=load_model("assets/models/memb_sales_pred.jbl")

        # Setup: Get predictions
        cols_to_ignore=['CUSTOMER_ID','FREQUENT_MENU_ITEMS','FREQUENT_MENU_TYPE','FREQUENT_TRUCK_BRAND','PREFERRED_TIME_OF_DAY','PROFIT','PROFIT_MARGIN(%)','AVG_QUANTITY', 
                       'AVG_UNIT_PRICE', 'AVG_SALES','STD_QUANTITY', 'STD_UNIT_PRICE', 'STD_SALES']
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
        st.markdown("""### Member's Segments""")
        cluster_counts = filtered_data.groupby('CLUSTER').size().reset_index(name='NUMBER OF MEMBERS')
        st.dataframe(cluster_counts, hide_index=True)
        with st.expander('Cluster terms'):
            st.write("""
                     1. Active: Members who have made a transaction recently
                     2. Engaged: Members who have made a transaction quite recently
                     3. Inactive: Members who have not made a transaction in a significant period.
                     4. High-Value: Members with a high monetary and frequency value
                     5. Moderate-Value: Members with a moderate monetary and frequency value
                     6. Low-Value: Members with a low monetary and frequency value.
                     """)
        with st.expander('Marketing Opportunities'):
            st.write("""
            - **Personalized Offers:** Tailor offers and promotions to active low-spending and engaged moderate-value members to increase their spending and encourage repeat purchases.

            - **Loyalty Programs:** Reward high-value loyal members with exclusive perks and incentives to strengthen their loyalty and retain them as brand advocates.

            - **Reactivation Campaigns:** Design targeted campaigns to reactivate inactive low-spending members and bring them back into the fold.

            - **Upselling and Cross-selling:** Identify opportunities to upsell and cross-sell to active moderate-value members, maximizing their average order value.
            """)

        # Number of members who churned and not churned
        st.markdown("""### Churn Analysis""")
        # Display assumption
        st.write('*Note: Churn is defined by whether or not a member will purchase from us in the next 14 days')
        # Display churn analysis
        churn_counts = filtered_data.groupby('CHURNED').size().reset_index(name='NUMBER OF MEMBERS')
        st.dataframe(churn_counts, hide_index=True)
        with st.expander('Marketing Opportunities cont.'):
            st.write("""
            - **Win-Back Campaigns:** For the customers who churned (i.e., did not make a purchase in the last 14 days), design targeted win-back campaigns. Offer them personalized incentives, 

            - **Customer Retention Programs:** Focus on retaining the existing customers who did not churn. Implement loyalty programs, offer rewards, and provide exceptional customer service to enhance their loyalty and encourage repeat purchases.
            """)
        

        # Display forecasted sales
        st.write('### Member Forecasted Sales')
        # #Get user input
        # Get the input from the user
        expected_purchases_input = st.text_input("Expected number of orders in each month:", "1")

        # Validate the input
        estimated_frequency = validate_integer_input(expected_purchases_input)

        # Display error message if input is not an integer
        if estimated_frequency is None:
            st.error("Please enter a valid integer for the expected number of orders.")
        else:
            # Setup data
            sales_model_input=filtered_data.groupby(['CLUSTER','CITY']).size().reset_index(name='NUMBER OF MEMBERS')
            #Hardcode last date as Tasty Bytes data will not update
            current_date=pd.to_datetime('2022-11-01')
            #Update frequency
            sales_model_input['FREQUENCY']=estimated_frequency*sales_model_input['NUMBER OF MEMBERS']
            #Load sales csv
            seg_Sales=pd.read_csv('assets/datasets/seg_sales.csv')
            #Get predictions
            next_month_pred, next_quarter_df, next_year_df=automate_sales_pred(current_date,sales_model_input,sales_model)
            
            #Total Sales by month,quarter,year
            next_month_sales=next_month_pred['NEXT_MONTH_SALES'].sum()/ 10**6
            next_quarter_sales=next_quarter_df['NEXT_QUARTER_SALES'].sum()/ 10**6
            next_year_sales=next_year_df['NEXT_YEAR_SALES'].sum()/ 10**6
            # Get MoM,QoQ,YoY growth
            prev_month_sales,prev_quarter_sales,prev_year_sales=get_sales_growth(sales_model_input,current_date,seg_Sales)
            mom_sales=(next_month_pred['NEXT_MONTH_SALES'].sum()-prev_month_sales['SALES'].sum())/prev_month_sales['SALES'].sum()*100
            qoq_sales=(next_quarter_df['NEXT_QUARTER_SALES'].sum()-prev_quarter_sales['SALES'].sum())/prev_quarter_sales['SALES'].sum()*100
            yoy_sales=(next_year_df['NEXT_YEAR_SALES'].sum()-prev_year_sales['SALES'].sum())/prev_year_sales['SALES'].sum()*100
          
            # Display assumption
            st.write('Assuming you are able to get each customers to purchase from you ',estimated_frequency,' time every month.')
            st.write('Latest date is based on the data lastest date which is 2022-11-01')
            st.write('These are your predicted sales:')
            # Display metrics
            col1,col2,col3=st.columns(3)
            col1.metric('Next Month Sales', f"${round(next_month_sales, 2)}M")
            col2.metric('Next Quarter Sales', f"${round(next_quarter_sales, 2)}M")
            col3.metric('Next Year Sales', f"${round(next_year_sales, 2)}M")
            col1.metric('Month-over-month', f"{round(mom_sales, 2)}%")
            col2.metric('Quarter-over-quarter', f"{round(qoq_sales, 2)}%")
            col3.metric('Year-over-year', f"{round(yoy_sales, 2)}%")
            
            # Success Metrics
            st.markdown('## Did we hit our Success Metrics?')
            st.markdown("""Currently, our current *:blue[Year on Year Member Sales Growth for 2022 stands at 16.05%]*. With data available up until 2022-11-01, 
                        we utilized our sales prediction model to forecast sales for the next two months. Under the assumption that our Churn Prediction model
                        has helped the marketing team to get each member to purchase at least twice a month, we *:blue[anticipate an impressive Year on Year 
                        Member Sales Growth of 36.5%]* for 2022. This accomplishment aligns with our Success Metrics, as we have 
                         *:blue[ attained more than 25% Year-over-Year (YoY) Member Sales Growth]* for 2022.""")


            sales_model_input=data.groupby(['CLUSTER','CITY']).size().reset_index(name='NUMBER OF MEMBERS')
            sales_model_input['FREQUENCY']=2*sales_model_input['NUMBER OF MEMBERS']
            current_date=pd.to_datetime('2022-01-01')
            #Get previous sales
            prev_month_sales,prev_quarter_sales,prev_year_sales=get_sales_growth(sales_model_input,current_date,seg_Sales)
            #Get next year actual sales
            next_year = pd.DataFrame(pd.date_range(current_date,current_date+pd.DateOffset(years=1)-pd.DateOffset(months=1), freq='MS'), columns=['DATE'])
            next_year['YEAR'] = next_year['DATE'].dt.year
            next_year['MONTH'] = next_year['DATE'].dt.month
            next_year.drop('DATE',axis=1,inplace=True)
            next_year_sales=pd.merge(seg_Sales,right=next_year,on=['YEAR','MONTH'],how='inner')
        
            #Get predictions
            next_month_pred, next_quarter_df, next_year_df=automate_sales_pred(current_date,sales_model_input,sales_model)
            # Calculate YoY
            actual_sales=next_year_sales['SALES'].sum()
            pred_sales=next_month_pred['NEXT_MONTH_SALES']*2
            pred_sales=actual_sales+pred_sales.sum()
            prev_sales=prev_year_sales['SALES'].sum()
            actual_yoy_sales=(actual_sales-prev_sales)/prev_sales.sum()*100
            pred_yoy_sales=(pred_sales-prev_sales)/prev_sales.sum()*100
            st.metric('Actual Year-over-year', f"{round(actual_yoy_sales, 2)}%")
            st.metric('Predicted Year-over-year', f"{round(pred_yoy_sales, 2)}%")

          
 
###########################
### Page Configurations ### 
###########################
if __name__ == "__main__":
    # Setting page configuration
    st.set_page_config(
        "Tasty Bytes Marketing by Ryan Liam",
        "📊",
        #initial_sidebar_state="expanded",
    )
    main()
