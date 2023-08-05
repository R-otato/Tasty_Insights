#--Team--
# Tutorial Group: 	T01 Group 4 

#--Import statements--
import streamlit as st
import pandas as pd
import numpy as np 
import joblib 

# path to assets
path = 'assets/'
# latest date of the dataset
latest_date = pd.to_datetime('2022-10-31')

def init_model():
    
    # churn prediction
    yeojohnsontransformer = joblib.load(path + '/models/yeojohnsontransformer.jbl')
    onehotencoder = joblib.load(path + '/models/onehotencoder.jbl')
    minmaxscaler = joblib.load(path + 'models/minmaxscaler.jbl')
    model = joblib.load(path + 'models/xgb_churn_model.jbl')
    
    # member demographic
    onehotencoder_cust_demo = joblib.load(path + 'models/cust_demographic_ohe.jbl')
    model_cust_demo = joblib.load(path + 'models/cust_demographic_model.jbl')
    
    return yeojohnsontransformer, onehotencoder, minmaxscaler, model, \
            onehotencoder_cust_demo, model_cust_demo

def init_dataset():
    
    # display dataset
    member_data = pd.read_csv(path + 'datasets/relavent_original_dataset.csv')
    
    # for churn prediction proxy values
    df_before_scaling_sample = pd.read_csv(path + 'datasets/before_scaling_dataset.csv')
    
    # display cluster information
    cluster_information = pd.read_csv(path + 'datasets/cluster_information.csv')
    
    return member_data, df_before_scaling_sample, cluster_information

# to search for a specific member by first name, last name, or combination
def search_member(member_data):
    search_term = st.text_input("Search for a member by First Name, Last Name, or combination:")

    if search_term:
        # Filter the member data based on the search term using partial matching 
        selected_member = member_data[member_data['FIRST_NAME'].str.contains(search_term, case=False, na=False) |
                                        member_data['LAST_NAME'].str.contains(search_term, case=False, na=False) |
                                        member_data.apply(lambda row: search_term in f"{row['FIRST_NAME']} {row['LAST_NAME']}".lower(), axis=1)]

        
        if not selected_member.empty:
            st.subheader("Selected Member Details")
            st.dataframe(selected_member)
        else:
            st.write("No matching member found.")

# get user input
def user_input(member_data):
    
    # cast date columns to datetime
    member_data['SIGN_UP_DATE'] = pd.to_datetime(member_data['SIGN_UP_DATE'])
    member_data['BIRTHDAY_DATE'] = pd.to_datetime(member_data['BIRTHDAY_DATE'])
    
    city = st.selectbox("City", np.sort(member_data['CITY'].unique()))
    gender = st.selectbox("Gender", np.sort(member_data['GENDER'].unique()))
    marital_status = st.selectbox("Marital Status", np.sort(member_data['MARITAL_STATUS'].unique()), index = 1)
    children_count = st.selectbox("Number of Children", np.sort(member_data['CHILDREN_COUNT'].unique()), index = 2)


    sign_up_date = st.date_input("Select Sign Up Date", 
                                min_value = member_data['SIGN_UP_DATE'].min(), 
                                max_value = member_data['SIGN_UP_DATE'].max(),
                                value = member_data['SIGN_UP_DATE'].max()-pd.Timedelta(days=365*1))
    birthday_date = st.date_input("Select Birthday Date", 
                                min_value = member_data['BIRTHDAY_DATE'].min(), 
                                max_value = member_data['BIRTHDAY_DATE'].max(),
                                value = member_data['BIRTHDAY_DATE'].max()-pd.Timedelta(days=365*23))

    # Sliders for days since last order, number of orders, and amount spent per order
    days_since_last_order = st.slider("Days Since Last Order:", 1, 40, 10)
    num_of_orders = st.slider("Number of Orders:", 1, 60, 40)
    amount_spent_per_order = st.slider("Amount Spent per Order ($):", 10, 70, 20)

    # DataFrame based on user input
    df_user_input = pd.DataFrame({
        'CITY': [city],
        'GENDER': [gender],
        'MARITAL_STATUS': [marital_status],
        'CHILDREN_COUNT': [children_count],
        'SIGN_UP_DATE': [sign_up_date],
        'BIRTHDAY_DATE': [birthday_date],
        'Days_Since_Last_Order': [days_since_last_order],
        'Number_of_Orders': [num_of_orders],
        'Amount_Spent_Per_Order': [amount_spent_per_order]
    })
    
    return df_user_input

# accepts a dataframe, returns a dataframe with AGE column
def get_age(df):
    
    # cast BIRTHDAY_DATE to datetime
    df['BIRTHDAY_DATE'] = pd.to_datetime(df['BIRTHDAY_DATE'])
    # Calculate age in years and round to the nearest whole number
    df['AGE'] = ((latest_date - df['BIRTHDAY_DATE']).dt.days/365).round()
    # Convert age to integers
    df['AGE'] = df['AGE'].astype(int)

    return df

# churn prediction data manupulation    
def churn_prediction_data_manupulation(df_user_input,df_before_scaling_sample):
    
    # convert to datetime
    df_user_input['SIGN_UP_DATE'] = pd.to_datetime(df_user_input['SIGN_UP_DATE'])

    # RFM
    df_user_input['RECENCY'] = df_user_input['Days_Since_Last_Order']
    df_user_input['FREQUENCY'] = df_user_input['Number_of_Orders']
    df_user_input['MONETARY'] = df_user_input['Amount_Spent_Per_Order']*df_user_input['Number_of_Orders']
    
    # Average Time Difference, Max and Min Days without Purchase
    # TODO - ryan suggests Or maybe take the mean or median of the 3 columnsAnd say the values are set to this Because...
    df_user_input['AVG_DAYS_BETWEEN_PURCHASE'] = df_before_scaling_sample['AVG_DAYS_BETWEEN_PURCHASE'].mean()
    df_user_input['MAX_DAYS_WITHOUT_PURCHASE'] = df_before_scaling_sample['MAX_DAYS_WITHOUT_PURCHASE'].mean()
    df_user_input['MIN_DAYS_WITHOUT_PURCHASE'] = df_before_scaling_sample['MIN_DAYS_WITHOUT_PURCHASE'].mean()
    
    # latest date of the dataset
    latest_date = pd.to_datetime('2022-10-31')
    
    # Age
    df_user_input = get_age(df_user_input)
    # Number of locations visited
    df_user_input['NUM_OF_LOCATIONS_VISITED'] = df_user_input['Number_of_Orders'] \
                                                - round(df_user_input['Number_of_Orders']*0.05)
    # Length of relationship
    df_user_input['LENGTH_OF_RELATIONSHIP'] = (latest_date - df_user_input['SIGN_UP_DATE']).dt.days
    # Relative Purchase Frequency and Monetary
    df_user_input['RELATIVE_PURCHASE_FREQUENCY'] = df_user_input['FREQUENCY'] / df_user_input['LENGTH_OF_RELATIONSHIP']
    df_user_input['RELATIVE_PURCHASE_MONETARY'] = df_user_input['MONETARY'] / df_user_input['LENGTH_OF_RELATIONSHIP']

    # drop unnecessary columns
    df_user_input = df_user_input.drop(columns=['Days_Since_Last_Order', 'Number_of_Orders','Amount_Spent_Per_Order'])
    
    # add dummy columns needed for numerical transformation
    df_user_input = df_user_input.assign(
        CUSTOMER_ID=1,
        COUNTRY='United States',
        MAX_ORDER_TS=latest_date,
        ORDER_TS=latest_date,
        DAYS_TO_NEXT_ORDER=999,
    )
    
    return df_user_input

# use preprocessing models to transform data
def churn_preprocessing(df_user_input_churn_cleaned,yeojohnsontransformer,onehotencoder,minmaxscaler):
    
    # perform numerical transformation
    df_user_input_churn_cleaned = yeojohnsontransformer.transform(df_user_input_churn_cleaned)
    
    # drop unnecessary columns
    df_user_input_churn_cleaned = df_user_input_churn_cleaned.drop(columns=['CUSTOMER_ID','DAYS_TO_NEXT_ORDER',
                                'MAX_ORDER_TS','ORDER_TS',
                                'COUNTRY','BIRTHDAY_DATE','SIGN_UP_DATE'])
    
    # add dummy columns needed for the one hot encoding
    df_user_input_churn_cleaned = df_user_input_churn_cleaned.assign(
        CHURNED = 0,
    )
        
    # perform one hot encoding
    df_user_input_churn_cleaned = onehotencoder.transform(df_user_input_churn_cleaned)
    df_user_input_churn_cleaned.rename({'MARITAL_STATUS_Divorced/Seperated':'MARITAL_STATUS_Divorced_Or_Seperated'}, axis=1,inplace=True)
    df_user_input_churn_cleaned.columns = map(str.upper, df_user_input_churn_cleaned.columns)

    # perform scaling
    df_user_input_churn_cleaned = df_user_input_churn_cleaned.drop(columns=['CHURNED'])
    df_user_input_churn_cleaned[df_user_input_churn_cleaned.columns] = minmaxscaler.transform(df_user_input_churn_cleaned[df_user_input_churn_cleaned.columns])
    
    return df_user_input_churn_cleaned

# get similar members as imputed
def similar_members_sales(df_user_input,df_before_scaling_sample):
    df_user_input_counterfactual = df_user_input.copy()
    counterfactual_columns = ['RECENCY','FREQUENCY','AGE','LENGTH_OF_RELATIONSHIP']
    df_non_churned = df_before_scaling_sample[df_before_scaling_sample['CHURNED']==0]
    
    # Calculate the Euclidean distances between df_user_input_counterfactual and all rows in df_non_churned
    user_input_array = df_user_input_counterfactual[counterfactual_columns].values
    non_churned_array = df_non_churned[counterfactual_columns].values

    # Calculate the pairwise Euclidean distances using NumPy broadcasting
    distances = np.linalg.norm(user_input_array - non_churned_array, axis=1)

    # Find the indices of the five rows with the smallest Euclidean distances
    top_five_indices = distances.argsort()[:1]

    # Get the 'MONETARY' and 'LENGTH_OF_RELATIONSHIP' values from the top five rows
    monetary_values = df_non_churned.iloc[top_five_indices]['MONETARY']
    relationship_lengths = df_non_churned.iloc[top_five_indices]['LENGTH_OF_RELATIONSHIP']

    # Calculate monetary value per month for the top five counterfactuals
    monetary_per_month = monetary_values / (relationship_lengths / 30)  # Assuming 30 days per month

    # Calculate the average monetary value per month for the top five counterfactuals
    lifetime_sales = monetary_values.mean()
    monthly_sales = monetary_per_month.mean()
    
    return lifetime_sales,monthly_sales

# member demographic data manupulation
def member_demographic_data_manupulation(df_user_input_cust_demo,churn_prediction):
    # simplify children info
    df_user_input_cust_demo['CHILDREN_COUNT'] = df_user_input_cust_demo['CHILDREN_COUNT'].map({
    '0': "No",
    '1': "Yes",
    '2': "Yes",
    '3': "Yes",
    '4': "Yes",
    '5+': "Yes",
    'Undisclosed':'Undisclosed'})

    df_user_input_cust_demo.rename({'CHILDREN_COUNT':'HAVE_CHILDREN'},inplace=True,errors='ignore',axis=1)
    
    # Define the age groups and corresponding labels
    age_bins = [0, 46, 66, 83]
    age_labels = ['Adults', 'Middle-Aged Adults', 'Seniors']
    
    # get age
    df_user_input_cust_demo = get_age(df_user_input_cust_demo)
    # Bin the age column based on the age groups and labels
    df_user_input_cust_demo['AGE_GROUP'] = pd.cut(df_user_input_cust_demo['AGE'], bins=age_bins, labels=age_labels)
    
    # Define the age groups and corresponding labels
    DTNO_bins = [0, 7, 14, 30, 999]
    DTNO_labels = ['<7 Days', '14 Days', '30 Days', '>30 Days']

    # Bin the age column based on the age groups and labels
    df_user_input_cust_demo['DAYS_TO_NEXT_ORDER'] = pd.cut(df_user_input_cust_demo['Days_Since_Last_Order'], bins=DTNO_bins, labels=DTNO_labels)
    
    # get churn
    df_user_input_cust_demo['CHURNED'] = churn_prediction[0]
    
    # drop unnecessary columns
    cust_seg_demo_df=df_user_input_cust_demo[['GENDER', 'MARITAL_STATUS', 'CITY', 'HAVE_CHILDREN', 'AGE_GROUP', 'CHURNED']]
    
    return cust_seg_demo_df

# show member segment
def get_member_segment(segment,cluster_information):
    st.write("Member Type:", cluster_information['Title'].iloc[segment])
    st.write(cluster_information['Info'][segment])

def main():
    # set page title and icon
    st.set_page_config(page_title="Churn Prediction", page_icon="😎")

    # Page Title
    st.title("Churn Prediction")
    
    # initalise dataset and model
    member_data, df_before_scaling_sample, cluster_information = init_dataset()
    yeojohnsontransformer, onehotencoder, minmaxscaler, model, onehotencoder_cust_demo, model_cust_demo = init_model()
    
    st.write("""
            This page allows us to predict if a member is likely to churn or not. 
            As explained in the Home page, to reach our goal of 25% YoY sales 
            growth is to reduce churn. This in turn increases the number of repeat members, contributing
            to the growth in sales.
             
            By predicting if a member is likely to churn, we can take action to retain the member. 
            This can be extrapolated to entire member demographics, allowing us to direct attention to stop
            groups of members likely to churn from churning.
             """)
    
    # Show basic statistics about the member data
    st.header("Member Data Overview")
    st.info("Using the last updated data of the members in United States (October and beyond)")
    st.markdown("Total number of Transactions:  **:blue[72M]**")
    st.markdown("Total number of Transactions by Members: **:blue[7.5M]**")
    st.markdown("Number of Members: **:blue[50K]**")
    
    # Display a sample of the member data
    st.subheader("Member Data")
    st.dataframe(member_data)
    
    # allow for user to search for a member
    search_member(member_data)
    
    # Show insights on member churn
    st.header("Member Churn Predictor")
    st.write("Input a member's details to predict if they are likely to churn or not")
    
    # User Input for Dropdowns and Sliders
    df_user_input = user_input(member_data)
    
    # Show the user input
    st.subheader("Member to Predict")
    st.write(df_user_input.rename(columns=str.lower).iloc[0])

    
    if st.button("Predict Churn!"):
        # perform data manupulation for churn prediction
        df_user_input_churn_cleaned = churn_prediction_data_manupulation(df_user_input,df_before_scaling_sample)
        # apply preprocessing models
        df_churn_preprocessed = churn_preprocessing(df_user_input_churn_cleaned,yeojohnsontransformer,onehotencoder,minmaxscaler)
        # perform prediction    
        churn_prediction = model.predict(df_churn_preprocessed)
        
        # get similar members as imputed
        lifetime_sales,monthly_sales = similar_members_sales(df_user_input,df_before_scaling_sample)
        
        cust_result_message_template = """This member is **:blue[{}]** to churn.
                            This means that the member is **:blue[{}]** purchasing from the Tasty Bytes, 
                            resulting in **:blue[{}]** the member and potential sales. A member similar to the input
                            member has an average sales generated of **:blue[${:.0f}]**, with a lifetime total of **:blue[${:.0f}]**. Tasty Bytes 
                            can expect to **:blue[{}]** this amount of sales if this member **:blue[{}]** churn."""
                            
        # churn results
        st.header("Prediction Results")
        if churn_prediction[0] == 1:
            st.markdown(cust_result_message_template.format('likely', 'unlikely to continue', 'losing', monthly_sales, lifetime_sales, 'lose', 'does'))
        else:
            st.markdown(cust_result_message_template.format('unlikely', 'likely to continue', 'retaining', monthly_sales, lifetime_sales, 'gain', 'does not'))
        
        # member demographic data manupulation
        cust_seg_demo_df = member_demographic_data_manupulation(df_user_input,churn_prediction)
        
        # perform one hot encoding
        cust_seg_demo_df_preprocessed = onehotencoder_cust_demo.transform(cust_seg_demo_df)
        
        # perform member demographic prediction
        cust_demo_results = model_cust_demo.predict(cust_seg_demo_df_preprocessed)   
        
        st.subheader('What type of member is this?')
        get_member_segment(int(cust_demo_results),cluster_information)
        
        with st.expander("All Member Types"):
            for i in range(len(cluster_information)):
                get_member_segment(i,cluster_information)
                
        st.subheader('What next?')
        st.markdown("""
                With the ability to predict **:blue[whether a member will churn]**, and the member type, 
                we can now take **:blue[preventative action]** to retain the member. By successfully foreseeing
                which members will churn and taking action to retain them, we **:blue[safeguard our existing member base]**
                from shrinkage whilst attracting new members. This approach ensures a
                **:blue[sustained growth in sales]**, and enables us to achieve our overarching goals.
                 """)
        
main()        





        
# TODO: add sales amount to each segment