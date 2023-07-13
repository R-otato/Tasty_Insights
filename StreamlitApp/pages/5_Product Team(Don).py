#--Import statements--
import streamlit as st
import pandas as pd


# Page Layout

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

# Input data
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