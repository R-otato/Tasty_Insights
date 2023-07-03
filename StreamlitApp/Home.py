# --Team--
# Tutorial Group: 	T01 Group 4

# Student Name 1:	Ryan Liam Poon Yang
# Student Number: 	S10222131E
# Student Name 2:	Teh Zhi Xian
# Student Number: 	S10221851J
# Student Name 3:	Chuah Kai Yi
# Student Number: 	S10219179E
# Student Name 4:	Don Sukkram
# Student Number: 	S10223354J
# Student Name 5:	Darryl Koh
# Student Number: 	S10221893J

# --Import statements--
import streamlit as st
from PIL import Image
import os

# --Page 1--
st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.image(Image.open('assets/Logo.png'))
st.write("# Welcome to Tasty Insights! 👋")

st.write("""
  ## What Problem we're Tying to Solve?

  Tasty bytes goals over the next 5 years are to improve its sales, 25% YOY expecting to grow
  from \$105M/Year to $320M/Year.

  Customer churn is when a customer stops paying for a company's service, in the case of tasty bytes
  it is when a customer stops buying products with tasty bytes trucks. It is more expensive to gain
  new customers than it is to retain existing customers and for a company that is looking to grow its profits
  customer retention is important in ensuring a sustainable growth in sales.

  This product aims to assist sales and marketing representatives better understand when customers churn so as to take appropriate action
  to prevent customers from leaving the service and to work towards creating a better customer experience.
  
  """)


st.sidebar.success("Select a demo above.")
