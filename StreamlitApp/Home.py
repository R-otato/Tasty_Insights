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

st.write("## What Problem we're Tying to Solve?")

st.image(Image.open('assets/tasty_byte_goals.jpg'))

st.write("""  Tasty bytes goals over the next 5 years are to improve its sales, 25% YOY expecting to grow
  from \$105M/Year to $320M/Year.

  Customer churn is when a customer stops paying for a company's service, in the case of tasty bytes
  it is when a customer stops buying products with tasty bytes trucks. It is more expensive to gain
  new customers than it is to retain existing customers and for a company that is looking to grow its profits
  customer retention is important in ensuring a sustainable growth in sales.

  This product aims to serve the understanding of churn rate and its affect on company KPI's relative to the stakeholders view
  so as to provide an actionable insight for the user to understand what they can do to reduce churn and the link their action has
  towards achieving company goals.
  
  """)

'''
Tabs:
1) What kinds of customers are churning the most
2) CFO - Next months Sales Predictions (Based on churn rate and average days to next order)
3) CEO - Operationally where is performing the best?
4) Marketing - Where are customers churning more and how do we advertise to the groups churning?
5) Food Truck - Do customers continue to use the service after I serve them?
'''

st.sidebar.success("Select a demo above.")