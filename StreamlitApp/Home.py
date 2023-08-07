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
import pandas as pd

# loading data
df_CTS = pd.read_csv('assets/datasets/CTS.csv')
df_CTS = df_CTS.sort_values(by=["YEAR", "MONTH"])

# --Page 1--
st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.image(Image.open('assets/images/Logo.png'))
st.write("# Welcome to Tasty Insights! 👋")

st.write("## What Problem we're Tying to Solve?")

st.image(Image.open('assets/images/tasty_byte_goals.jpg'))

st.write("""  Tasty bytes goals over the next 5 years are to improve its sales, 25% YOY expecting to grow
  from \$105M/Year to $320M/Year.

  Customer churn is when a customer stops paying for a company's service, in the case of tasty bytes
  it is when a customer stops buying products with tasty bytes trucks. It is more expensive to gain
  new customers than it is to retain existing customers and for a company that is looking to grow its profits
  customer retention is important in ensuring a sustainable growth in sales.
         
  Churn in our context is also a twin to customer recursion. We calculate churn by identify the customers days to next order and
  taking a threshold value to identify if a customer churns or not. Customer recursion is closely related
  to the profits and sales of an company and so we think it is a good indicator of certain KPI's. Our models do not 
  take into account other factors when providing a value in terms of profit increses etc as it is outside of scope but
  other factors do affect the outcome and change in sales other than churn rate.

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

st.write("## The value of Tasty Insights")

st.write("""
    ### How does Churn link to Sales?

    According to prelimnary research during the pitch phase we have established a link between Days to Next Order (target variable)
    which is what we derive our churn prediction from, there is a clear link between the average days to next order in a month with 
    that months sales.

    Through the prediction of churn rate we will imply the average days to next orders and in turn predict the potential sales of
    following month allowing for projection in sales and allow for informed steps to be taken in terms of operations and strategy.
         
    Customer recurssion is important for a businesses growth where when the return rate or the days to next orders is low the sales is
    likely to increase as well.

    Churn is a twin to customer recurssion, for the sake of predictions and model building internally we refer to the term as churn,
    in a business context the better term is recurssion or how often the customer will return. By identifying cause and effect,
    and providing insight for different perspectives to manage and control recussion we create actionable statements and insights
    tasty bytes to work towards their high-level goals.
    """)

st.dataframe(
  df_CTS,
  hide_index=True
)
"""
### Column Explainer
1) Churn Rate: The percentage of transactions that lead to a day to next order greater than 9 days
2) Sales: The amount of sales in the month period
3) Year: The year in which the data is from
4) Month: The month in which the data is from
5) Unique Customers: The number of unique customers in that month period where the data is from
6) Change in CR: The change in churn rate from previous month
7) Change in Sales: The change in sales from previous month
"""

"""
### DISCLAIMER:
A standard scenario for churns effect on the sales is the following...
Churn rate goes down (-%), this means customer recurssion increases which in turn means sales increases (+%).

However that may not always be the case where business is as usual (BAU). Scenarios occur where when churn goes down (-%)
sales also go down (-%) there may be other causes where the customer base is has actually declined which derives from the total
customers at the start sale period and churn calculation period being smaller.

A myriad of other factors could lead to a change in the actual derived values in which business situation changes such as lowered
average sale (the amount in each transaction) or the growth in customer base is not the same as previous months (new unique customers),
leading to a smaller growth.

Hence Churn is not the only variable in relation to the sales, other variables are at play and churn on its own will not be a complete
representation of potential sales but can act as an indicator.
"""

# st.sidebar.success("Select a demo above.")
