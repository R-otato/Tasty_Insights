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

# --Page 1--
st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.image(Image.open('assets/Logo.png'))
st.write("# Welcome to Tasty Insights! 👋")

st.write("""
  ## What Problem we're Tying to Solve?

  Imagine you own a telecommunication company and you want to have a Machine Learning model to predict the custumers that may probably churn in the next months. So I developed one and deployed in this prototype web application that would help you in this situation!

  For those who don't know, churn is when a custumer stops paying for a company's service. And as I've heard once, it is cheaper to keep the custumer you already have instead of spending more and more money and time to gather new ones.
""")


st.sidebar.success("Select a demo above.")
