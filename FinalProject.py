# Import packages
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

st.header('Welcome to my LinkedIn User Prediction App!')
st.subheader('Please answer the questions below to generate your prediction:')

income = st.selectbox(label="#1 What is your income level?:",
                      options=("Less than 10,000USD","10,000 to under 20,000USD","20,000 to under 30,000USD","30,000 to under 40,000USD","40,000 to under 50,000USD","50,000 to under 75,000USD","75,000 to under 100,000USD","100,000 to under 150,000USD","150,000USD"))
if income == "Less than 10,000USD":
    income =1
elif income == "10,000 to under 20,000USD":
    income =2
elif income == "20,000 to under 30,000USD":
    income =3
elif income == "30,000 to under 40,000USD":
    income =4
elif income == "40,000 to under 50,000USD":
    income =5
elif income == "50,000 to under 75,000USD":
    income =6
elif income == "75,000 to under 100,000USD":
    income =7
elif income == "100,000 to under 150,000USD":
    income =8    
else:
    income=9    
    
education = st.selectbox(label="#2 What is your education level?:",
                         options=("Less than high school (Grades 1-8 or no formal schooling)","High school incomplete (Grades 9-11 or Grade 12 with NO diploma)","High school graduate (Grade 12 with diploma or GED certificate)","Some college, no degree (includes some community college)","Two-year associate degree from a college or university","Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)","Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)","Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"))
if education == "Less than high school (Grades 1-8 or no formal schooling)":
    education =1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education =2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education =3
elif education == "Some college, no degree (includes some community college)":
    education =4
elif education == "Two-year associate degree from a college or university":
    education =5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education =6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education =7  
else:
    education=8 

parent = st.selectbox(label="#3 Are you a parent?:", 
                      options=("Yes","No"))
if parent == "Yes":
    parent = 1
else:
    parent = 0
    
married = st.selectbox(label="#4 Are you married?:",
                       options=("Yes","No"))
if married == "Yes":
    married = 1
else:
    married = 0

female = st.selectbox(label="#5 What is your gender?:", 
                      options=("Male","Female"))
if female == "Male":
    female = 0
else:
    female = 1
    
age = st.number_input('#6 What is your age?')
if age >98:
    age = NA
else:
    age = age

s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
        return np.where(x==1, 1, 0)

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

person = ["income", "education", "parent", "married", "female", "age"]

predicted_class = lr.predict([person])

probs = lr.predict_proba([person])

st.write(f"Predicted class: {predicted_class[0]}") 
st.write("0 = not a LinkedIn user, 1=LinkedIn user")
st.write(f"Probability that you are a LinkedIn user: {probs[0][1]}")
