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

image = Image.open('logo.png')

st.header('Welcome to my LinkedIn User Prediction App!')
st.subheader('Please answer the questions below to generate your prediction:')

income = st.radio('#1 What is your income level?:', ['Less than 10,000USD','10,000 to under 20,000USD','20,000 to under 30,000USD','30,000 to under 40,000USD','40,000 to under 50,000USD','50,000 to under 75,000USD','75,000 to under 100,000USD','100,000 to under 150,000USD','150,000USD+'])
if income == "Less than 10,000USD":
    income =1
elif Income == "10,000 to under 20,000USD":
    Income =2
elif Income == "20,000 to under 30,000USD":
    Income =3
elif Income == "30,000 to under 40,000USD":
    Income =4
elif Income == "40,000 to under 50,000USD":
    Income =5
elif Income == "50,000 to under 75,000USD":
    Income =6
elif Income == "75,000 to under 100,000USD":
    Income =7
elif Income == "100,000 to under 150,000USD":
    Income =8    
else:
    Income=9    
    
educ2 = st.radio('#2 What is your education level?:', ['1 - Less than high school (Grades 1-8 or no formal schooling)','2 - High school incomplete (Grades 9-11 or Grade 12 with NO diploma)','3 - High school graduate (Grade 12 with diploma or GED certificate)','4 - Some college, no degree (includes some community college)','5 - Two-year associate degree from a college or university','6 - Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)','7 - Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)','8 - Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])
if educ2 == "1 - Less than high school (Grades 1-8 or no formal schooling)":
    educ2 =1
elif educ2 == "2 - High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    educ2 =2
elif educ2 == "3 - High school graduate (Grade 12 with diploma or GED certificate)":
    educ2 =3
elif educ2 == "4 - Some college, no degree (includes some community college)":
    educ2 =4
elif educ2 == "5 - Two-year associate degree from a college or university":
    educ2 =5
elif educ2 == "6 - Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    educ2 =6
elif educ2 == "7 - Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ2 =7  
else:
    educ2=8 

par = st.selectbox(label="#3 Are you a parent?:", options=("Yes","No"))
if par == "Yes":
    par = 1
else:
    par = 0
    
marital = st.radio('#4 Are you married?:', ['1 - Yes','2 - No'])
if marital == "1 - Yes":
    marital = 1
else:
    marital = 0

female = st.radio('#5 What is your gender?:', ['1 - Male','2 - Female'])
if gender == "1 - Male":
    gender = 0
else:
    gender = 1
    
age = st.number_input('#6 What is your age?')

s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
        return np.where(x==1, 1, 0)

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0)),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])})

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

confusion_matrix(y_test,y_pred)

newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})

newdata["sm_li"] = lr.predict(newdata)

newdata

st.write(f"Predicted class: {predicted_class[0]}") 
st.write("0 = not a LinkedIn user, 1=LinkedIn user")
st.write(f"Probability that you are a LinkedIn user: {probs[0][1]}")
