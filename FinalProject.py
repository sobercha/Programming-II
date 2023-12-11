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

income = st.radio('#1 What is your income level?:', ['1 - Less than 10,000USD','2 - 10,000 to under 20,000USD','3 - 20,000 to under 30,000USD','4 - 30,000 to under 40,000USD','5 - 40,000 to under 50,000USD','6 - 50,000 to under 75,000USD','7 - 75,000 to under 100,000USD','8 - 100,000 to under 150,000USD','9 - 150,000USD+'])
educ2 = st.radio('#2 What is your education level?:', ['1 - Less than high school (Grades 1-8 or no formal schooling)','2 - High school incomplete (Grades 9-11 or Grade 12 with NO diploma)','3 - High school graduate (Grade 12 with diploma or GED certificate)','4 - Some college, no degree (includes some community college)','5 - Two-year associate degree from a college or university','6 - Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)','7 - Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)','8 - Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])
par = st.radio('#3 Are you a parent?:', ['1 - Yes','2 - No'])
marital = st.radio('#4 Are you married?:', ['1 - Yes','2 - No'])
gender = st.radio('#5 What is your gender?:', ['1 - Male','2 - Female'])
age = st.number_input('#6 What is your age?')


# Read the CSV file
s = pd.read_csv('social_media_usage.csv')

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, clean_sm("web1h")),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])})

# Create function to clean sm_li column
def clean_sm(x):
        test = np.where(x==1, 1, 0)
        return test  

# Drop missing data
ss = ss.dropna()

# Create target vector and feature set
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

#Initialize lrm with class weight set to balanced 
lr = LogisticRegression(class_weight='balanced')

#Fit model with training data
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

person = ['income', 'educ2', 'par', 'marital', 'gender', 'age']

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
st.write(f"Predicted class: {predicted_class[0]}") #0 = not a LinkedIn user, 1=LinkedIn user
st.write(f"Probability that you are a LinkedIn user: {probs[0][1]}")
