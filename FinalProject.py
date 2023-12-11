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

st.markdown("# Welcome to my LinkedIn User Prediction App!")
st.markdown("## Please answer the questions below to generate your prediction:")

# Read the CSV file
s = pd.read_csv('social_media_usage.csv')

# Create function to clean sm_li column
def clean_sm(x):
        test = np.where(x==1, 1, 0)
        return test  

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, clean_sm("web1h")),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])})

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

# New data for predictions
newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})

newdata["sm_li"] = lr.predict(newdata)

# New data for features: income, education, parent, married, female, age
person = [8, 7, 0, 1, 1, 42]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") #0 = not a LinkedIn user, 1=LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")


# New data for features: income, education, parent, married, female, age
person = [8, 7, 0, 1, 1, 82]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") #0 = not a LinkedIn user, 1=LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")


# ***
