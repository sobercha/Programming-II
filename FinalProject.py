import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

s = pd.read_csv('social_media_usage.csv')

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

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(class_weight='balanced')

st.title('Welcome to my LinkedIn User Predictor App!')

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
