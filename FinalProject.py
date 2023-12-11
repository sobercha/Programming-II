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

