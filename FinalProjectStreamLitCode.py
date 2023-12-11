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

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})

newdata["sm_li"] = lr.predict(newdata)

person = [8, 7, 0, 1, 1, 42]

predicted_class = lr.predict([person])

probs = lr.predict_proba([person])

print(f"Predicted class: {predicted_class[0]}") #0 = not a LinkedIn user, 1=LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")

person = [8, 7, 0, 1, 1, 82]

predicted_class = lr.predict([person])

probs = lr.predict_proba([person])

print(f"Predicted class: {predicted_class[0]}") #0 = not a LinkedIn user, 1=LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")


# ***
