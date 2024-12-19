import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
# data reduction
data = pd.read_csv("PATH")
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Height'].fillna(data['Height'].mean(),inplace=True)
data['Weight'].fillna(data['Weight'].mean(),inplace=True)


# data chocing
data = data.drop(columns=["ID",'Name',"NOC","Games","Year","Season","City",
                          "Event","Medal","Team"])

data.loc[data['Sex']=='M','Sex'] = 0
data.loc[data['Sex']=='F','Sex'] = 1

X = data.drop(columns='Sport')
Y = data['Sport']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.75,
                                                 random_state=2)

csf = DecisionTreeClassifier()

csf.fit(x_train,y_train)
x_train_predict = csf.predict(x_train)
x_train_acc = accuracy_score(x_train_predict,y_train)




csf.fit(x_test,y_test)
x_test_predict = csf.predict(x_test)
x_test_acc = accuracy_score(x_test_predict,y_test)

# Save model







input_data = (0,21,180,75)
input_data = np.asanyarray(input_data)
data_reform = input_data.reshape(1,-1)
predict = csf.predict(data_reform)

joblib.dump(csf,'ml_sport_model.joblib')





