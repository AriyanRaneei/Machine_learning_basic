import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
#Loading data
data = pd.read_csv("PATH")
#data = pd.DataFrame(data)

#print(data.describe())

#print(data.isnull().sum())

#print(data['Chance of Admit '].value_counts())


#print(data.groupby(data['Chance of Admit ']).mean())


X = data.drop(columns='Chance of Admit ')
Y = data['Chance of Admit ']

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

#Y = Y.astype('float')
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=3)


model = LogisticRegression()
model.fit(x_train,y_train)
x_train_predict = model.predict(x_train)
x_train_acc = accuracy_score(x_train_predict,y_train)


model.fit(x_test,y_test)
x_test_predict = model.predict(x_test)
x_test_acc = accuracy_score(x_test_predict,y_test)

input_data = (1,337,118,4,4.5,4.5,9.65,1)
input_data = np.asanyarray(input_data).reshape(1,-1)
predict = model.predict(input_data)
