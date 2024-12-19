import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_absolute_error , accuracy_score
import pickle

data = pd.read_csv("PATH")
standard = StandardScaler()
label = LabelEncoder()
data['name'] = label.fit_transform(data['name'])


# training datatest
X = data.drop(columns=['name','status'])
Y = data['status']

x_train, x_test ,y_train ,y_test = train_test_split(X,Y)


# using svm model

model = svm.SVC(kernel="linear")
model.fit(x_train,y_train)
x_train_predict = model.predict(x_train)
acc = accuracy_score(y_train,x_train_predict)


model.fit(x_test,y_test)
x_test_predict = model.predict(x_test)
acc2 = accuracy_score(y_test,x_test_predict)
ss = data.drop(columns=['status','name'])



ask = np.array([i for i in ss.loc[20]]).reshape(1,-1)]
print(ask)


predict = model.predict(ask)

#delpoy machine learning model with streamlit
#saving the trained model

file_name = "parckingson_trained.sav"

pickle.dump(model,open(file_name,'wb'))




import pickle
#loading saved model

model =pickle.load(open("parckingson_trained.sav",'rb'))
