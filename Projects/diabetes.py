import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
#Support vector Machine
#data ==> data preprocessing ==> train and test 
# ==> support vectore machine classifier  ==>trained support machine classifier
#Data collection
dataset = pd.read_csv('PATH')
#Number of Rows and Columns in dataset
print(dataset.shape)
#print(dataset.describe())
#print(dataset['Outcome'].value_counts())
# 0 ==> None dibetic people  1===>diabetic people

#print(dataset.groupby('Outcome').mean())

X = dataset.drop(columns='Outcome',axis=1)
Y = dataset['Outcome']
#data standardazation
scaler = StandardScaler()
scaler.fit(X)
Standard_data = scaler.transform(X)
# or we can use sklearn fit

X = Standard_data

# train and test spilit
x_train,x_test,y_train,y_test =  train_test_split(X,Y,test_size=0.2,
                                                  stratify=Y,random_state=2)



# support vector machine
classifer  = svm.SVC(kernel='linear')
#training support vector machine
classifer.fit(x_train,y_train)
#evaluate our model
x_train_predic = classifer.predict(x_train)
training_data_acc = accuracy_score(x_train_predic,y_train)

x_test_predic = classifer.predict(x_test)
test_data_acc = accuracy_score(x_test_predic,y_test)
#Making predictive system
input_data = (5,139,64,35,140,28.6,0.411,26)
#changing to numpy array
data_array = np.asanyarray(input_data)
data_reshaped = data_array.reshape(1,-1)
# standardize the new data
std_data = scaler.transform(data_reshaped)

prediction = classifer.predict(std_data)


