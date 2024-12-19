import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#ROCK AND MINE PROJECT
#SONAR DATA ==> DATAPERPROCESSING ==> TRAIN AND TEST SPILIT
# LOGSTIC REGRESSION MODEL SUPERIAL LEARNING ==> NEW DATA


#we dont have column names so we utilize header = None
sonar_data = pd.read_csv("PATH",header=None)
#statical describe
#print(sonar_data.describe())
#print(sonar_data[60].value_counts())
#R ==>ROCK
# M ==> Mine
print(sonar_data.groupby(60).mean())
#SEPRATING DATA
X = sonar_data.drop(columns=60,axis=1)
Y  = sonar_data[60]
#TRAIN AND TEST DATA
x_train ,x_test ,y_train ,y_test = train_test_split(X,Y,train_size=0.4,random_state=2,
                                                    stratify=Y)
#we want to seprate data with rock and mine so we utilize stratify =Y
#train machine learing Model MODEL TRAINNING
#logstic regresson
model = LogisticRegression()
#taring the model
model.fit(x_train,y_train)
#model evaluation
#accuracy on training data

x_train_predicton = model.predict(x_train)
traing_data_accuracy = accuracy_score(x_train_predicton,y_train)


#accuracy on test data

x_test_predicton = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predicton,y_test)
#Prediction system 
input_data = (0.0335,0.0258,0.0398,0.0570,0.0529,0.1091,0.1709,0.1684,0.1865,0.2660,0.3188,0.3553,0.3116,0.1965,0.1780,0.2794,0.2870,0.3969,0.5599,0.6936,0.7969,0.7452,0.8203,0.9261,0.8810,0.8814,0.9301,0.9955,0.8576,0.6069,0.3934,0.2464,0.1645,0.1140,0.0956,0.0080,0.0702,0.0936,0.0894,0.1127,0.0873,0.1020,0.1964,0.2256,0.1814,0.2012,0.1688,0.1037,0.0501,0.0136,0.0130,0.0120,0.0039,0.0053,0.0062,0.0046,0.0045,0.0022,0.0005,0.0031,)
# convert it to numpy  array
input_data_array = np.asarray(input_data)
#feature the numpy array as we are predicting for once intance
input_data_reshaped = input_data_array.reshape(1,-1)
prediction_input = model.predict(input_data_reshaped)

if (prediction_input[0] =='M'):
    print('ITS MINE')
else:
    print('ITS ROCK')

