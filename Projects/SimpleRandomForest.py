import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#heart disease with RandomForstReg

data = pd.read_csv("PATH")


X = data.drop(columns='target')
Y = data['target']


x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=3,test_size=0.4)


model= RandomForestRegressor(n_estimators=100)

model.fit(x_train,y_train)
x_test_predict = model.predict(x_test)

R_squrred_error = metrics.r2_score(y_test,x_test_predict)


fig,ax = plt.subplots(2,1,figsize=(10,10))

Y_test = list(y_test)


ax[1].plot(Y_test,color="blue",label="REAL")


ax[0].plot(x_test_predict,color="red",label="PREDICTED")




