import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score , mean_absolute_error

from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv("PATH")


data['Price'] = data['Price'].apply(lambda price: 1 if price > 1856 else 0)

data.replace({"Color":{"Black":0,"Silver":2,"White":3,
                       "Grey":4,"Blue":5,"Red":6,
                       "Green":7,"Orange":8,"Brown":9,
                       "Carnelian red":10,"Golden":11,"Beige":12,
                       "Sky blue":13,"Yellow":14,"purple":16,
                       "Pink":17}},inplace=True)

data.replace({"Wheel":{"Left wheel":0,"Right-hand drive":1,}},inplace=True)
data.replace({"Fuel type":{"Petrol":0,"Diesel":1,"Hybrid":2,
                           "LPG":3,"CNG":4,"Plug-in Hybrid":5,
                           "Hydrogen":6}},inplace=True)


data.replace({"Leather interior":{"Yes":0,"No":1}},inplace=True)

data.replace({"Gear box type":{"Automatic":0,"Tiptronic":1,"Manual":2,
                               "Variator":3}},inplace=True)


data.replace({"Drive wheels":{"Front":0,"4x4":1,"Rear":2,
                               "Variator":3}},inplace=True)


data.replace({"Doors":{"04-May":0,"02-Mar":1,">5":2,
                               "Variator":3}},inplace=True)

data.replace({"Engine volume":{'2.0 Turbo':20,"1.1 Turbo":11,
                              '0.8 Turbo':8 }}, inplace=True)
data.replace({"Color":{'Purple':15}}, inplace=True)






X = data.drop(columns=['ID', 'Price', 'Levy','Manufacturer','Category',"Mileage","Model","Engine volume"])

Y = data['Price']



x_train ,x_test,y_train,y_test = train_test_split(X,Y)
model = Lasso()

model.fit(x_train,y_train)
x_predict = model.predict(x_train)
acc = mean_absolute_error(y_train,x_predict)
