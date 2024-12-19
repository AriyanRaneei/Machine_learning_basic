import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRFRegressor
from sklearn import metrics

#data collection

data  = pd.read_csv("PATH")


data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)

mode_of_outlet = data.pivot_table(values='Outlet_Size',columns= 'Outlet_Type', 
                                  aggfunc=(lambda x: x.mode()[0]))



missing_values = data['Outlet_Size'].isnull()


data.loc[missing_values,"Outlet_Size"] = data.loc[missing_values,"Outlet_Type"].apply(lambda x:mode_of_outlet)

sns.set()

fig,ax = plt.subplots(figsize=(10,10))

sns.displot(data['Item_Weight'],rug=True,legend=True)
plt.show()




encoder = LabelEncoder()

data['Item_Identifier'] = encoder.fit_transform(data['Item_Identifier'])

data['Item_Fat_Content'] = encoder.fit_transform(data['Item_Fat_Content'])

data['Item_Type'] = encoder.fit_transform(data['Item_Type'])

data['Outlet_Location_Type'] = encoder.fit_transform(data['Outlet_Location_Type'])

data['Outlet_Type'] = encoder.fit_transform(data['Outlet_Type'])

data['Outlet_Identifier'] = encoder.fit_transform(data['Outlet_Identifier'])



model = XGBRFRegressor()



# train and test

X = data.drop(columns=['Outlet_Location_Type','Item_Outlet_Sales','Outlet_Size'],axis=1)

Y  = data['Item_Outlet_Sales']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=2)


#training model

model = XGBRFRegressor()
model.fit(x_train,y_train)


x_train_predict = model.predict(x_train)


# R_squuerd_value
# Range of zero to one



r2_train = metrics.r2_score(y_train,x_train_predict)
