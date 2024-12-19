import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , mean_absolute_error
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

heart_data = pd.read_csv("PATH")
#statistical about the data
print(heart_data.describe())

#checking the distribution of dataset
print(heart_data['target'].value_counts())





# 1 means Degective heart
# 0 means healthu heart
# spiliting feature and target
# if you drop colum axis=1 but for row axis most be 0
X = heart_data.drop(columns='target')
Y = heart_data['target']


# Spliting data into training data and test data

x_train , x_test , y_train ,y_test = train_test_split(X,Y,test_size=0.2,
                                                      stratify=Y,random_state=2)




# train ML Model
#Logestic regression model

model = LogisticRegression()
# its find the pattern between the models
model.fit(x_train,y_train)

x_predict = model.predict(x_train)
x_acc = accuracy_score(x_predict,y_train)



model.fit(x_test,y_test)

x_test_predict = model.predict(x_test)
x_test_acc = accuracy_score(x_test_predict,y_test)

# the accuracy between x_train and y-train should be same in the best way
# if the differences is so heuge will cause the overfitting


#build predictive system

input_data = (57,0,0,120,354,0,1,163,1,0.6,2,0,2,)



# change input data to numpy array
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)


x_input = model.predict(input_data)
