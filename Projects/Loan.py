import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_data = pd.read_csv("path")


#print(loan_data[' loan_status'].value_counts())
#print(loan_data.shape)

# statistical measuares
#print(loan_data.describe())
#Label encoding

loan_data.loc[loan_data[' education']==" Graduate",' education'] = 1
loan_data.loc[loan_data[' education']==" Not Graduate",' education'] = 0

loan_data.loc[loan_data[' loan_status']==" Approved",' loan_status'] = 1
loan_data.loc[loan_data[' loan_status']==" Rejected",' loan_status'] = 0
print(loan_data.head(10))
#Data visualization
loan_data.loc[loan_data[' self_employed']==" Yes",' self_employed'] = 1
loan_data.loc[loan_data[' self_employed']== " No",' self_employed'] = 0



#education loan_status

#sns.countplot(data=loan_data, x=' education',hue=' loan_status')
#sns.lineplot(data=loan_data, x=' education',y=' loan_status')
sns.countplot(data=loan_data, x=' self_employed',hue=' loan_status')
X = loan_data.drop(columns=['loan_id',' loan_status'])

Y = loan_data[' loan_status']
Y = Y.astype('int')
x_train,x_test , y_train,y_test = train_test_split(X,Y,stratify=Y,test_size=0.1,
                                                   random_state=3)

#Training the support vector machine modedl


classifier = svm.SVC(kernel='linear')
# training the support vector machine model


classifier.fit(x_train,y_train)



x_train_prediction = classifier.predict(x_train)
x_acc = accuracy_score(x_train_prediction,y_train)






x_test_prediction = classifier.predict(x_test)
x_test_acc = accuracy_score(x_test_prediction,y_test)
