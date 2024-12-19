import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv("PATH")


X = data.iloc[:, [3, 4]].values

# FIND THE EXACT NUMBER OF CLUSTER

# WCSS ==> WITHIN CLUSTERS

fig,ax = plt.subplots(3,1,figsize=(20,20))

ax[0].plot(data.iloc[:, 3].values,data.iloc[:, 4].values, "*")

# ELBO METHOD FOR FINDING FOR DIFFERENT NUMBER OF CLUSTERS



wssc  = []


for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=10)
    kmeans.fit(X)
    
    wssc.append(kmeans.inertia_)
    
    
print(wssc)


sns.set()
ax[1].plot(range(1,11),wssc)



# from picture we can understand that the number of clusters is 5


KMeans = KMeans(n_clusters=5,init='k-means++',random_state=10)



Y = KMeans.fit_predict(X)


ax[2].scatter(X[Y==0,0],X[Y==0,1],color="green",s=50)
ax[2].scatter(X[Y==1,0],X[Y==1,1],color="green",s=50)
ax[2].scatter(X[Y==2,0],X[Y==2,1],color="green",s=50)
ax[2].scatter(X[Y==3,0],X[Y==3,1],color="green",s=50)
ax[2].scatter(X[Y==4,0],X[Y==4,1],color="green",s=50)


Label = LabelEncoder()
data["Gender"] = Label.fit_transform(data["Gender"])




data['Cluster'] = Y




X_supervised = data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].values



Y_supervised = data['Cluster'].values





x_train,x_test,y_train,y_test = train_test_split(X_supervised,Y_supervised)


model = LogisticRegression()

model.fit(x_train,y_train)

x_train_predict = model.predict(x_train)


acc = accuracy_score(y_train,x_train_predict)



model.fit(x_test,y_test)

x_test_predict = model.predict(x_test)


acc_test = accuracy_score(y_test,x_tes
