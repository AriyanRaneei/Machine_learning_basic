import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("PATH")
#replace null values with null string

mail_data = dataset.where(pd.notnull(dataset),'')
#Label enconding for coneverting spam to 0 and ham 1
mail_data.loc[mail_data['Category']=='spam','Category'] = 0
mail_data.loc[mail_data['Category']=='ham','Category'] = 1
X = mail_data['Message']
Y = mail_data['Category']

x_train,x_test ,y_train,y_test = train_test_split(X,Y,random_state=2,
                                                  test_size=0.4)



#Feature Extraction
#transform text data to numerical value  that we can do in logistictregression model
feature_exe = TfidfVectorizer(min_df = 1, stop_words='english',lowercase=True)
x_train_feature = feature_exe.fit_transform(x_train)
x_test_feature = feature_exe.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
model = LogisticRegression()
model.fit(x_train_feature,y_train)
#evaluation the train model
#prediction on training data
prediction_on_training = model.predict(x_train_feature)
accuracy_training = accuracy_score(y_train,prediction_on_training)


prediction_on_test = model.predict(x_test_feature)
accuracy_test = accuracy_score(y_test,prediction_on_test)


#building prediction system
input_mail = ["Did I forget to tell you ? I want you , I need you, I crave you ... But most of all ... I love you my sweet Arabian steed ... Mmmmmm ... Yummy"]
#convert to numerical
input_data_feature = feature_exe.transform(input_mail)
prediction = model.predict(input_data_feature)

def show(prediction):
    s = None
    if prediction ==1:
        s = 'Ham'
    else:
        s = "Spam"
    print(s)

show(prediction)

