import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
data = pd.read_csv('PATH')


# logistic regression
# work flow of logestic regression

class LogisticRegression():
    
    #declaring learning rate and number of interation
    def __init__(self,learning_rate,no_interation): 
        
        self.learning_rate = learning_rate
        
        self.no_interation =  no_interation
        
        
    # fit function
    def fit(self,X,Y):
        
        self.X = X
        
        self.Y = Y 
        
        # numbebr of datapoints in dataset rows and columns
        self.m,self.n = X.shape
        
        #initating wieght and bias values
        
        self.w = np.zeros(self.n)
       
        self.b = 0
        
        
    
    
    
    def update(self):
        pass
    
    def predict(self):
        pass
    
    





#model  = LogisticRegression()
#model.fit()
