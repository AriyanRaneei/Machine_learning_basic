import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("PATH")


def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].trestbps
        y = points.iloc[i].thalach
        total_error += (y - (m * x + b)) **2
    
    return total_error /float(len(points))
    
    
    


def gradiant_desent(m_now, b_now, points,le):
    m_gradient = 0
    b_gradient = 0
    n  = len(points)
    for i in range(n):
        x =  points.iloc[i].trestbps
        y =  points.iloc[i].thalach
        
        m_gradient += -(2/n) * x *(y-(m_now *x + b_now))
        b_gradient +=  -(2/n) * x *(y-(m_now *x + b_now))
        
    m = m_now - m_gradient *le
    b = b_now - b_gradient * le
    
    return m, b
    
    
    
m = 0
b = 0
l = 0.0001
epochs = 1000
for i in range(epochs):
    if i % 50 ==0:
        print(f"Epoch: {i}")
    m,b = gradiant_desent(m,b,data,l)
    
    
    


#plt.scatter(data.thalach,data.trestbps)
plt.plot(np.array([m * x + b  for  x in range(1,100)]))
