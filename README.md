# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler
2. Type the required program
3. Print the program.
4. End the program. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:GREFFINA SANCHEZ P 
RegisterNumber: 212222040048

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0))) 
*/
```

## Output:

![269243654-ad4b83a1-00ea-4f72-990f-c196fd3bf178](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/72b278a0-9e9f-4edd-b5be-324662420d47)

![269244030-15b30287-f989-4456-8a8e-4bd3e5f5e2f5](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/ec8dcc01-ce11-462a-afc5-448b7e7da070)

![269244268-a04f5f38-aecc-4d3e-87cd-11af9cac187e](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/26944190-9a6f-4115-973c-e39e71770c1d)

![269244793-7014b73e-e6bb-42ec-8560-8df7bb841d75](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/cefa285a-6eae-4337-9678-c2404572bae1)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
