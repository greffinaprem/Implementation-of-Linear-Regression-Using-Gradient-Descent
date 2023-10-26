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

1. Profit prediction graph
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/73e5307f-56af-44c8-9e4f-fc1755e393df)

2. Compute cost value
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/16ccde2d-023e-40a0-8f58-97165aadde32)

3. h(x) value
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/dd7735b4-3f7b-4e1f-a8cc-52d0242570a3)

4. Cost function using gradient descent graph
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/9e43d355-c96f-41c9-8cc8-b0c191c621b3)

5. Profit prediction graph
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/6101dc61-d613-40ab-bbd4-8991ca8cad1c)

6. Profit for the population of 35000
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/da8b0c2d-bcdd-48a2-a3b3-66618454dcf7)

7. Profit for the population of 70000
   ![image](https://github.com/greffinaprem/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475603/c31baf75-40f3-403d-a98f-35b78ffbef4b)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
