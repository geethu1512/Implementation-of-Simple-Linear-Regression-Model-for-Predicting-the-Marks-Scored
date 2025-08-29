# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph.

5.Predict the regression for the marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GEETHU R
RegisterNumber:  212224040089
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())

X = df.iloc[:, :-1].values  # Hours (input)
Y = df.iloc[:, -1].values   # Scores (output)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("Predicted Scores:", Y_pred)
print("Actual Scores:", Y_test)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(X_test, Y_test, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")  # Line is from training
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("MSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)

```

## Output:
<img width="687" height="351" alt="1" src="https://github.com/user-attachments/assets/5283afcb-0ba9-4aef-85de-c7a937e39d44" />
<img width="782" height="572" alt="2" src="https://github.com/user-attachments/assets/a4b34b95-375a-46a0-b5a8-437a31344221" />
<img width="806" height="671" alt="3" src="https://github.com/user-attachments/assets/db5492ef-f72f-4932-b0b6-4b581f3ecc72" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
