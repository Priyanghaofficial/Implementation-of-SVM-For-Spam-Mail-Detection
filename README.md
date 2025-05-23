# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Priyangha G
RegisterNumber:212223040157 
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
## data
![image](https://github.com/user-attachments/assets/33c17bad-6e14-4095-83a7-6bc3a312954a)

## confusion matrix
![445421116-bc7d87f8-2d4e-44c6-b432-ba485c265491](https://github.com/user-attachments/assets/efc87242-c3f1-48f4-8240-edc5a3c38802)
## accuracy
![445420950-26bdd6ed-3c7e-4861-b52a-a5489c1302d7](https://github.com/user-attachments/assets/d1b52a4e-add4-4533-b2ec-914450655b75)

## classification report
![445421183-f0625d7f-001e-4061-8307-f31ae4a13740](https://github.com/user-attachments/assets/7411653b-9ca9-4ebf-9ba6-ec3f126c97bd)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
