# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.PARTHASARATHI
RegisterNumber:  212223040144
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
ENCODING :
![328234656-ed87456c-9dd8-418d-a960-1abad11477f2](https://github.com/user-attachments/assets/aa4dbe16-74d3-469b-a408-7e9bc276cb3e)

HEAD():
![328234925-8e2c3fec-2fe3-40c3-923a-1a1c3719e734](https://github.com/user-attachments/assets/9fee9ccd-f92b-4066-82f6-5514c8e9b155)

INFO():
![328235099-b48518c5-c983-44d3-9cc2-14924033aa91](https://github.com/user-attachments/assets/ee894521-94ff-440b-ac6f-7dc9db99d4e0)

isnull().sum():
![328235367-50754f89-e886-48c3-a285-44b76317b605](https://github.com/user-attachments/assets/b01f4c57-be2e-4747-a60a-e55fda158d25)

Prediction of y:
![328235504-8f3a2d63-9aa6-4da2-95c4-d53b87fde998](https://github.com/user-attachments/assets/42d38e8c-6e5d-4ea0-a37f-f1adead3f7b0)

Accuracy:
![328235573-d1dcce16-dc32-4ec2-a042-ce25bee461da](https://github.com/user-attachments/assets/fe143698-d048-4be4-acdc-339b501e81fc)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
