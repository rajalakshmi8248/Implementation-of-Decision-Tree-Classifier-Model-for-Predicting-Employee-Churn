# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Raja Lakshmi E
RegisterNumber: 212222220033 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## data.head()
![Screenshot 2024-04-02 093910](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/a5281883-eed0-439e-b4fa-922bbdfbed98)

## data.info()
![Screenshot 2024-04-02 093929](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/7c989f04-c2db-4949-a088-ad348bbec7c7)
## data.isnull().sum()
![Screenshot 2024-04-02 093941](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/3d23a2f9-d8bf-4ca2-b00c-3e581ca80770)
## data value count
![Screenshot 2024-04-02 094002](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/10ee6c65-157b-4f16-95bc-0dc9664953a9)
## data.head() for salary
![Screenshot 2024-04-02 094021](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/6492ed67-18af-46c1-8141-d168878dd59d)
## x.head()
![Screenshot 2024-04-02 094037](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/b3b08f53-9a93-4cdc-8403-6c601bae877e)
## accuracy value
![Screenshot 2024-04-02 094051](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/7026b909-59d8-4316-afaf-22270aa8dd90)
## data prediction
![Screenshot 2024-04-02 094125](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122860827/5f8fee59-9658-437e-8a98-8405059f69c6)












## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
