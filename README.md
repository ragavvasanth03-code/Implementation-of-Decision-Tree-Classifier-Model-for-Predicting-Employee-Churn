# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Read the Employee.csv dataset and check its structure and missing values.

2.Convert the categorical salary column into numeric values using LabelEncoder.

3.Select important features like satisfaction_level, number_project, salary, etc., as input (X) and take left as output (Y).

4.Split the dataset into training (80%) and testing (20%) sets.

5.Train a Decision Tree classifier using entropy as the splitting criterion.

6.Predict employee attrition on test data and calculate the model accuracy. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ragavan V
RegisterNumber: 212225240111
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from google.colab import files
import warnings

warnings.filterwarnings("ignore")

uploaded = files.upload()
csv_path = list(uploaded.keys())[0]
data = pd.read_csv(csv_path)

print("Data shape:", data.shape)
display(data.head())

print("\nMissing values:")
print(data.isnull().sum())

if data.isnull().any().any():
    data = data.dropna()

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"].astype(str))

print("\nSalary mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

feature_cols = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]

X = data[feature_cols]
y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=100, stratify=y
)

dt = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, digits=4))

new_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]

pred = dt.predict(new_employee)

print("\nNew Employee Prediction:")
print("Left" if pred[0] == 1 else "Stayed")

plt.figure(figsize=(14, 10))

plot_tree(
    dt,
    feature_names=feature_cols,
    class_names=["Stayed", "Left"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Employee Churn")
plt.show()

importances = pd.Series(
    dt.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nFeature Importances:")
display(importances)

```

## Output:
<img width="1626" height="1052" alt="Screenshot 2026-08-20 105157" src="https://github.com/user-attachments/assets/b0dbdd17-5a7d-4ff5-983d-a38272c84fab" />
<img width="1636" height="1092" alt="Screenshot 2026-08-20 105208" src="https://github.com/user-attachments/assets/23e67568-8799-47ea-b5da-e4506bbaaab8" />
<img width="587" height="438" alt="Screenshot 2026-08-20 105213" src="https://github.com/user-attachments/assets/8d0b7927-ac11-43bd-9007-c9f4379844fd" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
