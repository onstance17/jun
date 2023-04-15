#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
#尋找變數跟存活的關係性之類的
#整理好的資料的空值常是NaN
df = pd.read_csv("train.csv")
df.head()
df.info()

#遺漏值處理、格式轉換
#扔掉欄位 axis : {0 or 'index', 1 or 'columns'}
df.drop(['Name','Cabin','Destination'], axis=1, inplace=True)
#'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'
#視覺化觀察 可自由換變數比較 但目前要先找數值類的！
sns.pairplot(df[['Transported', 'HomePlanet']], dropna=True)

sns.pairplot(df[['Transported', 'VIP']], dropna=True)

sns.pairplot(df[['Transported', 'Age']], dropna=True)

sns.pairplot(df[['Transported', 'RoomService']], dropna=True)



#以下用Transported來分組 得到數值化平均值
df.groupby('Transported').mean()

#各欄位分布狀況累計
#每個人都不一樣的去計算是沒意義的
df['HomePlanet'].value_counts()
df['CryoSleep'].value_counts()
df['VIP'].value_counts()
df['RoomService'].value_counts()

#處理資料
#isnull是跑空值(缺失)與否的true or false
#缺失超過一半的欄位就丟掉 除非關鍵因素就往前重新蒐集
df.isnull().sum()
len(df)/2
df.isnull().sum() > (len(df)/2)

#既有資料填補可用中位數
#'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'
df['Age'].isnull().value_counts()
df['HomePlanet'].isnull().value_counts()
df['CryoSleep'].isnull().value_counts()
df['RoomService'].isnull().value_counts()
df['VIP'].isnull().value_counts()
df['FoodCourt'].isnull().value_counts()
df['ShoppingMall'].isnull().value_counts()
df['Spa'].isnull().value_counts()
df['VRDeck'].isnull().value_counts()

df.groupby('Transported')['Age'].median().plot(kind='bar')
df.groupby('Transported')['FoodCourt'].median().plot(kind='bar')
df.groupby('Transported')['RoomService'].median().plot(kind='bar')
df.groupby('Transported')['FoodCourt'].median().plot(kind='bar')
df.groupby('Transported')['ShoppingMall'].median().plot(kind='bar')
df.groupby('Transported')['Spa'].median().plot(kind='bar')
df.groupby('Transported')['VRDeck'].median().plot(kind='bar')

df.isnull().sum()

#idxmax指得到id目的地最大值
df['HomePlanet'].value_counts().idxmax()
df['HomePlanet'].fillna(df['HomePlanet'].value_counts().idxmax(),inplace = True )

df['CryoSleep'].value_counts().idxmax()
df['CryoSleep'].fillna(df['CryoSleep'].value_counts().idxmax(),inplace = True )

df['VIP'].value_counts().idxmax()
df['VIP'].fillna(df['VIP'].value_counts().idxmax(),inplace = True )

df['Age'].value_counts().idxmax()
df['Age'].fillna(df['Age'].value_counts().idxmax(),inplace = True )


df['RoomService'] = df.groupby('Age')['RoomService'].apply(lambda x: x.fillna(x.median()))
df['FoodCourt'] = df.groupby('Age')['FoodCourt'].apply(lambda x: x.fillna(x.median()))
df['ShoppingMall'] = df.groupby('Age')['ShoppingMall'].apply(lambda x: x.fillna(x.median()))
df['Spa'] = df.groupby('Age')['Spa'].apply(lambda x: x.fillna(x.median()))
df['VRDeck'] = df.groupby('Age')['VRDeck'].apply(lambda x: x.fillna(x.median()))

df.isnull().sum()
#getdummies指欄位值變化類型小的
df = pd.get_dummies(data=df, columns=['VIP', 'HomePlanet'])
df.head

df.drop('VIP_True', axis=1, inplace=True)
df.head
df.info()
df.corr()
x = df.drop(['Transported', 'CryoSleep'], axis = 1)
y = df['Transported']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=67)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

accuracy_score(y_test, predictions)
recall_score(y_test, predictions)
precision_score(y_test,predictions)

pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Transported','Predict Transported'], index=['True not Transported', 'True Transported'])
#pkl輸出成一個檔案、黑盒子
import joblib
joblib.dump(lr, 'Titanic-Spaceship-20230417.pkl' ,compress=3)