#為了上傳結果所生
import joblib
model_pretrained = joblib.load('Titanic-Spaceship-20230417.pkl')
import pandas as pd

df_test = pd.read_csv("test.csv")
df_test.drop(['Name','Destination'], axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
df_test.info()


df_test['HomePlanet'].value_counts().idxmax()
df_test['HomePlanet'].fillna(df_test['HomePlanet'].value_counts().idxmax(),inplace = True )

df_test['CryoSleep'].value_counts().idxmax()
df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(),inplace = True )

df_test['VIP'].value_counts().idxmax()
df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(),inplace = True )

df_test['Age'].value_counts().idxmax()
df_test['Age'].fillna(df_test['Age'].value_counts().idxmax(),inplace = True )

df_test['RoomService'] = df_test.groupby('Age')['RoomService'].apply(lambda x: x.fillna(x.median()))
df_test['FoodCourt'] = df_test.groupby('Age')['FoodCourt'].apply(lambda x: x.fillna(x.median()))
df_test['ShoppingMall'] = df_test.groupby('Age')['ShoppingMall'].apply(lambda x: x.fillna(x.median()))
df_test['Spa'] = df_test.groupby('Age')['Spa'].apply(lambda x: x.fillna(x.median()))
df_test['VRDeck'] = df_test.groupby('Age')['VRDeck'].apply(lambda x: x.fillna(x.median()))

df_test.isnull().sum()
df_test.info()

df_test = pd.get_dummies(data=df_test, columns=['VIP', 'HomePlanet'])


df_test.drop('VIP_True', axis=1, inplace=True)
df_test.drop('CryoSleep', axis=1, inplace=True)
 
predictions2 = model_pretrained.predict(df_test)
predictions2 

#prepare submit file
forSubmissionDF = pd.DataFrame(columns=['PassengerId', 'Transported'])
forSubmissionDF
forSubmissionDF['PassengerId'] =  pd.read_csv('test.csv')['PassengerId']
forSubmissionDF['Transported'] = predictions2


forSubmissionDF.to_csv('for_submission_carat.csv', index=False)
