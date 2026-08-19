
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('data/survey.csv')
print("Before cleaning:", df.shape) #load and print raw data 

df = df.dropna(subset=['annual_salary_usd'])#drop empty rows in salary column 
print("After dropping missing salary:", df.shape)

df = df[(df['annual_salary_usd'] >= 1000) & (df['annual_salary_usd'] <= 1_000_000)] 
#setting boundaries between $1000 - $1,000,000 
print("After removing unrealistic salaries:", df.shape) 

df.to_csv('cleaned_survey.csv', index=False) #saving cleaned dataset

df_model = df.dropna(subset=['WorkExp', 'annual_salary_usd']) #Drop rows missing in WorkExp & salary
print("Rows available for this model:", df_model.shape) 

X = df_model[['WorkExp']] 
y = df_model['annual_salary_usd']
#assigning variables - X = input feature, y = what we're predicting 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

model = LinearRegression()
model.fit(X_train, y_train) #model learns the pattern from the training 80%

predictions = model.predict(X_test) # model guesses salaries for the test 20% 
r2 = r2_score(y_test, predictions) #compare its guesses to the real salaries it was hidden from

print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R^2:", r2)