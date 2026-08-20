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

#adding more variables 
#Turning semicolon separated language/database lists into a simple count
def count_items(text):
    if text == '':
        return 0
    else:
        items = text.split(';')
        return len(items)

df['num_languages'] = df['LanguageHaveWorkedWith'].fillna('').apply(count_items)
df['num_databases'] = df['DatabaseHaveWorkedWith'].fillna('').apply(count_items)

features = ['WorkExp', 'num_languages', 'num_databases', 'Country']# total variables (x)

df_model2 = df.dropna(subset=features + ['annual_salary_usd'])
print("Rows available for this model:", df_model2.shape)

X2 = df_model2[features]
y2 = df_model2['annual_salary_usd']
X2_encoded = pd.get_dummies(X2, columns=['Country'])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2_encoded, y2, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train) #repeat same regression but with added variables

predictions2 = model2.predict(X2_test)
r2_2 = r2_score(y2_test, predictions2)

print("Intercept:", model2.intercept_)
print("R^2:", r2_2)

#testing on a new person
new_person = pd.DataFrame([{
    'WorkExp': 7,
    'num_languages': 4,
    'num_databases': 3,
    'Country': 'United Kingdom of Great Britain and Northern Ireland'
}])

new_person_encoded = pd.get_dummies(new_person, columns=['Country'])
new_person_encoded = new_person_encoded.reindex(columns=X2_train.columns, fill_value=0)
predicted_salary = model2.predict(new_person_encoded)[0]
#forces the new person's row to have exactly the same columns as training data
print("Predicted salary for new person:", round(predicted_salary, 2))