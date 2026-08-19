
import pandas as pd
import numpy as np

df = pd.read_csv('data/survey.csv')
print("Before cleaning:", df.shape) #load and print raw data 

df = df.dropna(subset=['annual_salary_usd'])#drop empty rows in salary column 
print("After dropping missing salary:", df.shape)

df = df[(df['annual_salary_usd'] >= 1000) & (df['annual_salary_usd'] <= 1_000_000)] 
#setting boundaries between $1000 - $1,000,000 
print("After removing unrealistic salaries:", df.shape) 

df.to_csv('cleaned_survey.csv', index=False) #saving cleaned dataset