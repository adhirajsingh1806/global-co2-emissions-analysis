import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pre-processing data
df = pd.read_csv(r"https://drive.google.com/uc?export=download&id=1gbTwzt4VkzrscLeBBCns3HcxhhjJ2b22", index_col = False)

#removing duplicates and filtering the data for years greater than or equal to 2000 and for the top 10 countries in the world by population
df = df.drop_duplicates()

df = df.query('Year >= 2000')

df = df.query('Country in ["Russia", "Mexico", "China", "USA", "Brazil", "Bangladesh", "India", "Nigeria", "Pakistan", "Indonesia"]')

#eliminating irrelevant columns and rows with missing values
df = df.drop(columns = ['Flaring','Other'])

df = df.fillna('')

df = df.dropna(subset=['Total','Coal','Oil','Gas','Cement'])

df = df.reset_index(drop=True)

print(df)

print(df.describe())

print(df.nunique())
