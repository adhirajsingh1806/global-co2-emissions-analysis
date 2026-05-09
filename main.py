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

print(df.sort_values(by = 'Per Capita', ascending = False).head(10))

#correlation analysis and heatmap
print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True), annot = True)

plt.show()

#pattern identification

sns.lineplot(data = df, x = 'Year', y = 'Total', hue = 'Country')

plt.show()

df2 =df.groupby('Country')[['Cement','Gas','Oil','Coal','Total']].mean(numeric_only=True).sort_values(by = 'Total', ascending = False)

df3 = df2.transpose()

df3.plot()

plt.show()

df.boxplot()

plt.show()

df4 = df.set_index('Country')

df4.plot()

plt.show()
