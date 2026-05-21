import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet

plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

top_countries = ["Russia", "Mexico", "China", "USA", "Brazil", "Bangladesh", "India", "Nigeria", "Pakistan", "Indonesia"]
start_year = 2000

#Data Cleaning and Preparation

df = pd.read_csv(r"https://drive.google.com/uc?export=download&id=1gbTwzt4VkzrscLeBBCns3HcxhhjJ2b22", index_col=False)

df = df.drop_duplicates()

df = df.query('Year >= @start_year')

df = df.query('Country in @top_countries')

df = df.drop(columns=['Flaring', 'Other'])

df = df.fillna('')

df = df.dropna(subset=['Total', 'Coal', 'Oil', 'Gas', 'Cement'])

df = df.reset_index(drop=True)

print(df)

print(df.isnull().sum())

# outlier detection using the IQR method
fuel_cols = ['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Per Capita']

Q1 = df[fuel_cols].quantile(0.25)
Q3 = df[fuel_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = (df[fuel_cols] < (Q1 - 1.5 * IQR)) | (df[fuel_cols] > (Q3 + 1.5 * IQR))

print(outliers.sum())

#Exploratory Data Analysis

print(df.describe())

print(df.nunique())

print(df.sort_values(by='Per Capita', ascending=False).head(10))

print(df.corr(numeric_only=True))

#Data Visualisation

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

sns.lineplot(data=df, x='Year', y='Total', hue='Country')
plt.title('Total CO2 Emissions Over Time')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.tight_layout()
plt.show()

df2 = df.groupby('Country')[['Cement', 'Gas', 'Oil', 'Coal', 'Total']].mean(numeric_only=True).sort_values(by='Total', ascending=False)

df4 = df.set_index('Country')
df4[['Coal', 'Oil', 'Gas', 'Cement', 'Total']].plot()
plt.title('Emissions by Country')
plt.tight_layout()
plt.show()

df.boxplot(column=fuel_cols)
plt.title('Distribution and Outliers by Source')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df2['Total'].plot(kind='bar')
plt.title('Average Total Emissions by Country')
plt.ylabel('Average Total CO2 Emissions (MtCO2)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.groupby('Country')['Total'].sum().sort_values(ascending=False).plot(kind='pie', autopct='%1.1f%%')
plt.title('Share of Total Emissions by Country (2000-2021)')
plt.ylabel('')
plt.tight_layout()
plt.show()

source_totals = df[['Coal', 'Oil', 'Gas', 'Cement']].sum()
source_totals.plot(kind='pie', autopct='%1.1f%%')
plt.title('Share of Total Emissions by Source')
plt.ylabel('')
plt.tight_layout()
plt.show()

df['Total'].plot(kind='hist', bins=20)
plt.title('Distribution of Total Emissions')
plt.xlabel('Total CO2 Emissions (MtCO2)')
plt.tight_layout()
plt.show()

df['Per Capita'].plot(kind='hist', bins=20)
plt.title('Distribution of Per Capita Emissions')
plt.xlabel('Per Capita CO2 Emissions')
plt.tight_layout()
plt.show()

#Model Building and Forecasting

world = df.groupby('Year')['Total'].sum().reset_index()

#Linear Regression
X = world[['Year']]
y = world['Total']

lr = LinearRegression()
lr.fit(X, y)

future_years = pd.DataFrame({'Year': np.arange(world['Year'].min(), 2031)})
lr_pred = lr.predict(future_years)

print('Linear Regression R2:', r2_score(y, lr.predict(X)))
print('Linear Regression MAE:', mean_absolute_error(y, lr.predict(X)))
print('Annual change (MtCO2/year):', lr.coef_[0])

plt.scatter(world['Year'], world['Total'], label='Actual')
plt.plot(future_years['Year'], lr_pred, color='blue', label='Linear Fit & Forecast')
plt.title('Linear Regression Forecast of Total Emissions')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#Prophet
prophet_df = world.rename(columns={'Year': 'ds', 'Total': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

m = Prophet()
m.fit(prophet_df)

future = m.make_future_dataframe(periods=9, freq='YS')
forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(9))

m.plot(forecast)
plt.title('Prophet Forecast of Total Emissions')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.tight_layout()
plt.show()
