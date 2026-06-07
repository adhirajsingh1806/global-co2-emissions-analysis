import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

top_countries = ["Russia", "Mexico", "China", "USA", "Brazil", "Bangladesh", "India", "Nigeria", "Pakistan", "Indonesia"]
start_year = 2000
split_year = 2017
forecast_end = 2030

#data cleaning and preparation

df = pd.read_csv(r"https://drive.google.com/uc?export=download&id=1gbTwzt4VkzrscLeBBCns3HcxhhjJ2b22", index_col=False)

df = df.drop_duplicates()
df = df.query('Year >= @start_year')
df = df.query('Country in @top_countries')
df = df.drop(columns=['Flaring', 'Other'])
df = df.fillna('')
df = df.dropna(subset=['Total', 'Coal', 'Oil', 'Gas', 'Cement'])
df = df.reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())
print("\nNull values:\n", df.isnull().sum())

# Outlier detection using the IQR method
fuel_cols = ['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Per Capita']
fuel_categories = ['Total', 'Coal', 'Oil', 'Gas', 'Cement']

Q1 = df[fuel_cols].quantile(0.25)
Q3 = df[fuel_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = (df[fuel_cols] < (Q1 - 1.5 * IQR)) | (df[fuel_cols] > (Q3 + 1.5 * IQR))
print("\nOutliers per column:\n", outliers.sum())

#exploratory data analysis

print("\nDescriptive Statistics")
print(df.describe())
print("\nUnique values:\n", df.nunique())
print("\nTop 10 by Per Capita:\n", df.sort_values(by='Per Capita', ascending=False).head(10))
print("\nCorrelation matrix:\n", df.corr(numeric_only=True))

#data visualisation

#correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

#total emissions over time by country
sns.lineplot(data=df, x='Year', y='Total', hue='Country')
plt.title('Total CO2 Emissions Over Time')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.tight_layout()
plt.show()

#average emissions by country
df2 = df.groupby('Country')[['Cement', 'Gas', 'Oil', 'Coal', 'Total']].mean(numeric_only=True).sort_values(by='Total', ascending=False)

#emissions by source 
df4 = df.set_index('Country')
df4[['Coal', 'Oil', 'Gas', 'Cement']].plot()
plt.title('Emissions by Country')
plt.tight_layout()
plt.show()

#box plots for outliers
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), fuel_cols):
    df.boxplot(column=col, ax=ax)
    ax.set_title(col)
    ax.set_ylabel('tCO2 per person' if col == 'Per Capita' else 'MtCO2')
fig.suptitle('Distribution and Outliers by Source')
plt.tight_layout()
plt.show()

#bar chart: average total emissions
df2['Total'].plot(kind='bar', color='green')
plt.title('Average Total Emissions by Country')
plt.ylabel('Average Total CO2 Emissions (MtCO2)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#pie chart: share by country
df.groupby('Country')['Total'].sum().sort_values(ascending=False).plot(kind='pie', autopct='%1.1f%%')
plt.title('Share of Total Emissions by Country (2000-2021)')
plt.ylabel('')
plt.tight_layout()
plt.show()

#pie chart: share by source
source_totals = df[['Coal', 'Oil', 'Gas', 'Cement']].sum()
source_totals.plot(kind='pie', autopct='%1.1f%%')
plt.title('Share of Total Emissions by Source')
plt.ylabel('')
plt.tight_layout()
plt.show()

#histograms
df['Total'].plot(kind='hist', bins=20, color='blue')
plt.title('Distribution of Total Emissions')
plt.xlabel('Total CO2 Emissions (MtCO2)')
plt.tight_layout()
plt.show()

df['Per Capita'].plot(kind='hist', bins=20, color='blue')
plt.title('Distribution of Per Capita Emissions')
plt.xlabel('Per Capita CO2 Emissions')
plt.tight_layout()
plt.show()

#model building and forecasting

world = df.groupby('Year')['Total'].sum().reset_index()
train = world[world['Year'] <= split_year]
test = world[world['Year'] > split_year]
test_years = list(range(split_year + 1, 2022))
actual_test = test['Total'].values
future_years_all = np.arange(world['Year'].min(), forecast_end + 1)

print(f"\nTrain set: {train['Year'].min()}-{train['Year'].max()} ({len(train)} years)")
print(f"Test set:  {test['Year'].min()}-{test['Year'].max()} ({len(test)} years)")

model_results = {}

#linear regression (aggregate)
print("MODEL 1: LINEAR REGRESSION (Aggregate)")

X_train = train[['Year']]
y_train = train['Total']
X_test = test[['Year']]

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_train_pred = lr.predict(X_train)
lr_test_pred = lr.predict(X_test)
lr_future_pred = lr.predict(pd.DataFrame({'Year': future_years_all}))

lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(actual_test, lr_test_pred)
lr_test_r2 = r2_score(actual_test, lr_test_pred)

print(f"Train : R²: {lr_train_r2:.4f}, MAE: {lr_train_mae:.2f} MtCO2")
print(f"Test  : R²: {lr_test_r2:.4f}, MAE: {lr_test_mae:.2f} MtCO2")
print(f"Annual change: {lr.coef_[0]:.2f} MtCO2/year")

model_results['Linear Regression (agg)'] = {
    'train_mae': lr_train_mae, 'train_r2': lr_train_r2,
    'test_mae': lr_test_mae, 'test_r2': lr_test_r2
}

plt.figure(figsize=(10, 5))
plt.scatter(world['Year'], world['Total'], label='Actual', zorder=5)
plt.plot(future_years_all, lr_future_pred, color='blue', linestyle='--', label='Linear Fit & Forecast')
plt.axvline(x=split_year, color='gray', linestyle=':', label='Train/Test Split')
plt.title('Model 1: Linear Regression Forecast')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#prophet (aggregate)
print("MODEL 2: PROPHET (Aggregate)")

prophet_train_df = train.rename(columns={'Year': 'ds', 'Total': 'y'})
prophet_train_df['ds'] = pd.to_datetime(prophet_train_df['ds'], format='%Y')

m_agg = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
m_agg.fit(prophet_train_df)

#in sample
in_sample_agg = m_agg.predict(prophet_train_df[['ds']])
prophet_agg_train_mae = mean_absolute_error(prophet_train_df['y'], in_sample_agg['yhat'])
prophet_agg_train_r2 = r2_score(prophet_train_df['y'], in_sample_agg['yhat'])

#out of sample
prophet_test_ds = pd.DataFrame({'ds': pd.to_datetime(test_years, format='%Y')})
out_sample_agg = m_agg.predict(prophet_test_ds)
prophet_agg_test_mae = mean_absolute_error(actual_test, out_sample_agg['yhat'].values)
prophet_agg_test_r2 = r2_score(actual_test, out_sample_agg['yhat'].values)

print(f"Train : R²: {prophet_agg_train_r2:.4f}, MAE: {prophet_agg_train_mae:.2f} MtCO2")
print(f"Test  : R²: {prophet_agg_test_r2:.4f}, MAE: {prophet_agg_test_mae:.2f} MtCO2")

model_results['Prophet (agg)'] = {
    'train_mae': prophet_agg_train_mae, 'train_r2': prophet_agg_train_r2,
    'test_mae': prophet_agg_test_mae, 'test_r2': prophet_agg_test_r2
}

# Full forecast for plotting
future_full = pd.DataFrame({'ds': pd.to_datetime(future_years_all, format='%Y')})
forecast_full_agg = m_agg.predict(future_full)

plt.figure(figsize=(10, 5))
plt.scatter(world['Year'], world['Total'], label='Actual', zorder=5)
plt.plot(future_years_all, forecast_full_agg['yhat'], color='orange', label='Prophet Forecast')
plt.fill_between(future_years_all, forecast_full_agg['yhat_lower'], forecast_full_agg['yhat_upper'], alpha=0.2, color='orange')
plt.axvline(x=split_year, color='gray', linestyle=':', label='Train/Test Split')
plt.title('Model 2: Prophet (Aggregate) Forecast')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#ARIMA (aggregate)
print("MODEL 3: ARIMA (Aggregate)")

train_world_vals = train['Total'].values

#grid search for best ARIMA order
best_aic = np.inf
best_order_agg = (1, 1, 0)
for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                model = ARIMA(train_world_vals, order=(p, d, q))
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order_agg = (p, d, q)
            except:
                continue

print(f"Best ARIMA order: {best_order_agg}")

model_agg_arima = ARIMA(train_world_vals, order=best_order_agg)
fit_agg_arima = model_agg_arima.fit()

#in sample
arima_agg_train_pred = fit_agg_arima.fittedvalues
arima_agg_train_mae = mean_absolute_error(train_world_vals, arima_agg_train_pred)
arima_agg_train_r2 = r2_score(train_world_vals, arima_agg_train_pred)

#out of sample
arima_agg_test_pred = fit_agg_arima.forecast(steps=len(test_years))
arima_agg_test_mae = mean_absolute_error(actual_test, arima_agg_test_pred)
arima_agg_test_r2 = r2_score(actual_test, arima_agg_test_pred)

print(f"Train : R²: {arima_agg_train_r2:.4f}, MAE: {arima_agg_train_mae:.2f} MtCO2")
print(f"Test  : R²: {arima_agg_test_r2:.4f}, MAE: {arima_agg_test_mae:.2f} MtCO2")

model_results['ARIMA (agg)'] = {
    'train_mae': arima_agg_train_mae, 'train_r2': arima_agg_train_r2,
    'test_mae': arima_agg_test_mae, 'test_r2': arima_agg_test_r2
}

#future forecast
model_full_arima = ARIMA(world['Total'].values, order=best_order_agg)
fit_full_arima = model_full_arima.fit()
n_future_agg = int(forecast_end - world['Year'].max())
arima_agg_future = fit_full_arima.forecast(steps=n_future_agg)
arima_agg_plot_years = np.arange(world['Year'].max() + 1, forecast_end + 1)

plt.figure(figsize=(10, 5))
plt.scatter(world['Year'], world['Total'], label='Actual', zorder=5)
plt.plot(train['Year'], fit_agg_arima.fittedvalues, color='green', label='ARIMA Fitted')
plt.plot(np.append(test_years, arima_agg_plot_years),
         np.append(arima_agg_test_pred, arima_agg_future),
         color='green', linestyle='--', label='ARIMA Forecast')
plt.axvline(x=split_year, color='gray', linestyle=':', label='Train/Test Split')
plt.title(f'Model 3: ARIMA{best_order_agg} (Aggregate) Forecast')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#prophet (country-level)
print("MODEL 4: PROPHET (Country-Level)")

prophet_country_preds_test = {}
prophet_country_preds_future = {}

for country in top_countries:
    cdf = df[df['Country'] == country][['Year', 'Total']].copy()
    train_c = cdf[cdf['Year'] <= split_year]

    pdf = train_c.rename(columns={'Year': 'ds', 'Total': 'y'})
    pdf['ds'] = pd.to_datetime(pdf['ds'], format='%Y')

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pdf)

    #test prediction
    future_test = pd.DataFrame({'ds': pd.to_datetime(test_years, format='%Y')})
    fc_test = m.predict(future_test)
    prophet_country_preds_test[country] = fc_test['yhat'].values

    #future forecast
    cdf_full = cdf.rename(columns={'Year': 'ds', 'Total': 'y'})
    cdf_full['ds'] = pd.to_datetime(cdf_full['ds'], format='%Y')
    m_full = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m_full.fit(cdf_full)
    future_all = pd.DataFrame({'ds': pd.to_datetime(np.arange(start_year, forecast_end + 1), format='%Y')})
    fc_full = m_full.predict(future_all)
    prophet_country_preds_future[country] = fc_full['yhat'].values

    actual_c = cdf[cdf['Year'] > split_year]['Total'].values
    mae_c = mean_absolute_error(actual_c, fc_test['yhat'].values)
    print(f"  {country:<12} Test MAE: {mae_c:>8.2f} MtCO2")

#aggregate
prophet_cl_test_total = pd.DataFrame(prophet_country_preds_test, index=test_years).sum(axis=1)
prophet_cl_test_mae = mean_absolute_error(actual_test, prophet_cl_test_total.values)
prophet_cl_test_r2 = r2_score(actual_test, prophet_cl_test_total.values)

prophet_cl_future_total = pd.DataFrame(prophet_country_preds_future, index=np.arange(start_year, forecast_end + 1)).sum(axis=1)

print(f"\n  Aggregated Test MAE: {prophet_cl_test_mae:,.2f} MtCO2")
print(f"  Aggregated Test R²:  {prophet_cl_test_r2:.4f}")

model_results['Prophet (country-level)'] = {
    'train_mae': np.nan, 'train_r2': np.nan,
    'test_mae': prophet_cl_test_mae, 'test_r2': prophet_cl_test_r2
}

plt.figure(figsize=(10, 5))
plt.scatter(world['Year'], world['Total'], label='Actual', zorder=5)
plt.plot(np.arange(start_year, forecast_end + 1), prophet_cl_future_total.values, color='purple', linestyle='--', label='Prophet Country-Level Forecast')
plt.axvline(x=split_year, color='gray', linestyle=':', label='Train/Test Split')
plt.title('Model 4: Prophet (Country-Level Aggregated) Forecast')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#ARIMA (country-level)
print("MODEL 5: ARIMA (Country-Level)")

arima_country_preds_test = {}
arima_country_preds_future = {}
arima_best_orders = {}

for country in top_countries:
    cdf = df[df['Country'] == country][['Year', 'Total']].copy().sort_values('Year')
    train_c = cdf[cdf['Year'] <= split_year]['Total'].values
    test_c = cdf[cdf['Year'] > split_year]['Total'].values
    full_c = cdf['Total'].values
    n_test = len(test_c)

    #grid search for best ARIMA order
    best_aic = np.inf
    best_order = (1, 1, 0)
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    m = ARIMA(train_c, order=(p, d, q))
                    f = m.fit()
                    if f.aic < best_aic:
                        best_aic = f.aic
                        best_order = (p, d, q)
                except:
                    continue

    arima_best_orders[country] = best_order

    #test prediction
    model = ARIMA(train_c, order=best_order)
    fit = model.fit()
    fc_test = fit.forecast(steps=n_test)
    arima_country_preds_test[country] = fc_test

    #future forecast
    n_future = int(forecast_end - cdf['Year'].max())
    model_full = ARIMA(full_c, order=best_order)
    fit_full = model_full.fit()
    fc_future = fit_full.forecast(steps=n_future)
    #combine historical + future for plotting
    arima_country_preds_future[country] = np.concatenate([full_c, fc_future])

    mae_c = mean_absolute_error(test_c, fc_test)
    print(f"  {country:<12} Order: {best_order} | Test MAE: {mae_c:>8.2f} MtCO2")

#aggregate
arima_cl_test_total = pd.DataFrame(arima_country_preds_test, index=test_years).sum(axis=1)
arima_cl_test_mae = mean_absolute_error(actual_test, arima_cl_test_total.values)
arima_cl_test_r2 = r2_score(actual_test, arima_cl_test_total.values)

arima_cl_future_total = pd.DataFrame(arima_country_preds_future, index=np.arange(start_year, forecast_end + 1)).sum(axis=1)

print(f"\n  Aggregated Test MAE: {arima_cl_test_mae:,.2f} MtCO2")
print(f"  Aggregated Test R²:  {arima_cl_test_r2:.4f}")

#year-by-year test predictions 
print("\n  Year-by-year test predictions:")
for yr, act, pred in zip(test_years, actual_test, arima_cl_test_total):
    err = abs(act - pred)
    pct = err / act * 100
    print(f"    {yr}: Actual={act:,.0f}  Predicted={pred:,.0f}  Error={err:,.0f} ({pct:.1f}%)")

model_results['ARIMA (country-level)'] = {
    'train_mae': np.nan, 'train_r2': np.nan,
    'test_mae': arima_cl_test_mae, 'test_r2': arima_cl_test_r2
}

plt.figure(figsize=(10, 5))
plt.scatter(world['Year'], world['Total'], label='Actual', zorder=5)
plt.plot(np.arange(start_year, forecast_end + 1), arima_cl_future_total.values, color='red', linestyle='--', label='ARIMA Country-Level Forecast')
plt.axvline(x=split_year, color='gray', linestyle=':', label='Train/Test Split')
plt.title('Model 5: ARIMA (Country-Level Aggregated) Forecast')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend()
plt.tight_layout()
plt.show()

#combined forecast comparison

plt.figure(figsize=(14, 7))
plt.scatter(world['Year'], world['Total'], color='black', label='Actual', zorder=5, s=40)

#linear regression (aggregate)
plt.plot(future_years_all, lr_future_pred, color='blue', linestyle='--', alpha=0.7, label=f'Linear Regression (MAE={lr_test_mae:,.0f})')

#prophet (aggregate)
plt.plot(future_years_all, forecast_full_agg['yhat'], color='orange', linestyle='--', alpha=0.7, label=f'Prophet Agg (MAE={prophet_agg_test_mae:,.0f})')

#ARIMA (aggregate)
arima_agg_all = np.concatenate([world['Total'].values, arima_agg_future])
arima_agg_all_years = np.arange(start_year, forecast_end + 1)
plt.plot(arima_agg_all_years, arima_agg_all, color='green', linestyle='--', alpha=0.7, label=f'ARIMA Agg (MAE={arima_agg_test_mae:,.0f})')

#prophet (country-level)
plt.plot(np.arange(start_year, forecast_end + 1), prophet_cl_future_total.values, color='purple', linestyle='--', alpha=0.7, label=f'Prophet Country (MAE={prophet_cl_test_mae:,.0f})')

#arima (country-level)
plt.plot(np.arange(start_year, forecast_end + 1), arima_cl_future_total.values, color='red', linewidth=2.5, label=f'ARIMA Country (MAE={arima_cl_test_mae:,.0f}) ★ BEST')

plt.axvline(x=split_year, color='gray', linestyle=':', alpha=0.5, label='Train/Test Split')
plt.axvline(x=2021, color='gray', linestyle='-.', alpha=0.5, label='Data Ends')
plt.title('CO2 Emissions Forecast Comparison : All Models (Top 10 Countries by Population)', fontsize=13)
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (MtCO2)')
plt.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

#model comparison table
print("FINAL MODEL COMPARISON (Test Set: 2018-2021)")

results_df = pd.DataFrame(model_results).T
results_df.columns = ['Train MAE', 'Train R²', 'Test MAE', 'Test R²']
results_df = results_df.sort_values('Test MAE')

print(f"\n{'Model':<30} {'Test MAE (MtCO2)':>18} {'Test R²':>10} {'vs Best':>10}")

best_mae = results_df['Test MAE'].min()
for name, row in results_df.iterrows():
    diff = ((row['Test MAE'] - best_mae) / best_mae * 100)
    marker = " ★" if row['Test MAE'] == best_mae else ""
    print(f"{name:<30} {row['Test MAE']:>14,.2f}     {row['Test R²']:>8.4f}  {diff:>+8.1f}%{marker}")

print(f"\nBest Model: {results_df.index[0]} : Test MAE = {best_mae:,.2f} MtCO2")
