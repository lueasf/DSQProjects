import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'], index_col='Dates') # index_col='Dates' pour def Dates comme index de la série temporelle
data['Prices'] = data['Prices'].astype(float)

train = data['Prices']
model = ExponentialSmoothing(data['Prices'], trend='add', seasonal='add', seasonal_periods=12)
# ce modèle capture une tendance linéaire additive et un cycle saisonnier additif
fitted_model = model.fit() 

forecast_steps = 12 # mois
forecast_index = pd.date_range(data.index[-1], periods=forecast_steps+1, freq='ME')[1:] # on commence à partir du mois suivant
# date_range génère une série de dates à partir de la dernière date connue, avec un pas de 1 mois
forecast = fitted_model.forecast(steps=forecast_steps) # forecast est une variable de type Series qui contient les prévisions sur les 12 mois suivants

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Prices'], label='Historical Prices')
plt.plot(forecast_index, forecast, label='Forecasted Prices', linestyle='--')
plt.title('Natural Gas Prices: Historical and Forecast')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()
plt.show()