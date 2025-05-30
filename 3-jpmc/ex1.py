import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX 

## Market forecasting : forecasting Natural Gas Market prices
# Exercice 1 : Extrapolation et prévision des prix du gaz naturel

"""
SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) est un modèle statistique
pour les séries temporelles.

-> Utilise des valeurs passées pour prédire les valeurs futures.
order = (p,d,q) : 
p = 1, signifie que le prix à t dépend du prix à t-1
d = 1, signifie que la série est rendue stationnaire
q = 1, signifie qu'on utilise l'erreur du cas précédent

seasonal_order (P,D,Q,s) : 
P = 1, signifie que le prix actuel dépend des valeurs passées au même mois de l'année précédente
D = 1, rend la série stationnaire
Q = 1, corrige les erreurs passées au même point du cycle
s = 12, signifie que le motif se repète tous les 12 mois

initialization = 'approximate_diffuse' : traite les valeurs initailes comme non importantes,
car les premièeres valeurs d'une série temporelle sont souvent peu fiables.

Ce modèle est choisie car le gaz naturel a des cycles annuels et mensuels.
"""

data = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'], dayfirst=True)
data['Prices'] = data['Prices'].astype(float)
data = data.set_index('Dates')

train = data['Prices']
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), initialization='approximate_diffuse')  
# Ce modèle capture une tendance mensuelle via order et un cycle mensuel via seasonal_order

forecast_steps = 12 # mois
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='ME')
# date_range génère une série de dates à partir de la dernière date connue, avec un pas de 1 mois

results = model.fit(disp=False) 
forecast = results.get_forecast(steps=forecast_steps, index=forecast_index)
forecast_df = forecast.predicted_mean.to_frame('Prices')

def estimate_price(input_date):
    target_date = pd.to_datetime(input_date)
        
    if target_date <= last_date:
        return round(data.loc[data.index <= target_date].iloc[-1].Prices, 2) # renvoie le dernier prix connu
    return round(forecast_df.loc[forecast_df.index >= target_date].iloc[0].Prices, 2) # renvoie le prix prédit

plt.plot(data, label='Données réelles', marker='o')
plt.plot(forecast_df, label='Prévision', marker='x', linestyle='--')
plt.title('Prévision des prix du gaz naturel')
plt.legend()
plt.grid(True)
plt.show()