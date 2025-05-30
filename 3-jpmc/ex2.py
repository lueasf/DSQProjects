from datetime import date
import math

## Pricing Models : Pricing Commodity storage contract (Natural Gas Contract)
# Exercice 2 : créer un prototype de **modèle de tarification** (pour évaluer un contrat.)
# Écrire une fonction capable d'utiliser les données générées pour évaluer le contrat.

"""
_injection date_ : moment où la matière première est achetée.
_withdrawal date_ : moment où elle est retirée du stockage et vendue.

Principe de valorisation: La valeur d'un accord commercial correspond au prix de revente 
moins le prix d'achat, moins les coûts associés.

Ex : achat d'1 million de BTU de gaz à 2$/BTU en été, vendue 3$/BTU en hiver.
Valeur = (3 - 2)*1 000 000 = 1 000 000$.
Coûts à déduire :
- stockage : 100k / moiz x 4 mois = 400k
- retrait : 10k x 2 (entrée et sortie) = 20k
- transport : 50k x 2 (aller-retour) = 100k
--> valeur finale = 480k

L'objectif est d'obtenir une estimation équitable satisfaisant à 
la fois le desk trading et le client.
"""

def price_contract(in_dates, in_prices, out_dates, out_prices, rate, storage_cost_rate, total_vol, injection_withdrawal_cost_rate):
    volume = 0 # actuel
    buy_cost = 0
    cash_in = 0 
    
    all_dates = sorted(set(in_dates + out_dates)) # tri des dates dans l'ordre chronologique
    
    for i in range(len(all_dates)): 
        start_date = all_dates[i]

        if start_date in in_dates: # logique d'achat
            if volume <= total_vol - rate:
                volume += rate
                buy_cost += rate * in_prices[in_dates.index(start_date)]
                injection_cost = rate * injection_withdrawal_cost_rate
                buy_cost += injection_cost
            else:
                print('Impossible, full storage capacity reached')
            
        elif start_date in out_dates: # logique de vente
            if volume >= rate:
                volume -= rate
                cash_in += rate * out_prices[out_dates.index(start_date)]
                withdrawal_cost = rate * injection_withdrawal_cost_rate
                cash_in -= withdrawal_cost
            else: 
                print('Impossible, not enough gas in storage')
                
    store_cost = math.ceil((max(out_dates) - min(in_dates)).days // 30) * storage_cost_rate # coût de stockage total
    return cash_in - store_cost - buy_cost

# Example usage of price_contract()
in_dates = [date(2022, 1, 1), date(2022, 2, 1), date(2022, 2, 21), date(2022, 4, 1)] #injection dates
in_prices = [20, 21, 20.5, 22] #prices on the injection days
out_dates = [date(2022, 1, 27), date(2022, 2, 15), date(2022, 3, 20), date(2022, 6, 1)] # extraction dates
out_prices = [23, 19, 21, 25] # prices on the extraction days
rate = 100000  # rate of gas in cubic feet per day
storage_cost_rate = 10000  # total volume in cubic feet
injection_withdrawal_cost_rate = 0.0005  # coût par pied cube
max_storage_volume = 500000 # maximum storage capacity of the storage facility
result = price_contract(in_dates, in_prices, out_dates, out_prices, rate, storage_cost_rate, max_storage_volume, injection_withdrawal_cost_rate)
print(result)