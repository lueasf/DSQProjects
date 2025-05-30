from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

## Credit risk Analysis : Calculate Loan Borrowers Probability of Default (PD)

# construire un modèle prédictif capable d'estimer la probabilité de défaut en fonction des caractéristiques des clients.
# PD : Probability of Default
# On suppose que le taux de recouvrement est de 10%, cad, la banque ne récupère que 10% de la somme due.

"""
Les données du CSV :
- customer_id 
- credit_lines_outstanding : nombre de lignes de crédit en cours (prêt en cours)
- loan_amt_outstanding : montant total du prêt en cours
- total_debt_outstanding 
- income 
- years_employed 
- fico_score : FICO Score est un score de crédit aux US, pour évaluer la solvabilité d'un emprunteur. 
    + il est élevé, moins c'est risqué. En dessous de 600, c'est risqué.
    Il varie entre 300 et 850.
- default
"""

df = pd.read_csv('Loan_Data.csv')
features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

# Montant du pret en cours rapporté au revenu, et de la dette totale au revenu
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

# la regression logistique permet de prédire une probabilité entre 0 et 1, avec une tolérance de 1e-5
clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000).fit(df[features], df['default'])
print(clf.coef_, clf.intercept_)
 
y_pred = clf.predict(df[features])

# Courbe ROC (Receiver Operating Characteristic)
# La courbe ROC est une représentation graphique de la performance d'un modèle de classification binaire.
# Elle trace le taux de vrais positifs (TPR) contre le taux de faux positifs (FPR) à différents seuils de classification.
fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print((1.0*(abs(df['default']-y_pred)).sum()) / len(df)) # taux d'erreur
print(metrics.auc(fpr, tpr)) # AUC : Area Under the Curve = mesure de la performance du modèle (1 = parfait, 0.5 = aléatoire)

# Resultat ici :
# taux d'erreur = 0.0037
# AUC = 0.9925