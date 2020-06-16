# librairies utilisées 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from fbprophet import Prophet 
from sklearn.metrics import r2_score, mean_absolute_error
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Chargement des données 
data_CA = pd.read_csv("C:\\Users\\mamahzoune\\Données utilisées\\CA_FR.csv")
# Visualisation des données 
data_CA
# Visualisation des types de variables 
data_CA.dtypes 
# Changement du type de la variable month en datetime 
data_CA["month"] = pd.to_datetime(data_CA["month"])
# renommer les deux variables month et ca en ds et y 
data_CA = data_CA.rename(columns = {"month":"ds","ca":"y"})
# Visualisation des données après les differents changement réalisés
data_CA
# paramétrage du modèle utilisé
my_model = Prophet(daily_seasonality=True,weekly_seasonality=True,interval_width=0.95)
# entrainement du modèle sur notre dataset 
my_model.fit(data_CA)
# Création les futures mois à prédire 
future_dates = my_model.make_future_dataframe(periods=2, freq='MS')
future_dates.tail(4)
# Prediction de la variable cible 
forecast = my_model.predict(future_dates)
A = forecast[['ds','yhat','yhat_lower','yhat_upper']]
# Trier les données selon la colonne Month (date)
data_CA = data_CA.sort_values(by =['ds'])
# rendre la colonne Month en index 
A = A.set_index('ds')
# Concaténation des valeurs réelles et les valeurs prédites 
B = pd.concat([A, data_CA]), axis = 1)
# Visualisation du tableau contenant les valeurs réelles et les valeurs prédites 
B 
# renommer les différentes colonnes du tableau B
B.columns = ["Predictions","Borne supérieure de Prédiction","Borne Inférieure de Prédiction","Réalité"]
# Visualisation à l'aide d'un graphique montrant les valeurs prédites et les valeurs réelles ainsi que la borne supérieure et la borne inférieure de la prédiction
plt.figure(figsize=(10,10))
plt.plot(B['Réalité'], color = 'black', label='Réalité')
plt.plot(B['Predictions'], color ='red', label='Prédiction')
plt.plot(B['Borne Supérieure de Prédiction'], color ='blue', label ='La borne supérieure de la prévision')
plt.plot(B['Borne Inférieure de Prédiction'], color ='green',label = 'La borne inférieure de la prévision')
plt.xlabel('Date')
plt.ylabel("Chiffre d'affaire")
plt.title("les prédictions du chiffre d'affaire")
plt.show()
# Visualtion de la saisonnalité et la tendance des futures prédictions
my_model.plot_components(forecast)
# Supprimer les valeurs manquantes pour faire de l'évaluation de notre model 
B.dropna(inplace=True)
# les differents metriques utilisées pour évaluer notre modèle
r2_score(B.Réalité,B.Predictions)
mean_absolute_error(B.Réalité,B.Predictions)
