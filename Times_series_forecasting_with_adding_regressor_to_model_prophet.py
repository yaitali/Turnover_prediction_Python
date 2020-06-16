# Librairies utilisées
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from fbprophet import Prophet 
from sklearn.metrics import r2_score, mean_absolute_error
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
# Chargement des données 
data = pd.read_csv("C:\\Users\\mamahzoune\\Desktop\\Données utilisées\\CA.csv",sep=',')
# Visualisation des données 
print(data)
# la corrélation entre les differents variables 
data.corr()
# Renommer les colonnes Mois et Chiffre d'affaire en ds et y
data = data.rename(columns ={"Mois":'ds',"Chiffre d'affaire":"y"})
# Trier les données selon la colonne date
data = data.sort_values(by=['ds'])
# Changement du type de la colonne ds en datetime
data.ds = pd.to_datetime(data.ds)
# Modification de la colonne date
data.ds = pd.date_range("2016-01", periods=50, freq='M')
# création des futures dates 
future_range = pd.date_range('2020-03-01', periods=2, freq='M')
# Création d'un dataframe contenant les valeurs de l'inflation pour les futures dates
future_inflation_data = pd.DataFrame({'future_date':future_range,'future_inflation':0})
# Changement du type de la colonne future_date en datetime
future_inflation_data['future_data'] = pd.to_datetime(future_inflation_data['future_data'])
# rendre la colonne future_data en index
future_inflation = future_inflation_data.set_index('future_data', inplace=True)
# remplir la colonne inflation pour les futures dates 
future_inflation_data.at['2020-03-31','future_inflation'] = 0.6737896
future_inflation_data.at['2020-04-30','future_inflation'] = 0.326233
# parametrage du model Prophet
m = Prophet(daily_seasonality=True,weekly_seasonality=True,interval_width=0.95)
m.add_regressor('Inflation')
# entrainement du model sur le jeu de données 
m.fit(data)
# rendre la colonne de notre dataset en index
index = data.set_index("ds", inplace = True)
# création d'une fonction qui permet de remplir la colonne inflation avec les futures données 
def Inflation(ds):
    date = (pd.to_datetime(ds)).date()
    if data[date:].empty:
       return future_inflation_data[date:]['future_inflation'].values[0]
    else:
       return (data[date:]['Inflation']).values[0]
    return 0
 future = m.make_future_dataframe(periods=2, freq='M')
 future['Inflation] = future["ds"].apply(Inflation)
 # Visualisation de la colonne Inflation avec les futures données 
 print(future)
 # Prediction du chiffre d'affaire 
 forecast = m.predict(future)
 B = forecast[['ds','yhat','yhat_lower','yhat_upper']]
 # Trier le jeu de données par date 
 data = data.sort_values(by=["ds"])
 # rendre la colonne ds du dataframe B en index 
 B = B.set_index('ds')
 # Concaténation des valeurs réelles avec les valeurs prédites 
 B = pd.concat([B, data], axis = 1)
 # visualisation du dataframe B
 B
 # Renommer les differents colonnes du dataframe B
 B.columns = ["Predictions","Borne Supérieure de Prédiction","Borne Inférieure de Prédiction","Réalité","Inflation"]
 # Visualtion des résultats de la prédiction et les valeurs réelles 
 plt.figure(figsize=(10,10))
 plt.plot(B["Réalité"], color = 'black', label='Réalité')
 plt.plot(B['Predictions'], color ='red',label='Prédiction')
 plt.plot(B['Borne Supérieure de Prédiction'], color ='blue', label='La borne supérieure de prévision')
 plt.plot(B["Borne Inférieure de Prédiction'], color = 'green',label='La borne Inférieure de la prévision')
 plt.xlabel('Date')
 plt.ylabel('Chiffre d'affaire')
 plt.title("Les prédictions du chiffre d'affaire")
 plt.show()
 # suppresion des valeurs manquantes pour évaluer notre modèle 
 B.dropna(inplace = True)
 # les métriques utilisées pour évaluer notre modèle 
 # R_squared 
 r2_score(B.Réalité,B.Predictions)  # 0.969429
 # Mean_absolute_error
 mean_absolute_error(B.Réalité,B.Predictions) # 380574.6521
  
