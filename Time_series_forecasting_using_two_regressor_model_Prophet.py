# librairies utilisées 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from fbprophet import Prophet
plt.style.use("fivethirtyeight")
import warnings 
warnings.filterwarnings("ignore")
# Chargement des données 
data_A = pd.read_csv("C:\\Users\\mamahzoune\\Desktop\\CA_FA.csv")
# Visualisation des données 
print(data_A)
# voir la corrélation entre les differentes variables 
data_A.corr()
# renommer les colonnes Mois et Chiffre d'affaire en ds et y
data_A = data_A.rename(columns ={"Mois":"ds","Chiffre d'affaire":"y"})
# Trier la base de données selon la date 
data_A = data_A.sort_values(by=["ds"])
# Changement du type de la colonne ds en datetime 
data_A.ds = pd.to_datetime(data_A.ds)
# Modification de la colonne date 
data_A.ds = pd.date_range('2016-01',periods = 50, freq ='M')
# Création des futures dates 
future_range = pd.date_range('2020-03-01',periods =2, freq ='M')
# Création d'un dataframe contenant les valeurs de l'inflation et le Taux_de_Chomage pour les futures dates
future_data = pd.DataFrame({'future_date':future_range,'future_inflation':0,'future_Taux_de_chomage':0})
# Changement du type de la colonne future_date en datetime 
future_data['future_date'] = pd.to_datetime(future_data['future_date'])
# rendre la colonne future_data en index 
future = future_data.set_index('future_date', inplace = True)
# remplir les deux colonnes Inflation et Taux de Chomage pour les futures dates
future_data.at['2020-03-31','future_inflation'] = 0.6737896
future_data.at['2020-04-30','future_inflation'] = 0.326233
future_data.at['2020-03-31','future_Taux_de_chomage'] = 7.6
future_data.at['2020-04-30','future_Taux_de_chomage'] = 8.7
# parametrage du model Prophet
m = Prophet(daily_seasonality=True,weekly_seasonality=True,interval_width=0.95)
m.add_regressor("Inflation")
m.add_regressor("Taux_de_Chomage ')
# entrainement du model Prophet
m.fit(data_A)
# rendre la colonne de notre data set en index 
index = data_A.set_index("ds",inplace = True)
# création d'une fonction qui permet de remplir les deux colonnes inflation et Taux de chomage  pour les futures dates 
def Inflation(ds):
    date = (pd.to_datetime(ds)).date()
    if data_A[date:].empty:
       return future_data[date:]['future_inflation'].values[0]
    else:
       return(data_A[date:]['Inflation']).values[0]
    return 0

def Taux_de_chomage(ds):
    date = (pd.to_datetime(ds)).date()
    if data_A[date:].empty:
         return future_data[date:]["future_Taux_de_chomage"].values[0]
    else:
         return(data_A[date:]['Taux_de_Chomage ']).values[0]
    return 0
    
future = m.make_future_dataframe(periods=2, freq="M")
future["Inflation"] = future["ds"].apply(Inflation)
future["Taux_de_Chomage '] = future['ds'].apply(Taux_de_chomage)
# Prediction du chiffre d'affaire pour les mois Mars et Avril en utilisant deux regressors
forecast = m.predict(future)
B = forecast[['ds','yhat','yhat_lower','yhat_upper']]
# rendre la colonne du dataframe B en index 
B = B.set_index('ds')
# Concaténation des valeurs réelles et les valeurs prédites 
B = pd.concat([B, data_A], axis=1)
# Visualisation du DataFrame B
Print(B)
# Renommer les différents colonnes du DataFrame B
B.columns = ["Predictions","Borne Supérieure de Prédiction","Borne Inférieure de Prédiction","Réalité","Inflation","Taux_de_Chomage"]
# Visualisation les résulats de la prédiction et les valeurs réelles 
plt.figure(figsize=(10,10))
plt.plot(B["Réalité"], color = 'black', label='Réalité')
plt.plot(B['Predictions'], color ='red',label='Prédiction')
plt.plot(B['Borne Supérieure de Prédiction'], color ='blue', label='La borne supérieure de prévision')
plt.plot(B["Borne Inférieure de Prédiction'], color = 'green',label='La borne Inférieure de la prévision')
plt.xlabel('Date')
plt.ylabel('Chiffre d'affaire')
plt.title("Les prédictions du chiffre d'affaire")
plt.show()
# Suppression des valeurs manquantes pour évaluer le modèle 
B.dropna(inplace = True)
# les métriques utilisées pour évaluer notre modèle
# R_squared 
r2_score(B.Réalité,B.Predictions)  # 0.975420
# Mean_absolute_error
mean_absolute_error(B.Réalité,B.Predictions) # 343176.636255

