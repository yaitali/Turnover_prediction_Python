# libraries utilisées 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.preprocessing  import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.impute import MissingIndicator 
from pandas import datetime
# Chargement des données 
data = pd.read_csv("C:\\Users\\redaa\\Desktop\\prédiction du chiffre d'affaire\\Prediction_Chiffre_d'affaire.csv",parse_dates=True)
# Visualisation de données 
data
# visualisation les noms des colonnes de la base de données
data.columns 
# supression des colonnes selectionnées 
data = data.drop(["Unnamed: 0","Date2","LOCATION.x","INDICATOR.x","SUBJECT.x","MEASURE.x","FREQUENCY.x","LOCATION.y","INDICATOR.y","SUBJECT.y","MEASURE.y","FREQUENCY.y"],axis=1)
# visualisation des données après suppression des colonnes selectionnées 
data
# nomination des colonnes 
data.columns =["Nombre_de_commandes","CA","Jours_Ouvrables","Source_Vente","Nombre_customers_KA","Nombre_customers_BA","DATE","PIB","Taux_de_Chomage"]
# changement du type de la colonne DATE
data.DATE = pd.to_datetime(data.DATE)
data.shape
data.head()
# visualisation des differentes mesures statistiques pour chaque colonne(moyenne , variance , minimum,maximum)
data.describe()
# remplacement des différentes valeurs manquantes pour les données PIB et Taux_de_Chomage par la moyenne
data.fillna(data["PIB"].mean())
data.fillna(data["Taux_de_Chomage"].mean())
# y contient la colonne du chiffre d'affaires
y = data.iloc[:,1].values 
# visualisation des sources de ventes sous forme d'un Camembert pour voir la source de vente la plus utilisée
data["Source_Vente"].value_counts(normalize=True).plot(kind='pie')
plt.show()
# Visualisation des nombres de jours ouvrables contribuées dans l'ensemble des commandes 
data["Jours_Ouvrables"].value_counts().plot.bar()
plt.show()
data["Jours_Ouvrables"].value_counts()
# visualisation des sources de ventes sous forme des batons  pour voir la source de vente la plus utilisée
data["Source_Vente"].value_counts(normalize=True).plot(kind='bar')
plt.show()
# visualisation des differents diagrammes qui montrent la distribution du chiffre d'affaire pour chaque source de vente 
for A in data["Source_Vente"].unique():
    sous_echantillon = data[data.Source_Vente == A] # création du sous_échantillo,
    print("-"*20)
    print(A)
    print("moy:\n",sous_echantillon["CA"].mean())
    print("med:\n",sous_echantillon["CA"].median())
    sous_echantillon["CA"].hist() # crée l'histogramme
    plt.show()
    sous_echantillon.boxplot(column ="CA",vert=False)
    plt.show()
# Encodage de la variable Source_Vente(variable qualitative) 
categories = list(data["Source_Vente"].unique())
map_dict = {'CASE':0,'Chat':1,'Customer Visit':2,'EPROCM':3,'Email':4,'Fax':5,'INTERCO':6,'Other':7,'PORTAL':8,'Phone':9,'Post':10,'Webshop':11,'e-Procurement':12}
print(categories)
F = data.Source_Vente.map(map_dict)
data = pd.concat([data.drop("Source_Vente",axis=1),F],axis='columns')
# visualisation des données après l'encodage 
data
 # analyse des différents colonnes en moyenne en les regroupant  par Jours Ouvrables
data.groupby(["Jours_Ouvrables"]).mean() 
# X contient les differents données 
X = data.iloc[:,:].values 
# regroupement des deux colonnes Taux_de_Chomage et Source_Vente 
column = ["Taux_de_Chomage","Source_Vente"]
data[column]
# remplacement des valeurs manquantes NaN par la moyenne pour la colonne Source_Vente
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis=0)
imputer = imputer.fit(data[column])
data[column] = imputer.transform(data[column])
data
# Encodage la variable Chiffre d'affaire(CA) 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
# diviser les données en deux parties : données d'entrainement et données de tests.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
# regroupement des données sans la Date pour faire de la standarisation 
X1=data.loc[:,["Nombre_de_commandes","CA","Jours_Ouvrables","Nombre_customers_KA","Nombre_customers_BA","PIB","Taux_de_Chomage","Source_Vente"]]
# standardisation des données 
sc_X1= StandardScaler()
X1_train = sc_X1.fit_transform(X1_train)
X1_test = sc_X1.transform(X1_test)
# indexing with Date column
data.set_index("DATE",inplace=True)
# Visualisation du PIB en fonction de la DATE
data['PIB'].plot(figsize=(9,6))
plt.show()
# Visualtion du Taux_de_Chomage
data['Taux_de_Chomage'].plot(figsize=(9,6))
plt.show()
# visualisation du chiffre d'affaire 2020 en fonction de la DATE
data.loc["2020","CA"].plot()
plt.xlabel("DATE")
plt.ylabel('Chiffra_affaire')
plt.show()
# visualisation des differents mesures statistiques ( moyenne,variance,minimum,maximum) pour le chiffre d'affaire par semaine 
data.loc['2020','CA'].resample('W').agg(['mean','std','min','max'])
# visualisation de la moyenne du chiffre d'affaire par semaine 
m = data.loc['2020','CA'].resample('W').agg(['mean','std','min','max'])
plt.figure(figsize=(12,8))
m['mean']['2020'].plot(label='moyenne par semaine')
plt.legend()
plt.show()
# visualisation la correlation entre les variables 
sns.heatmap(data.corr())
