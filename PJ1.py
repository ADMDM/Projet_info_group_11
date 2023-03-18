#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:11:04 2023

@author: adamdavidmalila
"""

# Importation des packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des fichiers
weather = pd.read_csv("weather.txt")
soil = pd.read_csv("Soil_measurements.txt")
sfd = pd.read_csv("SFD.txt")


# group weather data by year, day, and DOY, and aggregate variables
weather["wd"]=weather.groupby(["Rain"]).cumsum()



#True_New_Code :



#  PROJET INFO CODE    

# Certain '#' sont utiliser pour rendre plus claire la console


#importation des packages :

import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sb 


#Charger les donées respectivement; weather, soil_measurments, SFD

#1. weather
weather = pd.read_csv("weather.txt")
#print(weather.head())
#weather.info()

#2. soil_measurments
soil_measurments = pd.read_csv("Soil_measurements.txt")
#print (soil_measurments.head())
#soil_measurments.info()

#3. SFD 
SFD = pd.read_csv("SFD.txt")
#print (SFD.head())
#SFD.info()



#    DEBUTS DES TACHES 

# créeation du dataframe : data
#weather["Cum_Rain"] = weather["Rain"].cumsum()
#weather["Cum_Irr"] = weather["Irradiance"].cumsum()
#data_weather = weather.groupby(by='Day') [['Cum_Irr','Cum_Rain']].sum()


data_weather = weather.groupby(by='Day') [['Irradiance','Rain']].sum()
#print(data_weather.head()) 

data_weather1 = weather.groupby(by='Day') [['Temp','Wind','Rel_hum','SD']].mean()
#print(data_weather1.head())

data_soil = soil_measurments.groupby(by = 'Day') [['H45T', 'H45E']].sum()
#print(data_soil.head())

data_soil1 = soil_measurments.groupby(by = 'Day') [['Rain_E', 'Rain_T']].sum()
#print(data_soil1.head())

data_SFD = SFD.groupby(by = 'Day') [['E153', 'E159', 'E161', 'T13', 'T21', 'T22']].sum()
#print(data_SFD)

data = pd.merge(pd.merge(pd.merge(pd.merge(data_weather,data_weather1, how='outer', on= ('Day') ),data_soil,how='outer', on= ('Day')), data_soil1,how='outer', on= ('Day')), data_SFD,how='outer', on= ('Day'))




#Changement de l'unité de la variable Irradiance
# data ['Cum_Irr'] = data['Cum_Irr'] / 360
data ['Irradiance'] = data['Irradiance'] / 360
#print(data.head())


#changement d'index de data ; nouvel index : Date
data['Date'] = pd.date_range('1999-1-1',periods = 1096).strftime('%Y-%m-%d')
data.set_index('Date', inplace = True)

#save data 
data.to_csv('data_LBIR1271.csv')
   
