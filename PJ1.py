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



#Partie 2 du projet :

# Calculer les statistiques principales des variables

stats = pd.DataFrame({"Moyenne": data.mean(),"Variance": data.var(),"Minimum": data.min(),"Maximum": data.max()})


# Figure 1

fig, axs = plt.subplots(4, 1, figsize=(10, 10))

axs[0].plot(data.index, data['Rain_E'], title = 'Pluie au sol dans la for')

axs[0].set_ylabel('Rain_E (mm)')

axs[1].plot(data.index, data['Rain_T'], title = 'Rain_T')

axs[1].set_ylabel('Rain_T (mm)')

axs[2].plot(data.index, data['H45E'], title = 'H45E')

axs[2].set_ylabel('H45E (g/m³)')

axs[3].plot(data.index, data['H45T'], title = 'H45T')

axs[3].set_ylabel('H45T (g/m³)')

fig.suptitle('Figure 1: Rain_E, Rain_T, H45E, and H45T')

# Figure 2
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].plot(data.index, data['Irradiance'])
axs[0, 0].set_ylabel('Irradiance (kWh/m²)')
axs[0, 1].plot(data.index, data['Rain'])
axs[0, 1].set_ylabel('Rain (mm)')
axs[0, 2].plot(data.index, data['Temp'])
axs[0, 2].set_ylabel('Temp (°C)')
axs[1, 0].plot(data.index, data['Wind'])
axs[1, 0].set_ylabel('Wind (m/s)')
axs[1, 1].plot(data.index, data['Rel hum'])
axs[1, 1].set_ylabel('Rel_hum (%)')
axs[1, 2].plot(data.index, data['SD'])
axs[1, 2].set_ylabel('SD (kPa)')
fig.suptitle('Figure 2: Irradiance, Rain, Temp, Wind, Rel_hum, and SD')

 

# Figure 3
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].plot(data.index, data['E153'])
axs[0, 0].set_ylabel('E153 (kg/m²/s)')
axs[0, 1].plot(data.index, data['E159'])
axs[0, 1].set_ylabel('E159 (kg/m²/s)')
axs[0, 2].plot(data.index, data['E161'])
axs[0, 2].set_ylabel('E161 (kg/m²/s)')
axs[1, 0].plot(data.index, data['T13'])
axs[1, 0].set_ylabel('T13 (mm/d)')
axs[1, 1].plot(data.index, data['T21'])
axs[1, 1].set_ylabel('T21 (mm/d)')
axs[1, 2].plot(data.index, data['T22'])
axs[1, 2].set_ylabel('T22 (mm/d)')
fig.suptitle('Figure 3: E153, E159, E161, T13, T21, and T22')

 

plt.show()


#Evolution la pluie au sol et de l’humidit´e volumique du sol entre 1999 et
#2001 en forˆet ´eclaircie et t´emoin
