#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:11:04 2023

@author: adamdavidmalila
"""
  
# Projet groupe 11 :Script partie Informatique
# Importation des packages nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importation des données météorologiques (Weather.txt), du sol (Soil_measurements.txt) et de la densité du flux de sève (SFD.txt) en Dataframe Pandas

# Importation de la donnée Weather.txt
weather = pd.read_csv("Weather.txt")

# Importation de la donnée Soil_measurements.txt
soil = pd.read_csv("Soil_measurements.txt")

# Importation de la donnée SFD.txt
sfd = pd.read_csv("SFD.txt")


# Traitement des données

# Création des groupes de données en utilisant la méthode "groupby"

# Groupe de données : Irradiance et précipitations journalières
daily_weather = weather.groupby(by='Day')[['Irradiance', 'Rain']].sum()

# Groupe de données : Température, Vitesse du vent, Humidité relative et Déficit de saturation journaliers
daily_weather1 = weather.groupby(
    by='Day')[['Temp', 'Wind', 'Rel_hum', 'SD']].mean()

# Groupe de données : H45T et H45E journaliers
daily_soil = soil.groupby(by='Day')[['H45T', 'H45E']].sum()

# Groupe de données : Rain_E et Rain_T journaliers
daily_soil1 = soil.groupby(by='Day')[['Rain_E', 'Rain_T']].sum()

# Groupe de données : E153, E159, E161, T13, T21 et T22 journaliers
daily_sfd = sfd.groupby(
    by='Day')[['E153', 'E159', 'E161', 'T13', 'T21', 'T22']].sum()

# Fusion des groupes de données en un seul dataframe
data = pd.merge(pd.merge(pd.merge(pd.merge(daily_weather, daily_weather1, how='outer', on='Day'), daily_soil,
                how='outer', on='Day'), daily_soil1, how='outer', on='Day'), daily_sfd, how='outer', on='Day')


# Conversion de l'irradiance en kWh/m^2
data['Irradiance'] = data['Irradiance'] / 360


# Changement d'index de data : nouvel index = Date
data['Date'] = pd.date_range('1999-1-1', periods=1096).strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)


# Enregistrement du dataframe data dans un fichier CSV
data.to_csv('data_LBIR1271.csv')


# Calcul des statistiques principales des variables

# Création d'un dataframe contenant les statistiques principales de chaque variable
stats = pd.DataFrame({"Moyenne": data.mean(), "Variance": data.var(
), "Minimum": data.min(), "Maximum": data.max()})

# Affichage du dataframe stats
print(stats)


# Chargement du dataframe data depuis le fichier CSV
data = pd.read_csv('data_LBIR1271.csv', parse_dates=['Date'], index_col='Date')

# Convertir les valeurs de H45E et H45T
data['H45E'] = data['H45E']
data['H45T'] = data['H45T']

# Figure 1: Rain E, Rain T, H45E et H45T

# Création d'une figure contenant 4 sous-graphiques
fig1, axs1 = plt.subplots(4, 1, figsize=(10, 10))

# Plot de Rain_E (mm)
axs1[0].plot(data.index, data['Rain_E'])
axs1[0].set_title('Pluie au sol dans la forêt éclaircie [mm]')
plt.grid()

# Plot de Rain_T (mm)
axs1[1].plot(data.index, data['Rain_T'])
axs1[1].set_title('Pluie au sol dans la forêt témoin [mm]')
plt.grid()
# Plot de H45E (%)
axs1[2].plot(data.index, data['H45E'])
axs1[2].set_title('Humidité volumique dans le sol dans la forêt éclaircie [%]')
plt.grid()

# Plot de H45T (%)
axs1[3].plot(data.index, data['H45T'])
axs1[3].set_title('Humidité volumique dans la forêt témoin [%]')
plt.grid()

fig1.suptitle("Figure 1: Evolution la pluie au sol et de l'humidité volumique du sol entre 1999 et 2001 en foret eclaircie")
fig1.tight_layout(pad=3.0, h_pad=1.5)
plt.grid()
plt.show()

# Enregistrer les figure sous format pdf
fig1.savefig("figure1.png")

# Figure 2
fig2, axs2 = plt.subplots(6, 1, figsize=(10, 15))

# Plot de Irradiance (kWh/m²)
axs2[0].plot(data.index, data['Irradiance'])
axs2[0].set_title('Irradiance [kWh/m²]')
plt.grid()

# Plot de Rain (mm)
axs2[1].plot(data.index, data['Rain'])
axs2[1].set_title('Rain (mm)')
plt.grid()

# Plot de Temp (°C)
axs2[2].plot(data.index, data['Temp'])
axs2[2].set_title('Température [°C]')
plt.grid()

# Plot de Wind (m/s)
axs2[3].plot(data.index, data['Wind'])
axs2[3].set_title('Vitesse du vent [m/s]')
plt.grid()

# Plot de Rel_hum (%)
axs2[4].plot(data.index, data['Rel_hum'])
axs2[4].set_title('Humidité relative [%]')
plt.grid()

# Plot de SD (m)
axs2[5].plot(data.index, data['SD'])
axs2[5].set_title('Déficit de saturation [hPa]')
plt.grid()

fig2.suptitle(
    "Figure 2: Evolution de variables meterologique entre 1999 et 2001 ")
fig2.tight_layout(pad=3.0, h_pad=1.5)
plt.grid()
plt.show()
# Enregistrer les figure sous format pdf
fig2.savefig("figure 2.png")

# Figure 3
fig3, axs3 = plt.subplots(6, 1, figsize=(10, 15))

# Ajout du premier sous-graphique : E153 (g/s)
axs3[0].plot(data.index, data['E153'])
axs3[0].set_title('Arbre 153 (en forêt éclaircie)')
plt.grid()

# Ajout du deuxième sous-graphique : E159 (g/s)
axs3[1].plot(data.index, data['E159'])
axs3[1].set_title('Arbre 159 (en forêt éclairci)')
plt.grid()

# Ajout du troisième sous-graphique : E161 (g/s)
axs3[2].plot(data.index, data['E161'])
axs3[2].set_title('Arbre 161 (en forêt témoin)')
plt.grid()

# Ajout du quatrième sous-graphique : T13 (°C)
axs3[3].plot(data.index, data['T13'])
axs3[3].set_title('Arbre 13 en fore^t témoin)')
plt.grid()

# Ajout du cinquième sous-graphique : T21 (°C)
axs3[4].plot(data.index, data['T21'])
axs3[4].set_title('Arbre 21 (en forêt témoin)')
plt.grid()

# Ajout du sixième sous-graphique : T22 (°C)
axs3[5].plot(data.index, data['T22'])
axs3[5].set_title('Arbre 22 (en fore^t témoin)')
plt.grid()

# Ajout d'un titre à la figure
fig3.suptitle('Figure 3: Densité de flux de sève[L.h^-1.dm^-2]')
fig3.tight_layout(pad=3.0, h_pad=1.5)
plt.grid()
plt.show()

# Enregistrer les figure sous format pdf
fig3.savefig("figure3.png")


# Calculer la matrice de corrélation
corr_matrix = data.corr()

# Afficher la heatmap
sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de corrélation des variables météorologiques')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# Sélection des variables à inclure dans la figure
variables = ["Temp", "Rain", "Wind", "Irradiance",
             "Rel_hum", "SD", "Rain_E", "H45E", "E153"]

# Création d'un sous-dataframe contenant seulement ces variables
data_sub = data[variables]

# Création de la figure avec la fonction pairplot
sns.pairplot(data_sub, diag_kind="hist", corner=True)


# Sauvegarde de la figure
plt.savefig("pairplot.png")
