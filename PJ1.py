#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:11:04 2023

@author: adamdavidmalila
"""
    #Projet groupe 11 :Script partie Informatique
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
daily_weather1 = weather.groupby(by='Day')[['Temp', 'Wind', 'Rel_hum', 'SD']].mean()

# Groupe de données : H45T et H45E journaliers
daily_soil = soil.groupby(by='Day')[['H45T', 'H45E']].sum()

# Groupe de données : Rain_E et Rain_T journaliers
daily_soil1 = soil.groupby(by='Day')[['Rain_E', 'Rain_T']].sum()

# Groupe de données : E153, E159, E161, T13, T21 et T22 journaliers
daily_sfd = sfd.groupby(by='Day')[['E153', 'E159', 'E161', 'T13', 'T21', 'T22']].sum()

# Fusion des groupes de données en un seul dataframe
data = pd.merge(pd.merge(pd.merge(pd.merge(daily_weather, daily_weather1, how='outer', on='Day'), daily_soil, how='outer', on='Day'), daily_soil1, how='outer', on='Day'), daily_sfd, how='outer', on='Day')


# Conversion de l'irradiance en kWh/m^2
data['Irradiance'] = data['Irradiance'] / 360


# Changement d'index de data : nouvel index = Date
data['Date'] = pd.date_range('1999-1-1', periods=1096).strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)


# Enregistrement du dataframe data dans un fichier CSV
data.to_csv('data_LBIR1271.csv')


# Calcul des statistiques principales des variables

# Création d'un dataframe contenant les statistiques principales de chaque variable
stats = pd.DataFrame({"Moyenne": data.mean(), "Variance": data.var(), "Minimum": data.min(), "Maximum": data.max()})

# Affichage du dataframe stats
print(stats)


# Chargement du dataframe data depuis le fichier CSV
data = pd.read_csv('data_LBIR1271.csv', parse_dates=['Date'], index_col='Date')

# Convertir les valeurs de H45E et H45T de g/m³ à %
data['H45E'] = data['H45E'] * 0.1
data['H45T'] = data['H45T'] * 0.1

# Figure 1: Rain E, Rain T, H45E et H45T

# Création d'une figure contenant 4 sous-graphiques
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

# Plot de Rain_E (mm)
axs[0].plot(data.index, data['Rain_E'])
axs[0].set_title('Rain_E (mm)')

# Plot de Rain_T (mm)
axs[1].plot(data.index, data['Rain_T'])
axs[1].set_title('Rain_T (mm)')

# Plot de H45E (%)
axs[2].plot(data.index, data['H45E'].dropna(), 'o-',)
axs[2].set_title('H45E (%)')

# Plot de H45T (%)
axs[3].plot(data.index, data['H45T'].dropna(), 'o-')
axs[3].set_title('H45T (%)')

fig.suptitle('Figure 1: Rain_E, Rain_T, H45E, and H45T')
fig.tight_layout(pad=3.0, h_pad=1.5)
plt.show()


# Figure 2
fig, axs = plt.subplots(6, 1, figsize=(10, 15))

# Plot de Irradiance (kWh/m²)
axs[0].plot(data.index, data['Irradiance'])
axs[0].set_title('Irradiance (kWh/m²)')

# Plot de Rain (mm)
axs[1].plot(data.index, data['Rain'])
axs[1].set_title('Rain (mm)')

# Plot de Temp (°C)
axs[2].plot(data.index, data['Temp'])
axs[2].set_title('Temp (°C)')

# Plot de Wind (m/s)
axs[3].plot(data.index, data['Wind'])
axs[3].set_title('Wind (m/s)')

# Plot de Rel_hum (%)
axs[4].plot(data.index, data['Rel_hum'])
axs[4].set_title('Rel_hum (%)')

# Plot de SD (m)
axs[5].plot(data.index, data['SD'])
axs[5].set_title('SD (m)')

fig.suptitle('Figure 2: Irradiance, Rain, Temp, Wind, Rel_hum and SD')
fig.tight_layout(pad=3.0, h_pad=1.5)
plt.show()


# Figure 3
fig, axs = plt.subplots(6, 1, figsize=(10, 15))

# Ajout du premier sous-graphique : E153 (g/s)
axs[0].plot(data.index, data['E153']) 
axs[0].set_title('E153 (g/s)')

# Ajout du deuxième sous-graphique : E159 (g/s)
axs[1].plot(data.index, data['E159'])
axs[1].set_title('E159 (g/s)')

# Ajout du troisième sous-graphique : E161 (g/s)
axs[2].plot(data.index, data['E161'])
axs[2].set_title('E161 (g/s)')

# Ajout du quatrième sous-graphique : T13 (°C)
axs[3].plot(data.index, data['T13'])
axs[3].set_title('T13 (°C)')

# Ajout du cinquième sous-graphique : T21 (°C)
axs[4].plot(data.index, data['T21'])
axs[4].set_title('T21 (°C)')

# Ajout du sixième sous-graphique : T22 (°C)
axs[5].plot(data.index, data['T22'])
axs[5].set_title('T22 (°C)')

# Ajout d'un titre à la figure
fig.suptitle('Figure 3: E153, E159, E161, T13, T21 and T22')

# Ajustement de la disposition des sous-graphiques
fig.tight_layout(pad=3.0, h_pad=1.5)

# Affichage de la figure
plt.show()

# Sauvegarde de la figure
plt.savefig('figure3.png')



# Calculer la matrice de corrélation
corr_matrix = data.corr()

# Afficher la heatmap
sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de corrélation des variables météorologiques')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()



# Sélection des variables à inclure dans la figure
variables = ["Temp", "Rain", "Wind", "Irradiance", "Rel_hum", "SD", "Rain_E", "H45E", "E153"]

# Création d'un sous-dataframe contenant seulement ces variables
data_sub = data[variables]

# Création de la figure avec la fonction pairplot
sns.pairplot(data_sub, diag_kind="hist", corner=True)


# Sauvegarde de la figure
plt.savefig("pairplot.png")

