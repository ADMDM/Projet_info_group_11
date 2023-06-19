#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:39:07 2023

@author: adamdavidmalila
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import datetime as dt
import numpy as np
import scipy.io as sio
import evapotranspiration
from evapotranspiration import calculate_et0_matlab
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate

# Tâche 1: Chargement et affichage des données initiales

# Charger les données à partir du fichier 'data.mat'
data_mat = sio.loadmat('data.mat')
data = data_mat['data']

# Décaler les données d'une rangée vers le haut
data = np.roll(data, 1, axis=0)

# Convertir les données en DataFrame pandas pour une manipulation facile
data = pd.DataFrame(data, columns=['Days', 'Irradiance', 'Temperature', 'Precipitations', 'Vent', 'Humidité relative', 'Déficit de saturation', 'Flux E153', 'Flux E159', 'Flux E161', 'Flux T13', 'Flux T21', 'Flux T22', 'Humidité vol sol T', 'Humidité vol sol E', 'Pluie au sol E', 'Pluie au sol T'])

# Supprimer la première ligne (en-têtes de colonne)
data = data.drop(index=0)

# Créer une figure avec 4x4 sous-graphiques pour afficher les séries chronologiques de chaque variable
fig, axs = plt.subplots(4, 4, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    data.plot(x='Days', y=data.columns[i+1], ax=ax, legend=None)
    ax.set_title('Série chronologique de ' + data.columns[i+1])
    ax.set_xlabel('Jours')
    ax.set_ylabel(data.columns[i+1])
plt.tight_layout()
plt.show()
plt.savefig('mathfig1.png',dpi=300)

# Tâche 2 : Calcul et visualisation de ET0

# ET0 est un indicateur d'évapotranspiration. Il est calculé à partir des données à l'aide de la fonction calculate_et0_matlab().
et0_matlab = calculate_et0_matlab(data.to_numpy())

# Ajouter ET0 au DataFrame des données ET0 calculé est ajouté à notre tableau de données comme nouvelle colonne.
data['ET0'] = et0_matlab

# Afficher la série chronologique de ET0
plt.figure(figsize=(12, 6))
plt.plot(data['Days'], data['ET0'])
plt.xlabel("Jours")
plt.ylabel("ET0 (mm/jour)")
plt.title("Evapotranspiration ")
plt.show()
plt.savefig('mathfig2.png',dpi=300)


# Tâche 3 : Modélisation du stock d'eau dans le sol et visualisation des résultats

# Définition de l'équation différentielle pour le modèle du stock d'eau dans le sol 
#Parametres initiaux 
Dmax=15
Smin=70
Smax=150
Sinitial=150
lb=2
Kc=0.25

# Supprimer les valeurs manquantes dans ET0 et rain_T
et0_matlab.dropna(inplace=True)
rain_T=data.iloc[:1064,-2]
rain_T.dropna(inplace=True)

# Effectuer une interpolation linéaire pour P_interp et Et0_interp
P_interp = interpolate.interp1d(rain_T.index, rain_T.values, kind="linear")
Et0_interp = interpolate.interp1d(x=et0_matlab.index, y=et0_matlab['Et0'], kind="linear")

# Définir l'équation différentielle pour le modèle du stock d'eau dans le sol
def eq11(t, S,Dmax, Smin, Smax, lb, Kc):
    if (S - Smin) < 0 and lb<1:
        return 0
    else:
        P = P_interp(t)
        KcEt0 = Kc*(Et0_interp(t))
        S_new = P - KcEt0 - Dmax*(((S - Smin)/(Smax - Smin))**lb)
        return S_new

# Définition des temps d'évaluation pour l'intégration
t_eval = np.linspace(et0_matlab.index[0], et0_matlab.index[-1], 1064)
t_span=[et0_matlab.index[0], et0_matlab.index[-1]]

# Résolution de l'équation différentielle pour obtenir le stock d'eau dans le sol
stock_eau = solve_ivp(eq11, t_span, [Sinitial],args=(Dmax, Smin, Smax, lb, Kc), t_eval=t_eval)

# Afficher la série chronologique du stock d'eau dans le sol
plt.plot(stock_eau.t, stock_eau.y[0])
plt.xlabel("Jours")
plt.ylabel("Stock d'eau dans le sol (mm)")
plt.yticks(range(70, 151, 10))
plt.title("Modelisation du stock en eau : Methode de Runge Kunta")
plt.show()
plt.savefig('mathfig3.png',dpi=300)

# Tâche 4 : Analyse de sensibilité des paramètres du modèle

# Liste des paramètres et leurs valeurs correspondantes pour l'analyse de sensibilité
params = [('Dmax', np.linspace(1, 50, num=11, dtype=int)),
          ('Smin', np.linspace(60, 80, num=11, dtype=int)),
          ('Smax', np.linspace(100, 200, num=11, dtype=int)),
          ('lb', np.linspace(0.5, 5, num=10, dtype=float)),
          ('Kc', np.around(np.linspace(0.1, 0.4, num=11), decimals=2))]

# Boucle sur chaque paramètre pour créer une figure
# Chaque figure affiche le stock d'eau dans le sol en fonction du temps pour différentes valeurs du paramètre
for param, param_vals in params:
    fig, ax = plt.subplots(figsize=(6, 6))
    for param_val in param_vals:
        args = (Dmax, Smin, Smax, lb, Kc)
        if param == 'Dmax':
            args = (param_val, Smin, Smax, lb, Kc)
        elif param == 'Smin':
            args = (Dmax, param_val, Smax, lb, Kc)
        elif param == 'Smax':
            args = (Dmax, Smin, param_val, lb, Kc)
        elif param == 'lb':
            args = (Dmax, Smin, Smax, param_val, Kc)
        elif param == 'Kc':
            args = (Dmax, Smin, Smax, lb, param_val)
        
        var_stock_eau = solve_ivp(eq11, t_span, [Sinitial], args=args, t_eval=t_eval)
        ax.plot(var_stock_eau.t, var_stock_eau.y[0], label=f"{param} = {param_val}")
       
    # Réglages des axes et des étiquettes
    ax.set_xlabel("Jours")
    ax.set_ylabel("Stock d'eau dans le sol (mm)")
    ax.legend(fontsize=8, loc='upper right', labelspacing=0.05, ncol=3)
    ax.set_ylim(20, 275)
    # Affichage de la figure
    plt.show()
    plt.savefig('mathfig4.png',dpi=300)