#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:39:07 2023

@author: adamdavidmalila
"""

#Partie mathematqiue 


#%%Tache 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Importation des données climatiques
donnees_clim = sio.loadmat('data.mat')
data = pd.DataFrame(donnees_clim['data'], columns=["Days", "Irradiance", "Temperature", "Precipitations", "Vent", "Humidite relative",'deficit de saturation', "flux E153", "flux E159", "flux E161","flux T13", "flux T21", "flux T22", "hum vol sol T", "hum vol sol E", "pluie au sol E", "pluie au sol T"])

# Remplacement des zéros par des NaN
data = data.replace(0, np.nan)

# Création du subplot
fig, axs = plt.subplots(5, 3, figsize=(10, 10))

# Liste des noms des colonnes à tracer
cols = ['Irradiance', 'Temperature', 'Precipitations', 'Vent', 'Humidite relative',
        'deficit de saturation', 'flux E153', 'flux E159', 'flux E161',
        'flux T13', 'flux T21', 'flux T22', 'hum vol sol T', 'hum vol sol E', 
        'pluie au sol E', 'pluie au sol T']

# Boucle sur les rangées et les colonnes du subplot
for i, ax in enumerate(axs.flatten()):
    # Vérification si on a atteint la fin de la liste des noms de colonnes
    if i < len(cols):
        # Création du graphique
        col_name = cols[i]
        ax.plot(data['Days'], data[col_name])
        ax.set_title("Serie chronologique de {}".format(col_name))
        ax.set_xlabel("Jours")
    else:
        # Si on a atteint la fin de la liste, on cache l'axe
        ax.axis('off')

# Réglage des labels et de l'espacement entre les graphiques
plt.tight_layout()

# Affichage de la figure
plt.show()





#%%Tache 2 
from evapotranspiration import calculate_et0_matlab

# Calculer l'évapotranspiration de référence
et0 = calculate_et0_matlab(data.values)

et0 = et0.replace(0, np.nan)



# Afficher l'évapotranspiration de référence
plt.plot(data["Days"], et0)
plt.title("Évapotranspiration de référence en fonction du temps")
plt.xlabel("Jours")
plt.ylabel("Évapotranspiration de référence")
plt.show()

#%%Tache 3
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Définir l'équation différentielle
def equation(t, y):
    return y * np.sin(t)

# Intervalle de temps
t_span = [0, 10]

# Conditions initiales
y0 = 1

# Valeurs de t où on veut résoudre l'équation différentielle
t_eval = np.linspace(0, 10, 101)

# Paramètres initiaux
Dmax = 15
Smin = 70
Smax = 150
Kc = 0.25
lambda_ = 2

# Fonction pour l'équation 11
def equation_11(t, S, Dmax, Smin, Smax, Kc, lambda_, P, ET0):
    dSdt = P(t) - (Dmax * (S - Smin) * Kc * ET0(t)) / (Smax - Smin + lambda_ * (S - Smin))
    return dSdt

# Interpoler les données de précipitations et d'évapotranspiration de référence
P_interp = interp1d(data["Days"], data["Precipitations"], kind="linear", fill_value="extrapolate")
ET0_interp = interp1d(data["Days"], et0["Et0"], kind="linear", fill_value="extrapolate")

# Résoudre l'équation différentielle
initial_stock = 150
sol = solve_ivp(equation_11, (data["Days"].iloc[0], data["Days"].iloc[-1]), [initial_stock], args=(Dmax, Smin, Smax, Kc, lambda_, P_interp, ET0_interp), t_eval=data["Days"])

# Afficher les résultats
plt.plot(sol.t, sol.y[0])
plt.title("Stock d'eau dans la forêt en fonction du temps")
plt.xlabel("Jours")
plt.ylabel("Stock d'eau")
plt.show()


#%%Tache 3


import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Paramètres initiaux
Dmax = 15
Smin = 70
Smax = 150
Kc = 0.25
lambda_ = 2

# Fonction pour l'équation 11
def equation_11(t, S, Dmax, Smin, Smax, Kc, lambda_, P, ET0):
    dSdt = P(t) - (Dmax * (S - Smin) * Kc * ET0(t)) / (Smax - Smin + lambda_ * (S - Smin))
    return dSdt

# Interpoler les données de précipitations et d'évapotranspiration de référence
P_interp = interp1d(data["Days"], data["Precipitations"], kind="linear", fill_value="extrapolate")
ET0_interp = interp1d(data["Days"], et0["Et0"], kind="linear", fill_value="extrapolate")

# Résoudre l'équation différentielle
initial_stock = 150
sol = solve_ivp( equation_11 , (data["Days"].iloc[0], data["Days"].iloc[-1]), [initial_stock] , args=(Dmax, Smin, Smax, Kc, lambda_, P_interp, ET0_interp),t_eval=data["Days"], t_span=(0,1094))
print(data["Days"].iloc[0], data["Days"].iloc[-1])
print(t_eval[0], t_eval[-1])

# Afficher les résultats
plt.plot(sol.t, sol.y[0])
plt.title("Stock d'eau dans la forêt en fonction du temps")
plt.xlabel("Jours")
plt.ylabel("Stock d'eau")
plt.show()
