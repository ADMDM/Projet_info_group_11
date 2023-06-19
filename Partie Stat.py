#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module statistisques : Groupe 11 

Ce script effectue des analyses et des prédictions sur des données de la densité du flux de sève (SFD) et de variables climatiques.
"""

#%% Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importation des données à partir du fichier CSV du module informatqiue 
pre_data = pd.read_csv('data_LBIR1271.csv')

# Preparation des données
# Création du DataFrame "data" en sélectionnant les lignes 150 à 281 et en supprimant la colonne 'Date'
data = pre_data.iloc[150:282, :].drop(columns='Date')

# Réorganisation des colonnes et renommage pour ajouter les unités
order = ['E153', 'E159', 'E161', 'T13', 'T21', 'T22', 'Irradiance', 'Temp', 'Rain', 'Wind', 'Rel_hum', 'SD']
data = data.reindex(columns=order)
data.rename(columns={'Irradiance': 'Irradiance_[kWh/m2]', 'Temp': 'Temp_[°C]', 'Rain': 'Rain_[mm]', 'Wind': 'Wind_m/sec]', 'Rel_hum': 'Rel_hum_[%]'}, inplace=True)

# Création d'une figure pour afficher les différentes variables de sortie de flux solaire diffus (SFD)
ax = data.plot(y=["E153", "E159", "E161", "T13", "T21", "T22"], linewidth=1, grid=False, figsize=(16, 6))

# Étiquetage des axes pour une meilleure compréhension
plt.xlabel('Temps [jours de 1999]')
plt.ylabel("SFD [litres/(jour*dm²)]")

# Définition des limites de l'axe x et y pour une meilleure visualisation des données
ax.set_xlim(150,281)
ax.set_ylim(data.iloc[:,:6].min().min(),data.iloc[:,:6].max().max())

# Personnalisation des axes pour une meilleure visibilité
ax.xaxis.set_tick_params(direction='in', length=8, width=1, color='black', pad=5, top=True, bottom=True)
ax.yaxis.set_tick_params(direction='in', length=8, width=1, color='black', pad=5, right=True, left=True)
ax.set_facecolor('white')
ax.grid(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')

# Sauvegarde de la figure en tant que fichier PNG
plt.savefig('statfig1.png', dpi=300)
plt.show()

# Calcul des moyennes estimées et des variances estimées de SFD pour chaque arbre
m = data.mean()[:6]
v = data.var()[:6]

# Affichage des moyennes estimées et des variances estimées dans un tableau
stats_table = pd.DataFrame({'Arbre': ['E153', 'E159', 'E161', 'T13', 'T21', 'T22'],
                            'Moyenne estimée': m,
                            'Variance estimée': v})
print(stats_table)

SFD_mean_var = pd.concat([m, v], axis=1)
print(SFD_mean_var)

sfd=data.iloc[:,:6]
sfd_corr = sfd.corr()
print(sfd_corr)

#%% Calcul des corrélations

# Calcul de la matrice de corrélation R de taille 12 × 12
R = data.corr()

# Extraction des sous-matrices de corrélation
Rz = R.iloc[:6, :6]  # Corrélations entre les SFD des différents arbres
Rclim = R.iloc[6:, 6:]  # Corrélations entre les différentes variables climatiques
Rcross = R.iloc[6:, :6]  # Corrélations entre les variables climatiques et les SFD

# Classement des variables climatiques en fonction de leur corrélation avec les valeurs de SFD
correlation_with_SFD = Rcross.mean(axis=0)
clim_vars = ['Irradiance', 'Temp', 'Rain', 'Wind', 'Rel_hum', 'SD']
clim_sort = pd.DataFrame({'Variable climatique': clim_vars, 'Corrélation moyenne avec SFD': correlation_with_SFD})
clim_sort = clim_sort.sort_values(by='Corrélation moyenne avec SFD', ascending=False)
print(clim_sort)

# Identification des deux variables climatiques les plus corrélées au SFD
top_two_vars = clim_sort.head(2)['Variable climatique'].values
print("Les deux variables climatiques les plus corrélées au SFD :", top_two_vars)

# Explication de la logique derrière la corrélation entre le flux de sève et ces deux variables climatiques
# La corrélation positive entre le flux de sève et le vent (Wind) peut s'expliquer par l'effet de l'évapotranspiration accrue et de la convection induite par le vent. Le vent favorise l'évaporation de l'eau au niveau des feuilles et crée un mouvement d'air qui peut stimuler la montée de sève pour répondre à la demande accrue en eau.
# La corrélation positive entre le flux de sève et l'humidité relative (Rel_hum) est due à l'influence de l'humidité sur l'évapotranspiration et la demande en eau de l'arbre. Lorsque l'humidité relative est élevée, l'air est déjà saturé en humidité, ce qui réduit la demande en eau de l'arbre et diminue le flux de sève. En revanche, lorsque l'humidité relative est basse, l'air a une plus grande capacité à absorber l'humidité, ce qui augmente la demande en eau de l'arbre et favorise une plus grande montée de sève.

#%% Prédiction du flux de sève

# Calcul de la matrice de covariance de taille 12x12
C = data.cov()

# Calcul de la matrice de covariance moyenne C_mean
C_mean = pd.DataFrame(columns=['SFD', 'Irradiance', 'Temp', 'Rain', 'Wind', 'Rel_hum', 'SD'], index=['SFD', 'Irradiance', 'Temp', 'Rain', 'Wind', 'Rel_hum', 'SD'])

# Calcul de la moyenne des variables SFD pour chaque ligne de la sous-matrice de covariance C
SFD_mean = pd.DataFrame(C.iloc[6:, :6].mean(axis=1))

# Renommage de la colonne de la DataFrame des moyennes des variables SFD en 'SFD_mean'
SFD_mean.rename(columns={0: 'SFD_mean'}, inplace=True)

# Attribution des valeurs de la sous-matrice de covariance des variables climatiques à la matrice de covariance moyenne
C_mean.iloc[1:, 1:] = C.iloc[6:, 6:]

# Attribution des moyennes des variables SFD aux colonnes et lignes correspondantes de la matrice de covariance moyenne
C_mean.iloc[1:, :1] = SFD_mean
C_mean.iloc[:1, 1:] = SFD_mean.transpose()

# Attribution de la moyenne des éléments diagonaux de la sous-matrice de covariance des variables SFD à l'élément diagonal de la matrice de covariance moyenne
C_mean.iloc[:1, :1] = np.diag(np.diag(C.iloc[:6, :6])).mean()

#%% Calcul des variances conditionnelles
σ2_SFD = C_mean.iloc[:1, :1].values

# Calcul de la variance conditionnelle pour une variable climatique spécifique (b)
def var_cond(b):
    result = C_mean.iloc[:1, :1].values - ((C_mean.iloc[:1, b:b+1].values)**2) / C_mean.iloc[b:b+1, b:b+1].values
    return(result)

var_cond_irr = var_cond(1)
# Calcul de σ² (variance de SFD)
sigma2_SFD = C_mean.iloc[0, 0]

# Calcul des variances conditionnelles σ²_SFD|j pour chaque variable climatique
var_cond_single = []
for i in range(1, 7):
    var_cond_single.append(var_cond(i)[0][0])

# Calcul des variances conditionnelles σ²_SFD|j,k pour chaque paire de variables climatiques
def var_cond_pair(j, k):
    Cjk = C_mean.iloc[j, k]
    Cjj = C_mean.iloc[j, j]
    Ckk = C_mean.iloc[k, k]
    Cj = C_mean.iloc[0, j]
    Ck = C_mean.iloc[0, k]
   
    result = sigma2_SFD - ((Cj**2) / Cjj) - ((Ck**2) / Ckk) + (2 * Cj * Ck * Cjk) / (Cjj * Ckk)
    return result

var_cond_pairs = []
for i in range(1, 7):
    row = []
    for j in range(1, 7):
        if i == j:
            row.append(0)
        else:
            row.append(var_cond_pair(i, j))
    var_cond_pairs.append(row)

# Affichage des résultats
print("σ² (variance de SFD) :", sigma2_SFD)
print("\nVariances conditionnelles σ²_SFD|j pour chaque variable climatique :", var_cond_single)
print("\nVariances conditionnelles σ²_SFD|j,k pour chaque paire de variables climatiques :")
for row in var_cond_pairs:
    print(row)
    
# Calcul de R
Rj2 = (σ2_SFD - var_cond_single) / (σ2_SFD)
Rjk2 = (σ2_SFD - var_cond_pairs) / (σ2_SFD)
    
# Création d'un DataFrame pour les résultats
columns = ['σ²_SFD', 'σ²_SFD|Irradiance', 'σ²_SFD|Temp', 'σ²_SFD|Rain', 'σ²_SFD|Wind', 'σ²_SFD|Rel_hum', 'σ²_SFD|SD']
index = ['Variance', 'Var_Cond_Single', 'Var_Cond_Pair', 'R²', 'R²_j,k']
results = pd.DataFrame(index=index, columns=columns)

# Sélectionnez les variables climatiques j et k
j_data = data['Temp_[°C]']
k_data = data['Irradiance_[kWh/m2]']

#%% Calcul des valeurs ai, b et c

# Sélection des variables SFD et climatiques pour le calcul des valeurs a, b et c
X = [np.array(sfd.loc[:, "E153"]),
      np.array(sfd.loc[:, "E159"]),
      np.array(sfd.loc[:, "E161"]),
      np.array(sfd.loc[:, "T13"]),
      np.array(sfd.loc[:, "T21"]),
      np.array(sfd.loc[:, "T22"])]

# Calcul des moyennes des variables SFD et climatiques
Xmoy = [np.array(sfd.loc[:, "E153"].mean()),
        np.array(sfd.loc[:, "E159"].mean()),
        np.array(sfd.loc[:, "E161"].mean()),
        np.array(sfd.loc[:, "T13"].mean()),
        np.array(sfd.loc[:, "T21"].mean()),
        np.array(sfd.loc[:, "T22"].mean())]

# Calcul des coefficients b et c à partir de la covariance et de la variance
ba_Irr_hum = np.array(C_mean.iloc[[0], [1, 5]])
aa_Irr_hum = C_mean.iloc[[1, 5], [1, 5]].astype(float)
b_c = np.dot(ba_Irr_hum, np.linalg.inv(aa_Irr_hum))
b_c_df = pd.DataFrame(b_c, index=['b_c'], columns=['b', 'c'])

# Calcul des coefficients a pour chaque variable SFD et climatique
a = []
for i in range(0, 6):
    ai = (Xmoy[i] - b_c_df.loc[:, 'b'] * data.iloc[:, 6].mean() - b_c_df.loc[:, 'c'] * data.iloc[:, 10].mean()).item()
    a.append(ai)

# Création du DataFrame final avec les coefficients a, b et c
a_df = pd.DataFrame(a, index=sfd.columns, columns=['a'])
b_repeated = np.repeat(b_c_df.iloc[0, 0], 6)
c_repeated = np.repeat(b_c_df.iloc[0, 1], 6)

tableau_final = pd.DataFrame({
    'a': a_df['a'],
    'b': b_repeated,
    'c': c_repeated})

print(tableau_final)

#%% Espérance

# Calcul de l'espérance pour chaque variable SFD et climatique
esperance = pd.DataFrame(index=sfd.index, columns=sfd.columns)
for i in range(0, 6):
    for j in range(0, 132):
        E = a_df.iloc[i, 0] + b_c_df.loc[:, 'b'] * data.iloc[j, 6] + b_c_df.loc[:, 'c'] * data.iloc[j, 10]
        esperance.iloc[j, i] = ((a_df.iloc[i, 0] + b_c_df.loc[:, 'b'] * data.iloc[j, 6] + b_c_df.loc[:, 'c'] * data.iloc[j, 10]).item())

#%% Test sur l'arbre T13 pour vérifier nos réponses

plt.figure(figsize=(7, 5), dpi=300)
# Valeurs mesurées
plt.plot(sfd.index, data.iloc[:, 3], linewidth=1)

# Valeurs prédites
plt.plot(sfd.index, esperance.iloc[:, 3], linewidth=1)
plt.xlabel("Jours (du 31 mai au 09 octobre de l'année 1999)")
plt.ylabel("SFD [litres/(jour*dm²)]")
plt.legend(["Mesurées", "Prédites"])

ax = plt.gca()  # récupérer les axes actuels
ax.tick_params(axis='x', direction='in', length=4, width=1, color='black', pad=5, top=True, bottom=True)
ax.tick_params(axis='y', direction='in', length=4, width=1, color='black', pad=5, right=True, left=True)
ax.set_facecolor('white')
ax.grid(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')

plt.title('T13')
plt.xlim(150, 281)
plt.ylim(data.iloc[:, 3].min(), data.iloc[:, 3].max())
plt.show()

#%% Figure 2.2 pour l'arbre E161 choisi

pente = 1
c = esperance.iloc[:, 2].min()

figure4 = pd.DataFrame()
figure4['Observées'] = data.iloc[:, 2]
figure4['Prédites'] = esperance.iloc[:, 2]

# Définition des limites des axes en fonction des valeurs min et max des données
x_min = min(figure4['Observées'].min(), figure4['Prédites'].min())
x_max = max(figure4['Observées'].max(), figure4['Prédites'].max())
y_min = x_min
y_max = x_max

# Création du graphique
fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
ax.scatter(figure4['Prédites'], figure4['Observées'], color='blue', zorder=10, linewidths=1, s=10, clip_on=False)

# Tracé de la diagonale
ax.plot([x_min, x_max], [y_min, y_max], c='black', linewidth=0.9)

# Définition des limites des axes
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Définition des étiquettes des axes
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.xaxis.set_tick_params(direction='in', length=4, width=1, color='black', pad=5, top=True, bottom=True)
ax.yaxis.set_tick_params(direction='in', length=4, width=1, color='black', pad=5, right=True, left=True)
plt.xlabel("SFD prédits")
plt.ylabel("SFD observés")

# Modification du fond du graphique et des contours
ax.set_facecolor('white')
ax.grid(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
plt.title('E161')

plt.savefig('statE161_2.2.png', dpi=300)
plt.show()
