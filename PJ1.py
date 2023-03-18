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



   