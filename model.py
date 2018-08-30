# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:24:08 2018

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

style.use('seaborn')

df = pd.read_csv('Thermal Liquids Data.csv')
df.set_index('Run',inplace=True)

def temperature_function(initial_temperature,plate_temperature,thermal_conductivity,
                         density =990,specific_heat_capacity=4186,depth=0.17):
    return ((initial_temperature-plate_temperature)*
            np.exp((-thermal_conductivity/(density*specific_heat_capacity*depth**2))*x)
            + plate_temperature)
    
def function(c,A,t,x):
    return (A*np.exp(-x/t)+c)

def gradient(A,t,x1):
    return (-A/t*np.exp(-x1/t))
def line(m,x):
    return (m*x+(function(df['y0'][i],df['A1'][i],df['t1'][i],gradient_point)) - gradient(df['A1'][i],df['t1'][i],gradient_point)*gradient_point)
print(df)
x = np.linspace(0,7200,1000)
gradient_point = 100
plot_choice = [10,12,14,15]
for i in plot_choice:
    #plt.plot(x,function(df['y0'][i],df['A1'][i],df['t1'][i],x),label = df['Comment'][i])
    plt.plot(function(df['y0'][i],df['A1'][i],df['t1'][i],x),line(gradient(df['A1'][i],df['t1'][i],x),x),label = df['Comment'][i])
    print(gradient(df['A1'][i],df['t1'][i],gradient_point))

#plt.plot(x,temperature_function(19,80,630),label = 'theoretical')
plt.ylim(15,80)
plt.legend()
plt.show()

