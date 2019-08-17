#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:44:27 2019

@author: tauro
"""
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


import glob 

files_movement = list(glob.glob("movement_test_data/*.csv"))
files_movement.sort()

test_file_movement = files_movement[-1]
files_movement.pop(-1)

df_movement = pd.DataFrame()

for file in files_movement:
    df_open = pd.read_csv(file, header=None)
    df_movement = pd.concat([df_movement, df_open], axis=0)
    del df_open
    
df_movement = df_movement.reset_index(drop=True)

df_movement.columns = ['obs']


obs_per_sec = round(df_movement.shape[0]/(4*4*60))


mean = df_movement.obs.mean()

median = df_movement.obs.median()

#df_movement['obs_ma_1'] = df_movement['obs'].rolling(window=obs_per_sec).mean()

#df_movement['obs_ma_2'] = df_movement['obs'].rolling(window=2*obs_per_sec).mean()

#df_movement['obs_ma_4'] = df_movement['obs'].rolling(window=4*obs_per_sec).mean()

df_movement['obs_mean'] = mean

df_movement['diff_mean'] = mean - df_movement['obs']

#df_movement['diff_mean_ma_1s'] = df_movement['diff_mean'].rolling(window=obs_per_sec).mean()

#df_movement['diff_mean_ma_01s'] = df_movement['diff_mean'].rolling(window=obs_per_sec//10).mean()

df_movement['delta_mean'] = abs(df_movement['diff_mean'] / mean)

df_movement['delta_mean_ma_100ms'] = df_movement['delta_mean'].rolling(window=obs_per_sec//10).mean()

df_movement['delta_mean_ma_200ms'] = df_movement['delta_mean'].rolling(window=obs_per_sec//5).mean()

df_movement['diff_1'] = abs(df_movement['obs'].diff(1))

df_movement['diff_1_rolling_mean_abs'] = df_movement['diff_1'].rolling(window=10).mean()

df_movement['diff_2'] = abs(df_movement['diff_1_rolling_mean_abs'].diff())


aa = df_movement.iloc[20000:30000]    # Long movement (possibly intense)

bb = df_movement.iloc[50000:52000]     # Calm

cc = df_movement.iloc[64000:65000]   # Very Short movement

dd = df_movement.iloc[95000:98000]   # Medium movement

ee = df_movement.iloc[4000:7000]     #Movement

ff = df_movement.iloc[30000:35000]   #Movement

#from matplotlib.pyplot import figure
#figure(num=None, figsize=(60, 6))
#plt.plot(df_movement['obs'])



#Flagging as movement those with delta_mean_ma_5th > 0.2 25 observations after recording

#df_movement_test = df_movement.copy()

df_movement['movement'] = 0

for index, row in df_movement.iterrows():

    df_movement.loc[index-49:index+1, 'movement'] = np.where(
            
            (df_movement.iloc[index]['delta_mean_ma_200ms'] > 0.2) & (~np.isnan(df_movement.iloc[index]['delta_mean_ma_200ms']))
            , 1, 0)

    
df_movement.loc[(np.isnan(df_movement['delta_mean_ma_200ms'])), 'delta_mean_ma_200ms'] = 0



import statistics

from statistics import StatisticsError


#movement_detected = []
#
#for sec in range(0, df_movement.shape[0], obs_per_sec):
#    try:
#        verdict = bool(np.where(statistics.mode(df_movement.iloc[sec:sec+obs_per_sec]['movement']) == 1, True, False))
#    except StatisticsError:
#        verdict = True
#        
#    movement_detected.append(verdict)










df_movement_test = pd.read_csv(test_file_movement, header=None, names=['obs'])

#mean_test = df_movement_test.obs.mean()

#median_test = df_movement_test.obs.median()

df_movement_test['obs_mean'] = mean

df_movement_test['diff_mean'] = mean - df_movement_test['obs']

#df_movement['diff_mean_ma_1s'] = df_movement['diff_mean'].rolling(window=obs_per_sec).mean()

#df_movement['diff_mean_ma_01s'] = df_movement['diff_mean'].rolling(window=obs_per_sec//10).mean()

df_movement_test['delta_mean'] = abs(df_movement_test['diff_mean'] / mean)

df_movement_test['delta_mean_ma_100ms'] = df_movement_test['delta_mean'].rolling(window=obs_per_sec//10).mean()

df_movement_test['delta_mean_ma_200ms'] = df_movement_test['delta_mean'].rolling(window=obs_per_sec//5).mean()



df_movement_test['movement'] = 0

for index, row in df_movement_test.iterrows():

    df_movement_test.loc[index-49:index+1, 'movement'] = np.where(
            
            (df_movement_test.iloc[index]['delta_mean_ma_200ms'] > 0.2) & (~np.isnan(df_movement_test.iloc[index]['delta_mean_ma_200ms']))
            , 1, 0)

    
df_movement_test.loc[(np.isnan(df_movement_test['delta_mean_ma_200ms'])), 'delta_mean_ma_200ms'] = 0






import statistics

from statistics import StatisticsError


movement_detected = pd.Series()

timesteps = np.arange(0, df_movement_test.shape[0], obs_per_sec)

for sec in timesteps:
    try:
        verdict = bool(np.where(statistics.mode(df_movement_test.iloc[sec:sec+obs_per_sec]['movement']) == 1, True, False))
    except StatisticsError:
        verdict = True
        
    movement_detected[str(sec)] = verdict




plt.plot(df_movement_test.obs)

for i, sec in enumerate(timesteps):
    if movement_detected[str(sec)] == True:
        plt.axvspan(sec, sec+obs_per_sec-1, alpha=0.3, color='r', zorder=20)





dd = df_movement_test.iloc[95000:98000]   # Medium movement




#from scipy.fftpack import fft
#from scipy.signal.windows import bohman
#
#intervals = range(0, df_movement1.shape[0], 250)
#
#
#max_amp_index = {}
#
#for interval in intervals:
#    window = bohman(250)
#    
#    max_index = np.argmax(window)
#    
#    max_val = window.max()
#    
#    max_amp_index[str(interval)] = (max_val, max_index)
#
#
#
#plt.plot(window)
#
#
#df_movement['ma_2'] = df_movement['obs'].rolling(window=2, min_periods=1).mean()
#
#
#
#
#
#
#ft = fft(df_movement.obs)
#
#sns.lineplot(data=df_movement)
#
#sns.lineplot(data=ft)
#
#
#df_movement.obs.mean()

