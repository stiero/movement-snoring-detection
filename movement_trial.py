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

files = list(glob.glob("test_data/*.csv"))
files.sort()

#test_file = files[-1]
#files.pop(-1)

df = pd.DataFrame()

for file in files:
    df_open = pd.read_csv(file, header=None)
    df = pd.concat([df, df_open], axis=0)
    del df_open
    
df = df.reset_index(drop=True)

df.columns = ['obs']


obs_per_sec = round(df.shape[0]/(4*4*60))


mean = df.obs.mean()

median = df.obs.median()

#df['obs_ma_1'] = df['obs'].rolling(window=obs_per_sec).mean()

#df['obs_ma_2'] = df['obs'].rolling(window=2*obs_per_sec).mean()

#df['obs_ma_4'] = df['obs'].rolling(window=4*obs_per_sec).mean()

df['obs_mean'] = mean

df['diff_mean'] = mean - df['obs']

#df['diff_mean_ma_1s'] = df['diff_mean'].rolling(window=obs_per_sec).mean()

#df['diff_mean_ma_01s'] = df['diff_mean'].rolling(window=obs_per_sec//10).mean()

df['delta_mean'] = df['diff_mean'] / mean

df['delta_mean_ma_01s'] = df['delta_mean'].rolling(window=obs_per_sec//10).mean()

df['delta_mean_ma_5th'] = df['delta_mean'].rolling(window=obs_per_sec//5).mean()



aa = df.iloc[20000:30000]    #Movement

bb = df.iloc[6000:9000]     #Calm

cc = df.iloc[14000:20000]   #Movement

dd = df.iloc[45000:50000]   #Calm

ee = df.iloc[4000:7000]     #Movement

ff = df.iloc[30000:35000]   #Movement

from matplotlib.pyplot import figure
figure(num=None, figsize=(60, 6))
plt.plot(df['obs'])



#Flagging as movement those with delta_mean_ma_5th > 0.2 25 observations after recording

df_test = df.copy().reset_index()

for index, row in df_test.iterrows():

    df_test.iloc[index, 'movement'] = df_test['delta_mean_ma_5th']






#from scipy.fftpack import fft
#from scipy.signal.windows import bohman
#
#intervals = range(0, df1.shape[0], 250)
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
#df['ma_2'] = df['obs'].rolling(window=2, min_periods=1).mean()
#
#
#
#
#
#
#ft = fft(df.obs)
#
#sns.lineplot(data=df)
#
#sns.lineplot(data=ft)
#
#
#df.obs.mean()

