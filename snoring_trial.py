#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:40:46 2019

@author: msr
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


import glob 

files_snoring = list(glob.glob("snoring_test_data/*.csv"))
files_snoring.sort()

test_file_snoring = files_snoring[-1]
files_snoring.pop(-1)

df_snoring = pd.DataFrame()

for file in files_snoring:
    df_open = pd.read_csv(file, header=None)
    df_snoring = pd.concat([df_snoring, df_open], axis=0)
    del df_open
    
    
df_snoring = df_snoring.reset_index(drop=True)

df_snoring.columns = ['obs']

obs_per_sec = round(df_snoring.shape[0]/(4*4*60))

mean = df_snoring.obs.mean()

median = df_snoring.obs.median()


df_snoring['obs_mean'] = mean

df_snoring['diff_mean'] = mean - df_snoring['obs']


df_snoring['delta_mean'] = (df_snoring['diff_mean'] / mean)

df_snoring['delta_mean_rolling_100ms'] = df_snoring['delta_mean'].rolling(window=obs_per_sec//10).mean()

df_snoring['delta_mean_rolling_200ms'] = df_snoring['delta_mean'].rolling(window=obs_per_sec//5).mean()


df_snoring['delta_mean_rolling_std_100ms'] = df_snoring['delta_mean'].rolling(window=25).std()

df_snoring['roll_mean_obs_10'] = df_snoring['obs'].rolling(window=10).mean()

df_snoring['diff_1'] = df_snoring['obs'].diff(1)

df_snoring['rolling_std_2'] = df_snoring['obs'].rolling(window=2).std()

df_snoring['rolling_std_2_rolling_mean_10'] = df_snoring['rolling_std_2'].rolling(window=10).mean()

df_snoring['diff_5_abs'] = abs(df_snoring['obs'].diff(5))

df_snoring['diff_5_abs_centred'] = abs(df_snoring['diff_5_abs'] - df_snoring['diff_5_abs'].mean())

df_snoring['diff_5_abs_rolling_mean'] = df_snoring['diff_5_abs'].rolling(window=10).sum()

df_snoring['autocorr_10'] = df_snoring['obs'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False)

df_snoring['autocorr_5'] = df_snoring['obs'].rolling(window=5).apply(lambda x: x.autocorr(), raw=False)

df_snoring['autocorr_5_plus_diff'] = df_snoring['autocorr_5'] / df_snoring['diff_5_abs']


df_snoring['diff_mean_rolling'] = df_snoring['diff_mean'].rolling(window=obs_per_sec//10).mean()

aa = df_snoring.iloc[600:800]  # Snoring

bb = df_snoring.iloc[:500]      # No snoring

cc = df_snoring.iloc[1650:2000]      # Snoring

dd = df_snoring.iloc[54200:54600]   # Snoring

ee = df_snoring.iloc[52900:53300]   # No snoring

ff = df_snoring.iloc[42000:42400]

gg = df_snoring.iloc[42600: 43000]


bk = df_snoring.copy()

df_snoring['snoring'] = 0

for index, row in df_snoring.iterrows():
    
    df_snoring.loc[index, 'snoring'] = np.where(
            df_snoring.loc[index:index+249, 'rolling_std_2'].mean() > 0.75, 1, 0)






import statistics

from statistics import StatisticsError

snoring_detected = []

for sec in range(0, df_snoring.shape[0], obs_per_sec):
    try:
        verdict = bool(np.where(statistics.mode(df_snoring.iloc[sec:sec+obs_per_sec]['snoring']) == 1, True, False))
    except StatisticsError:
        verdict = True
        
    snoring_detected.append(verdict)















df_snoring_test = pd.read_csv(test_file_snoring, header=None)


df_snoring_test = df_snoring_test.reset_index(drop=True)

df_snoring_test.columns = ['obs']

obs_per_sec = round(df_snoring_test.shape[0]/(4*4*60))

mean = df_snoring_test.obs.mean()

median = df_snoring_test.obs.median()


df_snoring_test['obs_mean'] = mean

df_snoring_test['diff_mean'] = mean - df_snoring_test['obs']


df_snoring_test['delta_mean'] = (df_snoring_test['diff_mean'] / mean)

df_snoring_test['delta_mean_rolling_100ms'] = df_snoring_test['delta_mean'].rolling(window=obs_per_sec//10).mean()

df_snoring_test['delta_mean_rolling_200ms'] = df_snoring_test['delta_mean'].rolling(window=obs_per_sec//5).mean()


df_snoring_test['delta_mean_rolling_std_100ms'] = df_snoring_test['delta_mean'].rolling(window=25).std()

df_snoring_test['roll_mean_obs_10'] = df_snoring_test['obs'].rolling(window=10).mean()

df_snoring_test['diff_1'] = df_snoring_test['obs'].diff(1)

df_snoring_test['rolling_std_2'] = df_snoring_test['obs'].rolling(window=2).std()

df_snoring_test['rolling_std_2_rolling_mean_10'] = df_snoring_test['rolling_std_2'].rolling(window=10).mean()

df_snoring_test['diff_5_abs'] = abs(df_snoring_test['obs'].diff(5))

df_snoring_test['diff_5_abs_centred'] = abs(df_snoring_test['diff_5_abs'] - df_snoring_test['diff_5_abs'].mean())

df_snoring_test['diff_5_abs_rolling_mean'] = df_snoring_test['diff_5_abs'].rolling(window=10).sum()

df_snoring_test['autocorr_10'] = df_snoring_test['obs'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False)

df_snoring_test['autocorr_5'] = df_snoring_test['obs'].rolling(window=5).apply(lambda x: x.autocorr(), raw=False)

df_snoring_test['autocorr_5_plus_diff'] = df_snoring_test['autocorr_5'] / df_snoring_test['diff_5_abs']


df_snoring_test['diff_mean_rolling'] = df_snoring_test['diff_mean'].rolling(window=obs_per_sec//10).mean()






df_snoring_test['snoring'] = 0

for index, row in df_snoring_test.iterrows():
    
    df_snoring_test.loc[index, 'snoring'] = np.where(
            df_snoring_test.loc[index:index+249, 'rolling_std_2'].mean() > 0.75, 1, 0)




import statistics

from statistics import StatisticsError

snoring_detected = pd.Series()

timestep = 25

for sec in range(0, df_snoring_test.shape[0], timestep):
    try:
        verdict = bool(np.where(statistics.mode(df_snoring_test.iloc[sec:sec+timestep]['snoring']) == 1, True, False))
    except StatisticsError:
        verdict = True
        
    snoring_detected[str(sec)+":"+str(sec+timestep)] = verdict









dd.autoreg_std.mean()

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


result = autocorr(aa.obs)


np.polyfit(cc.index, cc.autoreg_std_rolling_mean_abs, 1)

from scipy import stats

stats.signaltonoise(aa)

import statsmodels.api as sm

decomposition = sm.tsa.filters.filtertools.recursive_filter(aa.obs)

decomposition = sm.tsa.seasonal_decompose(aa.obs, model='additive')






from matplotlib.pyplot import figure
figure(num=None, figsize=(20, 6))
plt.plot(df1.iloc[:5000]['obs'])













from scipy.fftpack import fft

sample_rate = 250

N = 960 * sample_rate

time = np.linspace(0, 2, N)




y = fft(cc.obs)

y = 2/N * np.abs (freq_data [0:np.int (N/2)])



cc['fft'] = y


f, t, Sxx = scipy.signal.spectrogram(cc.obs, 1)

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()






import plotly.plotly as py
import plotly.graph_objs as go

amplitude = np.absolute(np.fft.fft(cc.obs))[1:]

amplitude = amplitude[0:(int(len(amplitude)/2))]


frequency = np.linspace(0,10000, len(amplitude))

plt.plot(frequency, amplitude)

trace = go.Scatter(x = frequency, y = amplitude)
data = [trace]
layout = go.Layout(title="Frequency vs Amplitude after FFT",
                  xaxis=dict(title='Frequency'),
                  yaxis=dict(title='Amplitude'))
fig = go.Figure(data=data, layout=layout)

py.iplot(fig)




