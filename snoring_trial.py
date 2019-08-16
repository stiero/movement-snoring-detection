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

files = list(glob.glob("snoring_test_data/*.csv"))
files.sort()

test_file = files[-1]
files.pop(-1)

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


df['obs_mean'] = mean

df['diff_mean'] = mean - df['obs']


df['delta_mean'] = (df['diff_mean'] / mean)

df['delta_mean_ma_100ms'] = df['delta_mean'].rolling(window=obs_per_sec//10).mean()

df['delta_mean_ma_200ms'] = df['delta_mean'].rolling(window=obs_per_sec//5).mean()


df['delta_mean_std_10ms'] = df['delta_mean'].rolling(window=10).std()

df['roll_mean_obs_10'] = df['obs'].rolling(window=10).mean()

df['autoreg_4'] = df['obs'].diff(1)

df['autoreg_std'] = df['obs'].rolling(window=2).std()

df['autoreg_std_rolling_mean'] = df['autoreg_std'].rolling(window=10).mean()

df['autoreg_abs'] = abs(df['obs'].diff(5))

df['autoreg_abs_mean'] = abs(df['autoreg_abs'] - df['autoreg_abs'].mean())

df['autoreg_std_rolling_mean_abs'] = df['autoreg_abs'].rolling(window=10).sum()

df['autocorr'] = df['obs'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False)

df['autocorr_5'] = df['obs'].rolling(window=5).apply(lambda x: x.autocorr(), raw=False)

df['autocorr_5_plus_diff'] = df['autocorr_5'] / df['autoreg_4']


df['diff_mean_rolling'] = df['diff_mean'].rolling(window=obs_per_sec//10).mean()

aa = df.iloc[600:800]  # Snoring

bb = df.iloc[:500]      # No snoring

cc = df.iloc[1650:2000]      # Snoring






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




