# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:05:46 2019

@author: Roshan
"""

import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal
from numpy import fft

fs = 1000
lt = 100
t = np.linspace(0,lt,fs)
w = 2*np.pi/lt

plt.subplot(311)
y = 75*np.sin(w*250*t) + 100*np.sin(w*5*t) + 15*np.cos(w*210*t) + 50*np.cos(w*400*t)
plt.plot(t,y)

freq =fft.fftfreq(fs)*fs
mask = freq > 0

fft_vals = np.array(fft.fft(y))
norm_fft = (2.0/fs)*np.abs(fft_vals)

plt.subplot(312)
plt.plot(freq[mask],norm_fft[mask])

plt.subplot(313)
plt.plot(t,fft.ifft(fft_vals))
plt.show()