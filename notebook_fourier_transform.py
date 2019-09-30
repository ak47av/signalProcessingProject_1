import plotly as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import ipywidgets as widgets
import numpy as np
from scipy import special
from scipy import signal

py.offline.init_notebook_mode(connected = True)

from numpy.fft import fft,ifft,fftfreq

Fs = 500
x = np.linspace(0,1,Fs)

#layout = go.Layout(
 #   title='SIMPLE EXAMPLE',
  #  yaxis=dict(
   #     title='volts'
    #),
    #xaxis=dict(
     #   title='nanoseconds'
    #)
#)


def update_plot(freq1,freq2):
    fig = make_subplots(rows=2,cols=1)
    
    data = []
    sig = np.sin(2*np.pi*freq1*x) + np.cos(2*np.pi*freq2*x)
    trace1 = go.Scatter(
        x=x,
        y=sig,
        mode='lines',
        name='sines', 
        line=dict(
            shape='spline'
        )
    )
    freqs = fftfreq(Fs)
    mask = freqs>0
    
    trace2 = go.Scatter(
        x=freqs[mask],
        y=2.0*np.abs(fft(sig/Fs))[mask],
        mode='lines',
        name='fourier transform',
        line = dict(
            shape ='spline'
        )
    )
    fig.add_trace(trace1,row=1,col=1)
    fig.add_trace(trace2,row=2,col=1)
    fig.update_layout()
    py.offline.iplot(fig)


freq1 = widgets.FloatSlider(min=1, max=100, value=1, description='freq1')
freq2 = widgets.FloatSlider(min=1, max=100, value=1, description='freq2')
widgets.interactive(update_plot,freq1=freq1,freq2=freq2)
