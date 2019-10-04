import plotly as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import ipywidgets as widgets
import numpy as np
from scipy import special
from scipy import signal
from numpy.fft import fft,ifft,fftfreq

py.offline.init_notebook_mode(connected = True)


Fs = 10000

Lx = 1

t = np.linspace(0,0.5,Fs)

def update_plot(freq1,index):
    fig = make_subplots(rows=3,cols=1)
    
    fc = 100
    
    carrier_wave = np.cos(2*np.pi*fc*t)
    trace1 = go.Scatter(
        x=t,
        y=carrier_wave,
        mode='lines',
        name='carrier wave', 
        line=dict(
            shape='spline'
        )
    )
    
    message_wave = np.cos(2*np.pi*freq1*t) 
    trace2 = go.Scatter(
        x=t,
        y=message_wave,
        mode='lines',
        name='sine message', 
        line=dict(
            shape='spline'
        )
    )
    
    mod_wave = np.cos(2*np.pi*fc*t + index*np.sin(2*np.pi*freq1*t))
    trace3 = go.Scatter(
        x=t,
        y=mod_wave,
        mode='lines',
        name='modulated wave', 
        line=dict(
            shape='spline'
        )
    )
    
    fig.add_trace(trace1,row=1,col=1)
    fig.add_trace(trace2,row=2,col=1)
    fig.add_trace(trace3,row=3,col=1)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Amplitude')
    fig.update_layout()
    py.offline.iplot(fig)
    
    
freq1 = widgets.FloatSlider(min=1, max=20, value=1, description='freq1')
index = widgets.FloatSlider(min=1, max=5, value=1, description='f sensitivity')
widgets.interactive(update_plot,freq1=freq1,index=index)    
    
