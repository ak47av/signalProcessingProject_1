import numpy as np
import math
import plotly as py
import plotly.graph_objs as go
from numpy.fft import fft, fftfreq, ifft
import ipywidgets as widgets
from plotly.subplots import make_subplots

Fs = 1000
t=np.linspace(0,1,Fs)

def update_plot(freq1,freq2,amp1,amp2):
    fig = make_subplots(rows=4,cols=1)   
    Fs=1000    
    
    t=np.linspace(0,1,Fs)
    M=amp1*np.sin(2*np.pi*freq1*t)            #Message Wave
    C=amp2*np.sin(2*np.pi*freq2*t)             #Carrier Wave

    F=1*np.cos(((2*np.pi*freq2 )-(2*np.pi*freq1))*t)/2    
    G=1*np.cos(((2*np.pi*freq2 )+(2*np.pi*freq1))*t)/2      
    
    Y=C+F-G              #Amplitude Modulation Formula, y=Ac*np.sin(wc*t)+(Ac*u/2)*np.cos((wc-wm)*t)-(Ac*u/2)*np.cos((wc+wm)*t)
                                                          #C              #F                         #G
    
    #For Frequency Response of the Amplitude modulated wave
    freqs = fftfreq(Fs)           
    mask = freqs>0
    fft_vals = np.asarray(fft(Y))
    fft_phase = 2.0*(fft_vals/Fs)
    fft_theo = 2.0*np.abs(fft_vals/Fs)
    
    trace1 = go.Scatter(
        x=t,
        y=M,
        mode='lines',
        name='Message Wave', 
        line=dict(
            shape='spline'
        )
    )
    
    trace2 = go.Scatter(
        x=t,
        y=C,
        mode='lines',
        name='Carrier Wave',
        line = dict(
            shape ='spline'
        )
    )
       
    trace3 = go.Scatter(
        x=t,
        y=Y,
        mode='lines',
        name='AM wave',
        line = dict(
            shape ='spline'
        )
    )
    
    f = np.arange(0,1000)
    
    trace4 = go.Scatter(
        x=f[mask],
        y=fft_theo[mask],
        mode='lines',
        name='Frequency Spectrum',
        line = dict(
            shape ='spline'
        )
    )
        
    fig.add_trace(trace1,row=1,col=1)
    fig.add_trace(trace2,row=2,col=1)
    fig.add_trace(trace3,row=3,col=1)
    fig.add_trace(trace4,row=4,col=1)
    fig.update_layout()
    py.offline.iplot(fig)
 
freq1 = widgets.FloatSlider(min=0, max=50, value=1, description='Freq1')
freq2 = widgets.FloatSlider(min=75, max=150, value=100, description='Carrier Frequency')
amp1 = widgets.FloatSlider(min=1,max=5,value=1,description="Message Amplitude")
amp2 = widgets.FloatSlider(min=1,max=3,value=1,description="Carrier Amplitude")

widgets.interactive(update_plot,freq1=freq1,freq2=freq2,amp1=amp1,amp2=amp2)
