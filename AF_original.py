from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x2 = pd.read_csv("./ecg/AF/MUSE_20180111_155542_84000.csv",header=None)


fs = 500 # 샘플링 주파수


for i in range(0,1):
    x = x2.iloc[:,i]
    
    
    #xx = np.fft.fftshift(x)
    
    xx = np.fft.fft(x)/len(x)
    xx=np.fft.fftshift(xx)
    #print(x)
    plt.plot(xx)
    
    
    
    f, t, Sxx = signal.spectrogram(xx, fs)
    
  
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    
    plt.title('fftshift')

    #plt.xlim(0,5000)
    #plt.ylim(-700,700)
    
    
    plt.show()
    
    
    i = i+1
        
    
    