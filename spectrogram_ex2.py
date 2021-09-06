from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x2 = pd.read_csv("./ecg/AF/MUSE_20180111_155542_84000.csv",header=None)



fs = 500 # 샘플링 주파수


for i in range(0,1):
    x = x2.iloc[:,i]
    
    
    #x = np.fft.fftshift(x)
    #x = np.fft.fft(x)/len(x)
    
    #plt.plot(x)
    
    
    f, t, Sxx = signal.spectrogram(x, fs)
    
  
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.xlim(0,10)
    plt.ylim(0, 40)
    
    plt.colorbar(label='color')
    
    plt.show()
    
    
    i = i+1
        
    
    