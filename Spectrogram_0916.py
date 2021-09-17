from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x2 = pd.read_csv("./AF/MUSE_20180111_155542_84000.csv",header=None)



fs = 500 # 샘플링 주파수


for i in range(0,1):
    x = x2.iloc[:,i]
    

    f, t, Sxx = signal.spectrogram(x, fs)
    # 추가
    Sxx = abs(Sxx)
    mask = Sxx > 0
    Sxx[mask] = np.log(Sxx[mask])
    
    plt.pcolormesh(t, f, Sxx, cmap='PiYG', shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.xlim(0,100)
    #plt.ylim(0,50)
    
    cc = plt.colorbar(label='color')
    plt.clim(-30, 10)
    plt.show()
    
    

        
    
    