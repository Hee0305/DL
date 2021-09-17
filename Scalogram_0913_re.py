# -*- coding: utf-8 -*-
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
my_dpi = 96

'''
disease = ['AF','AFIB','AT','AVNRT','AVRT','SAAWR','SB','SI','SR','ST','SVT']
for path in disease:
    os.mkdir(path) 
'''


folder = ['AF', 'SB', 'SI', 'AVNRT', 'ST', 'SAAWR', 'AT', 'SVT', 'AFIB', 'AVRT', 'SR']


for i in range(0,1): # AF ~ SVT
    F = folder[i]
    FF = "./"+folder[i]
    print(FF) 
    file = os.listdir(FF)
    print("file: {}".format(file))
    
    for j in range(len(file)): # MUSE 파일
        MFile = file[j]
        print(MFile)
        
        MName = MFile.split('.')
        MM = MName[0]
        
        df = pd.read_csv(FF + "/" + MFile)
        print(df)
        
        # Patient scalogram 
        scalo = './image/scalogram/'+ F + '/' + MM
        os.mkdir(scalo)
        
        
        
        '''
        # Patient spectrogram
        spectro = './image/spectrogram/'+ F + '/' + MM
        os.mkdir(spectro)
        '''
        img = df.iloc[:,1] # lead2 만
        print(img)
        wavelet = signal.ricker
                        
        widths = np.arange(1,1000)
        cwtmatr = signal.cwt(img, wavelet, widths)
        
        
        plt.figure(figsize=(299/my_dpi, 299/my_dpi), dpi=my_dpi)
        
        plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        
        plt.xlim(0, 1000)
        plt.ylim(0, 50)
        
        # x,y 축 눈금 라벨 제거
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
        
        
        #plt.show()
        plt.savefig(scalo + '/' + MM + "_lead2.png") 
        plt.close()



        
