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


for i in range(1,2): # AF ~ SVT
    F = folder[i]
    FF = "./"+folder[i]
    #print(FF) 
    file = os.listdir(FF)
    #print("file: {}".format(file))
    
    for j in range(len(file)): # MUSE 파일
        MFile = file[j]
        #print(MFile)
        
        MName = MFile.split('.')
        MM = MName[0]
        
        df = pd.read_csv(FF + "/" + MFile)
        #print(df)
        
        # Patient scalogram 
        scalo = './image/scalogram/'+ F + '/' + MM
        os.mkdir(scalo)
        
        
        
        '''
        # Patient spectrogram
        spectro = './image/spectrogram/'+ F + '/' + MM
        os.mkdir(spectro)
        '''
        for k in range (0,12):
            img = df.iloc[:,k] 
            #print(img)
            wavelet = signal.ricker
                            
            widths = np.arange(1,1000)
            cwtmatr = signal.cwt(img, wavelet, widths)
            
            
            fig = plt.figure(figsize=(387/my_dpi, 397/my_dpi), dpi=my_dpi)
            
            plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
                       vmax=1000, vmin=-1000)
            
            #plt.xlim(0, 5000)
            plt.ylim(0, 50)
            '''
            # x,y 축 눈금 라벨 제거
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            '''
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            ax = plt.gca()
            ax.set_axis_off()
            fig.add_axes(ax)
            
            
           
            
            a = str(k+1)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(scalo + '/' + "lead_" + a + ".png",
                        transparent=True,
                        #dpi=96, 
                        pad_inches=0,
                        bbox_inches="tight")  # 를 넣어버리면 299*299 가 깨짐
            
            
            plt.close('all')
    


        
