# -*- coding: utf-8 -*-
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

'''  폴더만들기
path2_list = ['./AF_IMG','./SB_IMG','./SI_IMG','./AVNRT_IMG','./ST_IMG','./SAAWR_IMG','./AT_IMG','./SVT_IMG','./AFIB_IMG','./AVRT_IMG','./SR_IMG']
for path2 in path2_list:
    os.mkdir(path2) 
'''


folder = ['AF', 'SB', 'SI', 'AVNRT', 'ST', 'SAAWR', 'AT', 'SVT', 'AFIB', 'AVRT', 'SR']
folder2 = ['AF_IMG', 'SB_IMG', 'SI_IMG', 'AVNRT_IMG', 'ST_IMG', 'SAAWR_IMG', 'AT_IMG', 'SVT_IMG', 'AFIB_IMG', 'AVRT_IMG', 'SR_IMG']
my_dpi = 96
for i in range(0,1): # SB ~ SR 
    Fd = "./"+folder[i]
    #print(Fd) 
    Fd2 = "./"+folder2[i]
    file = os.listdir(Fd)
    #print("file: {}".format(file))
    
    for j in range(len(file)): # MUSE 파일
        MFile = file[j]
        print(MFile)
        
        df = pd.read_csv(Fd + "/" + MFile)
        #print(df)
        
       
        
        for k in range (0,12): # 12 lead
            lead=["L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12"]
            
            img = df.iloc[:,k]
            #print(img)
            
            print(lead[k])
            path = Fd + "/" + MFile + lead[k]
            path2 = Fd2 + "/" + MFile + lead[k]
            print(path2)
            
            #plt.plot(x2)
            
            wavelet = signal.ricker
                
            widths = np.arange(1,1000)
            cwtmatr = signal.cwt(img, wavelet, widths)
            
            
            plt.figure(figsize=(299/my_dpi, 299/my_dpi), dpi=my_dpi)
            
            plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
                       vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max()) 
            
            
            #cc = plt.colorbar()
            plt.clim(-1000, 1000)
            plt.xlim(0, 1000)
            plt.ylim(0, 50)
           
            '''
            # ------x,y 축 눈금 라벨 제거-----------
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            # -------------------------------------
            '''
            
            
            #plt.show()
            
            plt.savefig(path2+".png") 
            plt.close()
            
            print(MFile)
            
            
            
            
            
            

