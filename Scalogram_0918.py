from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
my_dpi = 96

def scalogram(img):
    wavelet = signal.ricker                     
    widths = np.arange(1,300)
    cwtmatr = signal.cwt(img, wavelet, widths)                
    plt.figure(figsize=(387/my_dpi, 397/my_dpi), dpi=my_dpi)                
    plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
               vmax=1000, vmin=-1000)          
    plt.axis("off")
    plt.ylim(0, 50)

def folder(i):    
    folder = ['AF', 'AFIB', 'AT', 'AVNRT', 'AVRT', 'SAAWR', 'ST', 'SVT', 'SB', 'SI', 'SR']
    F = folder[i]
    FF = "./"+folder[i]
    #print(FF) 
    file = os.listdir(FF)
    #print("file: {}".format(file))
    return F,FF,file



for i in range(0,11): # AF ~ SR
    F,FF, file = folder(i)
    if (i==0) or (i==1):
        dd = "AFIB"
        
    elif (i==2,3,4,5,6,7):  
        dd = "GSVT"
        
    elif (i==8):  
        dd = "SB"
                 
    else:
        dd = "SR"
    for j in range(len(file)): # MUSE 파일
        
        MFile = file[j]
        
        N = str(j+1)
        fileN = str(len(file))
        
        MName = MFile.split('.')
        MM = MName[0]
        
        df = pd.read_csv(FF + "/" + MFile)
        
        print(f"{dd} 파일 {fileN}개 중 {N}번째")
        
        for k in range (0,12):            
            path = "./DL/Scalogram"
            img = df.iloc[:,k]             
            scalogram(img)
          
            a = str(k+1)
            plt.savefig(f"{path}/lead{a}/{dd}/{MM}.png",
                        transparent=True,
                        #dpi=96, 
                        pad_inches=0,
                        bbox_inches="tight")
            
            
            plt.close('all')
        
    


        
