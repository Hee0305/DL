from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
my_dpi = 96

def scalogram(img):
    wavelet = signal.ricker                     
    widths = np.arange(1,1000)
    cwtmatr = signal.cwt(img, wavelet, widths)                
    plt.figure(figsize=(387/my_dpi, 397/my_dpi), dpi=my_dpi)                
    plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
               vmax=1000, vmin=-1000)            
    plt.ylim(0, 50)

def folder(i):    
    folder = ['AF', 'AFIB', 'AT', 'AVNRT', 'AVRT', 'SAAWR', 'ST', 'SVT', 'SB', 'SI', 'SR']
    #F = folder[i]
    FF = "./"+folder[i]
    #print(FF) 
    file = os.listdir(FF)
    #print("file: {}".format(file))
    return FF,file



for i in range(0,11): # AF ~ SR
    if (i==0) or (i==1):
        FF, file = folder(i)
        print(FF)
        #print(file)
    
    elif (i==2,3,4,5,6,7):
        FF, file = folder(i)
        print(FF)
    
    elif (i==8):
        FF, file = folder(i)
        print(FF)
    else:
        FF, file = folder(i)
        print(FF)
        
        
        
   
        
        
    
'''
    for j in range(len(file)): # MUSE 파일
        MFile = file[j]
        print(MFile)
        
        MName = MFile.split('.')
        MM = MName[0]
        
        df = pd.read_csv(FF + "/" + MFile)
        #print(df)
        
        # Patient scalogram 
        scalo = './image/scalogram/'+ F + '/' + MM
        os.mkdir(scalo)

        
    

        for k in range (0,12):
            img = df.iloc[:,k] 
            #print(img)
            scalogram(img)
          
            a = str(k+1)
            plt.savefig(scalo + '/' + "lead_" + a + ".png",
                        transparent=True,
                        #dpi=96, 
                        pad_inches=0,
                        bbox_inches="tight")  # 를 넣어버리면 299*299 가 깨짐
            
            
            plt.close('all')
'''


        
