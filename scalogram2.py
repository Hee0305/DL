from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#x = pd.read_csv("./ecg/AF/MUSE_20180111_155542_84000.csv",header=None)
x = pd.read_csv("./ecg/AF/MUSE_20180112_121939_99000.csv",header=None)
print(x)
x2 = x.iloc[:,0]
print(x2)

#x3 = np.values(x2)
#print(type(x3))


plt.plot(x2)
#plt.xlim(0, 1000)
#plt.ylim(0, 50)
#plt.figure()


sig = x2
wavelet = signal.ricker

widths = np.arange(1,1000)
cwtmatr = signal.cwt(sig, wavelet, widths)

plt.imshow(cwtmatr, origin='lower',cmap='PiYG', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max()) # extent 대신 xlim, ylim 사용하면 됨


cc = plt.colorbar(label='color')
plt.clim(-1000, 1000)
plt.xlim(0, 1000)
plt.ylim(0, 50)
plt.show()
