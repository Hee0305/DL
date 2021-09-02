from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = pd.read_csv("./ecg/AF/MUSE_20180111_155542_84000.csv",header=None)
print(x)
x2 = x.iloc[:,0]
print(x2)

#x3 = np.values(x2)
#print(type(x3))

widths = np.arange(0, 100) 

cwtmatr = signal.cwt(x2, signal.cascade, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()