from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x2 = pd.read_csv("./ecg/AF/MUSE_20180111_155542_84000.csv",header=None)
print(x2)
print(type(x2))

x = x2.iloc[:,0]

fs = 500 # 샘플링 주파수

'''
f = 샘플 주파수 배열
t = 세그먼트 시간의 배열
Sxx = x의 스펙트로그램
'''



f, t, Sxx = signal.spectrogram(x, fs)

plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 50)
plt.show()