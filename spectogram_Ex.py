from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dd = pd.read_csv("")

rng = np.random.default_rng()



fs = 10e3 # 샘플링 빈도 
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

'''
f = 샘플 주파수 배열
t = 세그먼트 시간의 배열
Sxx = x의 스펙트로그램
'''
# 단면 출력
f, t, Sxx = signal.spectrogram(x, fs) # x=측정값의 시계열 , fs = 샘플링 빈도 
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# 양면 출력
f, t, Sxx = signal.spectrogram(x, fs, return_onesided=False) # return_onesided=False -> 양면스펙트럼
plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()