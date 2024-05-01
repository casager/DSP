import numpy as np 
import matplotlib.pyplot as plt #not needed
from scipy.io.wavfile import write

import soundfile as sf

filename = 'demoSong.wav'


'''Read Song'''
songArray, Fs = sf.read(filename, dtype='float32')  

N = 4
print(Fs, N)

'''
Calculate Frequency Indices

This section has been removed...add it back in
'''


'''Prep Arrays'''
indices = (np.round(indices/Fs * 4096))
window = np.hanning(2048)
output = np.zeros([len(songArray),2],dtype = 'float32')
invert = np.zeros(4096, dtype = 'complex')
#prep an additional array for your spectrogram

'''
Calculate filtered song
'''
for j in range(int((len(songArray)-2048)/100)):
    idx = int(j*100)
    invert = np.zeros(4096,dtype ='complex')
    fDomain = songArray[idx:idx+2048,0] * window
    fDomain = np.fft.fft(fDomain, 4096)   
    for k in range(N):
        indexK = int(indices[k])
        indexNext = int(indices[k+1])
        Mag = np.mean(np.real(fDomain[indexK:indexNext]))
        phase = np.mean(np.imag(fDomain[indexK:indexNext]))
        invert[indexK:indexNext] = Mag +1j*phase
        invert[4096 - indexNext:4096-indexK] = Mag +1j*phase
    r = np.fft.ifft(invert)
    output[idx:idx+2048,0] = output[idx:idx+2048,0] + np.real(r[0:2048])
output[:,1] = output[:,0]

'''
Plot spectrogram

Add your code here
'''


'''
Write File
'''
print(output.shape, Fs)
plt.plot(output)
plt.show()
write('myProcessed_' + str(N) +'.wav', Fs, output)