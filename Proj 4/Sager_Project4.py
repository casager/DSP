import numpy as np 
import matplotlib.pyplot as plt #not needed
from scipy.io.wavfile import write

import soundfile as sf

filename = 'c:/Users/Carson/Documents/DSP/Proj 4/LTTM.wav'


'''Read Song'''
songArray, Fs = sf.read(filename, dtype='float32')  
#print(songArray)

N = [8, 16, 32, 64]

print(Fs, N)

'''
Calculate Frequency Indices

This section has been removed...add it back in
'''
for num_channels in N:
    # Define the frequency range and number of channels
    low_freq = 100  # Hz
    high_freq = 4000  # Hz

    # Calculate the number of channels for each frequency range
    num_channels_low = round(0.6 * num_channels)  # ~60% of channels for 100 Hz - 1000 Hz
    num_channels_high = num_channels - num_channels_low  # Remaining channels for 1000 Hz - 4000 Hz

    # Calculate the bandwidths for each frequency range
    bandwidth_low = (1000 - low_freq) / num_channels_low
    bandwidth_high = (high_freq - 1000) / num_channels_high

    # Calculate the frequency indices for each channel in range 1 (100 Hz - 1000 Hz)
    indices_low = np.linspace(low_freq, 1000, num_channels_low + 1)
    indices_low = np.round(indices_low).astype(int)

    # Calculate the frequency indices for each channel in range 2 (1000 Hz - 4000 Hz)
    indices_high = np.linspace(1000, high_freq, num_channels_high + 1)
    indices_high = np.round(indices_high).astype(int)

    # Combine the frequency indices for both ranges
    indices = np.concatenate((indices_low[:-1], indices_high))
    #print(indices) #bandwidth between the 9 indices (for N=8)

    '''Prep Arrays'''
    indices = (np.round(indices/Fs * 4096))
    window = np.hanning(2048)
    output = np.zeros([len(songArray),2],dtype = 'float32')
    invert = np.zeros(4096, dtype = 'complex')
    #prep an additional array for your spectrogram

    # spectrogram_size = (512, len(songArray)//2048)  # Adjusted based on the length of the audio
    # spectrogram = np.zeros(spectrogram_size, dtype='float32')

    '''
    Calculate filtered song
    '''
    for j in range(int((len(songArray)-2048)/100)):
        idx = int(j*100)
        invert = np.zeros(4096,dtype ='complex')
        fDomain = songArray[idx:idx+2048,0] * window
        fDomain = np.fft.fft(fDomain, 4096)   
        for k in range(num_channels):
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
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.specgram(output[:, 0], NFFT=4096, Fs=Fs, noverlap=2048)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 7500) #added this
    plt.title(f'Spectrogram ({num_channels} Channels)')
    cbar = plt.colorbar()
    cbar.set_label('Magnitude (dB)')
    #plt.show()

    '''
    Write File
    '''
    plt.subplot(1, 2, 2)
    #print(output.shape, Fs)
    plt.plot(output)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(f'Time-domain waveform ({num_channels} Channels)')    
    
    plt.tight_layout()


    write('myProcessed_' + str(num_channels) +'.wav', Fs, output)
plt.show()    