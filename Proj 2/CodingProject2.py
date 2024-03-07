import numpy as np
import matplotlib.pyplot as plt

# Parameters
frequency = 1  # Hz
amplitude = 1
sampling_rate = 1000  # Hz
duration = 2  # seconds

# Time array
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Continuous sinusoidal signal
continuous_signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Sampling (digitization)
sample_indices = np.arange(0, len(t), sampling_rate // frequency)
sampled_values = continuous_signal[sample_indices]

# Plot continuous signal with samples
plt.figure(figsize=(10, 6))
plt.plot(t, continuous_signal, label='Continuous Signal')
plt.scatter(t[sample_indices], sampled_values, color='red', label='Samples')
plt.title('Continuous Sinusoidal Signal with Samples')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.show()

