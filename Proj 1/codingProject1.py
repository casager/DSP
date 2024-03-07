import numpy as np
import matplotlib.pyplot as plt
import math

N = 10
a = 0.95
simulationLength = 100

n = np.arange(simulationLength)
#h = np.ones(N)
#h = np.exp(range(N)) #exponential attempt for h
h = list(range(N)) #linear attempt for h w/ slope=1 (triangle)
plt.figure("h")
plt.stem(h)
y = np.zeros(simulationLength)

x = np.power(a, n)

plt.figure("Input")
plt.stem(x)
#plt.show() #comment out if want to see both input and output at same time

for i in range(len(y)):
    samplesToSum = N
    if i < N:
        samplesToSum = i
    for j in range(samplesToSum):
        y[i] = y[i] + x[i-j]*h[N-1-j]
plt.figure('Output')
plt.stem(y)
plt.show()
