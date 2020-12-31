import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(-2,6,10000)

#get fake data for sum of gaussians
#takes list of amplitudes "amp", centers "mu" and widths "sig"
def noisy_gaussian(x, amp, mu, sig):
    y=0
    for a, m, s in zip(amp, mu, sig):
        y += a*np.exp(-np.power(x - m, 2.) / (2 * np.power(s, 2.)))

    #finally add noise
    y += np.random.normal(0, 0.2, len(x))
    return y

#y = noisy_gaussian(x, 2, 1)
#plt.plot(x, y, 'x')
#plt.show()