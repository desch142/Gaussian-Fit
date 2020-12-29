import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(-2,6,10000)
def noisy_gaussian(x, mu, sig):
    y = 2*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + 5*np.exp(-np.power(x - 5, 2.) / (2 * np.power(sig, 2.)))
    y += np.random.normal(0, 0.4, len(x))
    return y

#y = noisy_gaussian(x, 2, 1)
#plt.plot(x, y, 'x')
#plt.show()