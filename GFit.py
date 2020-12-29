import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from scipy.signal import find_peaks, find_peaks_cwt
import matplotlib.pyplot as plt
import warnings


class GFit():
    def __init__(self, x, y, x_err=None, y_err=None):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

    def get_peaks_dips(self, number_peaks, stepsize=0.01, max_iter=1000):
        #sort data in x
        p = 0.5
        pks, pks_dict = find_peaks(self.y, prominence=p)
        n = len(pks)
        counter = 1

        it_counter = 0
        while n != number_peaks:
            it_counter += 1
            diff = n - number_peaks
            if it_counter == max_iter:
                message = "Due to same prominences peaks could not be resolved, peakfind still executed"
                warnings.warn(message, category=Warning, stacklevel=2)
                break
            p += diff * stepsize
            pks, pks_dict = find_peaks(self.y, prominence=p)

            n = len(pks)
            counter += 1
        print("#####")
        print(f"{n}/{number_peaks} peaks found")
        print("#####")
        return [self.x[pks], self.y[pks]]

    def get_guess(self):
        pass

    def fit(self):
        pass

    def plot_save(self, xlabel, ylabel, title, savename):
        pass

from noisy_data_create import noisy_gaussian
x = np.linspace(-2,6,10000)
y = noisy_gaussian(x, 2, 1)
test = GFit(x,y)
plt.plot(x, y, 'x')
xpks, ypks = test.get_peaks_dips(1)
plt.plot(xpks, ypks, 'x')
plt.show()