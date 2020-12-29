import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt
import warnings


class GFit():
    def __init__(self, x, y, x_err=None, y_err=None):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

    def get_peaks_dips_default(self, number_peaks, window_size=None, smoothing=False, show_smoothing=True,stepsize=0.01, max_iter=1000):
        #sort data in x
        p = 0.5

        if smoothing == True:
            if window_size == None:
                window_size = int(len(self.x) * 0.2)
            y_data = self.smoothing(window_size=window_size)
            if show_smoothing == True:
                plt.plot(self.x, y_data)
        else:
            y_data = self.y


        pks, pks_dict = find_peaks(y_data, prominence=p)
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
            pks, pks_dict = find_peaks(y_data, prominence=p)

            n = len(pks)
            counter += 1

        print(f"Peakfinder Message: {n}/{number_peaks} peaks found")
        return [self.x[pks], y_data[pks]]

    def smoothing(self, window_size):
        y_data = np.convolve(self.y, np.ones((window_size,)) / window_size, mode='same')
        return y_data


    def get_guess(self):
        pass

    def fit(self):
        pass

    def plot_save(self, xlabel, ylabel, title, savename):
        pass

from noisy_data_create import noisy_gaussian
x = np.linspace(-2,6,10000)
y = noisy_gaussian(x, 2.1, 1)


plt.plot(x,y)
test = GFit(x,y)
xpks, ypks = test.get_peaks_dips_default(2, smoothing=True, window_size=400)
plt.plot(xpks, ypks, 'x')
plt.show()
