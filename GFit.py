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

    def get_peaks_dips_default(self, number_peaks, window_size=None, smoothing=False, show_smoothing=True,
                               stepsize=0.01, max_iter=1000):
        """
        This method aims to find a wanted number of peaks and automatically finds peaks for further computations

        :param number_peaks: int; The number of peaks to find
        :param window_size: int; A paramater used for the smoothing of the data
        :param smoothing: boolean; Parameter if smoothing should be applied or not
        :param show_smoothing: boolean; Parameter if the smoothed data should be shown
        :param stepsize: float; How much the prominence should be increased/decreased after one iteration
        :param max_iter: int; Stopping condition for the algorithm, since multiple peaks can have the same prominence and
        therefore the algorithm sometimes cannot find the wanted number of parameters.
        :return: returns the found (x,y) data-points for the peaks and dips
        """
        #sort data in x
        p = 0.5

        if smoothing:
            y_data = self.smoothing(window_size=window_size)
            if show_smoothing:
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
            p += diff * stepsize # change the prominence into the "right" direction
            pks, pks_dict = find_peaks(y_data, prominence=p)

            n = len(pks)
            counter += 1

        print(f"Peakfinder Message: {n}/{number_peaks} peaks found")
        return [self.x[pks], y_data[pks]]

    def smoothing(self, window_size):
        """
        This method smooths data using np.convolve. It is useful if the data is ver noisy. In otherwords a moving average
        is calculated to smooth the data.
        :param window_size: int; parameter for the window size of the convolution
        :return: smoothed data
        """
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
