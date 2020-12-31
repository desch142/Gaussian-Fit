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
                               stepsize=0.01, max_iter=1000, plotcheck=False):
        """
        This method aims to find a wanted number of peaks and automatically finds peaks for further computations

        :param number_peaks: int; The number of peaks to find
        :param window_size: int; A paramater used for the smoothing of the data
        :param smoothing: boolean; Parameter if smoothing should be applied or not
        :param show_smoothing: boolean; Parameter if the smoothed data should be shown
        :param stepsize: float; How much the prominence should be increased/decreased after one iteration
        :param max_iter: int; Stopping condition for the algorithm, since multiple peaks can have the same prominence and
        therefore the algorithm sometimes cannot find the wanted number of parameters.
        :param plotcheck: Plot data and found peaks+dips to check whether peak/dipfind worked correctly
        :return: returns the indices of the peaks and dips
        """


        if smoothing:
            y_data = self.smoothing(window_size=window_size)

        else:
            y_data = self.y

        #####Find peaks#####
        #Find number_peaks peaks by iteratively varying the prominence
        #and using scipy.signal.find_peaks(data, prominence)
        p = 0.5
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

        #x- and y-values of peaks for plotting
        x_pks, y_pks=self.x[pks], y_data[pks]

        #####Find dips#####
        #Find the number_peaks-1 dips by finding the minimum in between the peaks

        #list to store dips indices
        dps=[]

        for i in range(0, len(pks)-1):
            #find index of dip between pks[i] and pks[i+1] and append to dps list
            #add pks[i] since index of y_data_btw starts at zero
            dps.append(np.argmin(y_data[pks[i]:pks[i+1]])+pks[i])


        print(f"Dipfinder Message: {len(dps)}/{number_peaks-1} dips found")

        #x- and y-values of dips for plotting
        x_dps, y_dps=self.x[dps], y_data[dps]

        #optionally plot data, smoothed_data, peaks and dips
        if plotcheck:
            plt.plot(self.x, self.y, label="Data")
            if smoothing & show_smoothing:
                plt.plot(self.x, y_data)
            plt.plot(x_pks, y_pks,linestyle='', marker="^",markersize=12, label=f"{n}/{number_peaks} Peaks")
            plt.plot(x_dps, y_dps,linestyle='', marker="v",markersize=12, label=f"{len(dps)}/{number_peaks-1} Dips")
            plt.legend()

        #return indices of peaks and dips
        return [pks, dps]

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
#create random sum of 5 gaussians
x = np.linspace(0,10,10000)
y = noisy_gaussian(x, amp=1+np.random.rand(5)*10, mu=np.random.rand(5)*0.75+np.arange(1,9,8/5), sig=0.05+np.random.rand(5))*0.5
print(np.random.rand(5)*0.75+np.arange(1,10,9/5))


#test peak/dipfinder
test = GFit(x,y)

#find peaks+dips and plot
test.get_peaks_dips_default(5, smoothing=True, window_size=150, plotcheck=True)

plt.show()
