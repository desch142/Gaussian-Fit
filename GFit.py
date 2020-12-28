import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings


class GFit():
    def __init__(self, x, y, x_err, y_err):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

    def get_peaks_dips(self, number_peaks, stepsize=0.01):
        #sort data in x
        p = 0.5
        pks, pks_dict = find_peaks(self.y, prominence=p)
        n = len(pks)
        counter = 1
        diff = n - number_peaks
        diff_old=diff
        while n != number_peaks:
            n_old = n
            if n_old != n:
                diff_old = diff
            diff = n - number_peaks
            if diff_old == -diff:
                message = "Due to same prominences peaks could not be resolved, peakfind still executed"
                warnings.warn(message, category=Warning, stacklevel=2)
                break

            p += diff * stepsize
            pks, pks_dict = find_peaks(self.y, prominence=p)
            if diff == 1 or diff == -1:
                prominences = pks_dict['prominences']

            n = len(pks)
            print(n)
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





