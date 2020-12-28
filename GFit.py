import numpy as np
from scipy.optimize import curve_fit
from scipy import odr


class GFit():
    def __init__(self, x, y, x_err, y_err):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

    def get_peaks_dips(self):
        pass

    def get_guess(self):
        pass

    def fit(self):
        pass

    def plot_save(self, xlabel, ylabel, title, savename):
        pass





