import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class GFit():
    def __init__(self, x, y, x_err, y_err):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err
        #self.peaks, self.dips = self.get_peaks_dips()

    def get_peaks_dips(self, height, prominence, distance, ):
        pass
        #return [peaks, dips]

    def get_guess(self):
        pass

    def fit(self):
        pass

    def plot_save(self, xlabel, ylabel, title, savename):
        pass





