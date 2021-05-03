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
        self.fitparams = None

    def get_peaks_dips(self, number_peaks, window_size=None, smoothing=False, show_smoothing=False,
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
        # Find number_peaks peaks by iteratively varying the prominence
        # and using scipy.signal.find_peaks(data, prominence)
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
            p += diff * stepsize    # change the prominence into the "right" direction
            pks, pks_dict = find_peaks(y_data, prominence=p)

            n = len(pks)
            counter += 1

        print(f"Peakfinder Message: {n}/{number_peaks} peaks found")

        #x- and y-values of peaks for plotting
        x_pks, y_pks=self.x[pks], y_data[pks]

        #####Find dips#####
        # Find the number_peaks-1 dips by finding the minimum in between the peaks

        # list to store dips indices
        #dps=[]

        #for i in range(0, len(pks)-1):
            #find index of dip between pks[i] and pks[i+1] and append to dps list
            #add pks[i] since index of y_data_btw starts at zero
        #    dps.append(np.argmin(y_data[pks[i]:pks[i+1]])+pks[i])

        # proposal for the dip find
        dps = [np.argmin(y_data[d[0]:d[1]]) + pks[idx] for idx, d in enumerate(zip(pks[:-1], pks[1:]))]

        print(f"Dipfinder Message: {len(dps)}/{number_peaks-1} dips found \n")

        # x- and y-values of dips for plotting
        x_dps, y_dps = self.x[dps], y_data[dps]

        # optionally plot data, smoothed_data, peaks and dips
        if plotcheck:
            plt.plot(self.x, self.y,color='C0', label="Data")
            if smoothing & show_smoothing:
                plt.plot(self.x, y_data, label="Smoothed Data")
            plt.plot(x_pks, y_pks,linestyle='', marker="^",markersize=12, label=f"{n}/{number_peaks} Peaks")
            plt.plot(x_dps, y_dps,linestyle='', marker="v",markersize=12, label=f"{len(dps)}/{number_peaks-1} Dips")
            plt.legend()

        return [pks, dps]


    def smoothing(self, window_size):
        """
        This method smooths data using np.convolve. It is useful if the data is ver noisy. In other words a moving average
        is calculated to smooth the data.
        :param window_size: int; parameter for the window size of the convolution
        :return: smoothed data
        """
        y_data = np.convolve(self.y, np.ones((window_size,)) / window_size, mode='same')
        return y_data


    def get_guess(self, peaks_idx, dips_idx, plotcheck=False):
        """
        This method guesses the amplitudes, means and the variances for the gaussian fit
        :param peaks_idx: list int; a list/numpy array of indices where the peaks lie
        :param dips_idx: list int; a list/numpy array of indices where the infered dips lie
        :param plotcheck: boolean; if the 'dip' before the first peak and the 'dip' after last peak should be displayed
        for sanity checks
        :return: returns a list containing a single offset and a list of triples where each triple consists of (amp, mean, var) as a guess for the parameters for the fit
        """

        # First add 'dips' before the first peak and after the last peak
        # with this the method will calculate the variance of the data in between the dips.
        left = self.x[:peaks_idx[0]]
        right = self.x[peaks_idx[-1]:]
        len_left = len(left)
        len_right = len(right)

        diff_lft_peak_dip = dips_idx[0] - peaks_idx[0]

        diff_rght_peak_dip = peaks_idx[-1] - dips_idx[-1]

        if len_left < diff_lft_peak_dip:
            lft_idx = int(len_left / 2)
        else:
            lft_idx = diff_lft_peak_dip

        if len_right < diff_rght_peak_dip:
            rght_idx = int(len_right / 2) + peaks_idx[-1]
        else:
            rght_idx = diff_rght_peak_dip + peaks_idx[-1]

        # merge all the dips
        dips = [lft_idx] + dips_idx + [rght_idx]

        #guess for offset simply taking minimum value in data
        guess_offset = np.min(self.y)

        #guess for gaussians
        var = [np.var(self.x[d1:d2]) for d1, d2 in zip(dips[:-1], dips[1:])]
        amp= self.y[peaks_idx]-guess_offset
        mean = self.x[peaks_idx]

        guess_gaussians = [g for g in zip(amp, mean, var)]

        guess=np.append(guess_offset, guess_gaussians)
        print('Guess successfull \n')

        if plotcheck:
            plt.plot(self.x, self.y,color='C0')
            plt.plot(self.x[lft_idx], self.y[lft_idx], '^', markersize=12,label='left')
            plt.plot(self.x[rght_idx], self.y[rght_idx], '^', markersize=12, label='right')
            #plot guess
            x_g=np.linspace(self.x.min(), self.x.max(), 10**5)
            y_g=self.fitfunction(x_g, *guess)
            plt.plot(x_g, y_g, label='Initial guess', linestyle='--')
            plt.legend()

        return guess

    def fitfunction(self, x, *params):
        """
        This method returns the value of a sum of gaussians and is used for the gaussian fit
        :param x: float; the argument at which to evaluate the sum of gaussians
        :param *params: list float; a list containing a global offset and amplitude, mean and variance for each gaussian
        :return: returns the value y of the sum of gaussians at argument x
        """
        y=0
        #add offset
        y+=params[0]

        #add sum of gaussians
        for amp, mean, var in zip(params[1::3], params[2::3], params[3::3]):
            y+=amp*np.exp(-np.power(x - mean, 2.) / (2 * np.power(var, 2.)))

        return y

    def fit(self, guess, plotcheck=False, fitparamsname=None):
        """
        This method fits the data with a sum of gaussians and returns the fit parameters
        :param guess: list; list containing starting values for the fit of the gaussian amplitude (guess should be obtained using the method get_guess)
        :param plotcheck: boolean; plot of the fitted function for sanity checks
        :param fitparamsname: string; filename for txt file in which the fit parameters can be saved
        :return: returns the determined fit parameters and their errors as list
        """
        #Use standard Chi2 fit if there are no x-errors:
        if self.x_err is None:
            popt, pconv = curve_fit(self.fitfunction, self.x, self.y,sigma=self.y_err, p0=guess, maxfev=10**6)

            # extract fit parameters
            fit = popt  # value
            dfit = np.sqrt(pconv.diagonal())  # error
            parameters = np.column_stack((fit, dfit))
        else:
            #fit using ODR if there are x-errors
            #define new fit function since scipy odr requires a different format for the fit function
            def fitfunction_odr(p, x):
                return self.fitfunction(x, *p)
            func=odr.Model(fitfunction_odr)
            data=odr.RealData(self.x,self.y,sx=self.x_err, sy=self.y_err)
            odrfit=odr.ODR(data, func, beta0=guess, maxit=10**3)
            output=odrfit.run()
            fit=output.beta
            dfit=output.sd_beta
            parameters = np.column_stack((fit, dfit))

        #plot fit for sanity checks
        if plotcheck==True:
            plt.plot(self.x, self.y,color='C0')

            fitted_x = np.linspace(np.min(self.x), np.max(self.x), 10000)
            fitted_y = self.fitfunction(fitted_x, *fit)

            # plot fit
            plt.plot(fitted_x, fitted_y, linestyle='-', color="black", marker=' ', label='Fit')
            plt.legend()

        #print fitparameters and optionally save fitparameters as table in txt file

        table=np.empty((len(parameters)+1,3), dtype=object)
        table[0,0]=""
        table[0,1:3]=["Value", "Error"]
        table[1, 0] = "Offset"
        table[1:, 1:3] = parameters
        print('Fit successfull, the fit parameters are: \n')
        print('Offset:')
        print(table[1,1:3])
        for i in range(0,len(parameters)-1,3):
            print(f'Gaussian {int(i/3+1)} (Amplitude, Mean, SDV):')
            print(table[int(i+2):int(i+5),1:3])
            table[i+2,0]=f"Gaussian {int(i/3+1)}: Amplitude"
            table[i+3, 0] = f"Gaussian {int(i/3+1)}: Mean"
            table[i+4, 0] = f"Gaussian {int(i/3+1)}: SDV"

        if fitparamsname != None:
            np.savetxt(fitparamsname,table,delimiter="\t", fmt="%s")

        #return fitparameters for further use
        self.fitparams=parameters

        return parameters

    def plot_save(self, xlabel='x', ylabel='y', datalegend='Data', fitlegend='Fit', title='', savename='gaussian_fit.pdf', grid=True):
        """
        This method plots the data together with the fit with customizable labels, title and legend and saves plot as file
        :param xlabel: string; label of x-axis
        :param ylabel: string; label of y-axis
        :param datalegend: string; legend of the data points
        :param fitlegend: string; legend of the fit
        :param title: string; title of plot
        :param savename: string; filename to safe plot - we recommend to use .pdf to obtain a vector graphic
        :return: returns None
        """
        #Plot data with errorbars
        plt.figure(figsize=(5, 3))
        plt.errorbar(x=self.x,xerr=self.x_err, y=self.y,yerr=self.y_err,linestyle='none',marker='+', elinewidth=1, capsize=1.5, capthick=1, label=datalegend)

        fitted_x = np.linspace(np.min(self.x), np.max(self.x), 10000)
        fitted_y = self.fitfunction(fitted_x, *self.fitparams[:,0])

        plt.plot(fitted_x, fitted_y, linestyle='-', color='black', marker=' ', label=fitlegend, zorder=10)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(savename)
        plt.clf()
        print('\n Plot successfully saved')

        return None

test=False
if test==True:
    plt.clf()
    np.random.seed(1231)
    from noisy_data_create import noisy_gaussian
    #create random sum of 5 gaussians

    x = np.linspace(0,10,150)
    y = noisy_gaussian(x, amp=5+np.random.rand(4)*5, mu=np.random.rand(4)*1+np.linspace(1.5,8.5,4), sig=0.05+np.random.rand(4))*0.5
    xerr = np.random.rand(len(x))*0.3
    yerr = np.random.rand(len(x))*0.3
    sampledata=np.column_stack([x,xerr,y,yerr])
    #np.savetxt('example_data.txt', sampledata)


    #test peak/dipfinder
    test = GFit(x, y,  x_err=xerr, y_err=yerr)

    #find peaks+dips and plot
    pks, dps = test.get_peaks_dips(4, smoothing=False, window_size=150, plotcheck=True, show_smoothing=False)

    #get guess
    guess=test.get_guess(pks, dps, plotcheck=True)

    #do fit
    fitparams=test.fit(guess, plotcheck=True)
    plt.show()
    plt.clf()
    #save plot
    test.plot_save(title='Test fit', xlabel='x test', ylabel='y test')

