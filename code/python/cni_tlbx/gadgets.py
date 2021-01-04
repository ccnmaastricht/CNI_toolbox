'''
-----------------------------------------------------------------------------
                                   LICENSE

Copyright 2020 Mario Senden

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------------
                                 DESCRPTION

This module contains functions either to be used in conjunction with
the core tools of the CNI toolbox; or to be used by the tools but
without being clearly assignable to one specific tool.

'''

import numpy as np
import scipy as sc
from scipy.fft import fft, ifft
from scipy.special import gamma


def gaussian(mu_x, mu_y, sigma, x, y):
    '''
    Parameters
    ----------
    mu_x: float
        center of Gaussian along x direction
    mu_y: float
        center of Gaussian along y direction
    sigma: float
        spread of Gaussian
    x: floating point array
        x-coordinates
    y: floating point array
        y-coordinates

    Returns
    -------
    floating point array
    '''

    return np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))

def two_gamma(timepoints):
    '''
    Parameters
    ----------
    timepoints : floating point array (1D)

    Returns
    -------
    hrf : floating point array
        hemodynamic response function
    '''

    hrf = (6 * timepoints**5 * np.exp(-timepoints)) / gamma(6) \
                - 1 / 6 * (16 * timepoints ** 15 * np.exp(-timepoints)) \
                    / gamma(16)
    return hrf

def size(x,length):
    '''
    Parameters
    ----------
    x : floating point array of unknown dimension
    length: integer

    Returns
    -------
    out : integer array with 'length' elements
        size in each of 'length' dimensions
    '''

    out = np.ones(length)
    dims = x.ndim
    out[:dims] = x.shape

    return out.astype(int)

def stimpad(stimulus):
    '''
    Parameters
    ----------
    stimulus: floating point array (height by width by samples)

    Returns
    -------
    padded_stimulus: floating point array (height by width by samples)
        zero padded stimulus
    '''

    height, width, samples = stimulus.shape
    res = np.maximum(width,height)

    padded_stimulus = np.zeros((res,res,samples))

    height_lower = int((res - height) / 2)
    height_upper = height_lower + height
    width_lower = int((res - width) / 2)
    width_upper = width_lower + width
    padded_stimulus[height_lower:height_upper,
        width_lower:width_upper,:] = stimulus

    return padded_stimulus

def regress(Y, X, l = 0.):
    '''
    Parameters
    ----------
    Y: floating point array (observations-by-outcomes)
        outcome variables
    X: floating pint array (observation-by-predictors)
        predictors
    l: float
        (optional) ridge penalty parameter

    Returns
    -------
    beta: floating point array (predictors-by-outcomes)
        beta coefficients
    '''

    if X.ndim>1:
        n_observations, n_predictors = X.shape

    else:
        n_observations = X.size
        n_predictors = 1


    if n_observations < n_predictors:
        U, D, V = np.linalg.svd(X, full_matrices = False)

        D = np.diag(D)
        beta = np.matmul(
            np.matmul(
                np.matmul(
                    np.matmul(
                        V.transpose(),
                        sc.linalg.inv(
                            D**2 +
                            l * np.eye(n_observations))),
                    D),
                U.transpose()), Y)
    else:
        beta = np.matmul(
            np.matmul(
            sc.linalg.inv(
            np.matmul(X.transpose(), X) +
            l * np.eye(n_predictors)),
            X.transpose()), Y)

    return beta

def correct_autocorr(X, W):
    '''
    Parameters
    ----------
    X: floating point array
        timecourses
    W: floating point array
        AR(2) model weights

    Returns
    -------
    X_corrected: floating point array (2D)
        timecourses corrected for autocorrelation
    '''

    rows, cols = size(X, 2)
    X = np.reshape(X, (rows, cols))
    X_corrected = np.zeros((rows, cols))
    for j in range(cols):
        x_shift_0 = X[:, j].reshape(rows,-1)
        x_shift_1 = np.append(0,X[0:-1, j]).reshape(rows, -1)
        x_shift_2 = np.append(np.zeros(2), X[0:-2, j]).reshape(rows, -1)
        x_sliced = np.hstack([x_shift_0, x_shift_1, x_shift_2])
        X_corrected[:, j] = np.matmul(x_sliced, W)

    return X_corrected

class online_processor:
    '''
    class for performing real time processing

    obj = online_processor(n_channels) creates an instance of the class
    n_channels determines the number of independent channels (voxels,
    pixels etc.) to which processing is applied

    optional inputs are
      - sampling_rate : float
            sampling rate of fMRI acquisition (TR) in seconds (default =  2)
      - l_kernel      : int
            length of hemodynic convolution kernel in seconds (default = 34)

    this class has the following functions

      - x_conv = online_processor.convolve(self, x)
        performs one step of real-time convolution
      - x_next = online_processor.update(self,x)
        performs one step of real-time z-transformation
      - online_processor.reset()
        resets all variables tracking the signal
    '''
    def __init__(self, n_channels, sampling_rate = 2., l_kernel = 34):
        self.n_channels = n_channels
        self.p_sampling = sampling_rate
        self.l_kernel = l_kernel

        self.step = 1
        self.mean = np.zeros(self.n_channels)
        self.previous_mean = np.zeros(self.n_channels)
        self.M2 = np.ones(self.n_channels)
        self.sigma = np.zeros(self.n_channels)

        self.l_subsampled = int(self.l_kernel / self.p_sampling)
        timepoints = np.arange(0., self.l_kernel, self.p_sampling)
        self.hrf_fft = fft(two_gamma(timepoints), axis=0)
        self.x_conv = np.zeros((self.l_subsampled, self.n_channels))

    def convolve(self, x):
        '''
        Parameters
        ----------
        x: floating point array
            current data sample

        Returns
        -------
        floating point array
            convolved sample
        '''

        x_fft = fft(np.vstack((x.reshape(1,-1), np.zeros((self.l_subsampled - 1, self.n_channels)))), axis=0)
        self.x_conv = np.vstack((self.x_conv, np.zeros((1, self.n_channels))))
        self.x_conv[self.step:self.step + self.l_subsampled, :] += np.abs(ifft(
            x_fft * np.expand_dims(self.hrf_fft , axis=1), axis=0))

        return self.x_conv[self.step, :]

    def update(self, x):
        '''
        Parameters
        ----------
        x: floating point array
            current data sample

        Returns
        -------
        floating point array
            z-normalized sample
        '''

        self.__update_mean__(x);
        self.__update_sigma__(x);
        self.step += 1

        return (x - self.mean) / self.sigma

    def reset(self):
        '''
        resets all variables tracking the signal
        '''

        self.step = 1
        self.mean = np.zeros(self.n_channels)
        self.previous_mean = np.zeros(self.n_channels)
        self.M2 = np.ones(self.n_channels)
        self.sigma = np.zeros(self.n_channels)

        self.l_subsampled = int(self.l_kernel / self.p_sampling) - 1
        timepoints = np.arange(0., self.l_kernel, self.p_sampling)
        self.hrf_fft = fft(two_gamma(timepoints), axis=0)
        self.x_conv = np.zeros((self.l_subsampled + 1, self.n_channels))

    def __update_mean__(self, x):
        '''
        updates the mean of the tracked variable
        '''

        self.previous_mean = self.mean
        self.mean += (x - self.mean) / self.step

    def __update_sigma__(self, x):
        '''
        updates the standard deviation of the tracked variable
        '''

        self.M2 += (x - self.previous_mean) * (x - self.mean)
        if self.step==1:
            self.sigma = np.sqrt(self.M2 / self.step)
        else:
            self.sigma = np.sqrt(self.M2 / (self.step - 1))
