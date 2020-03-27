
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
'''

import numpy as np 
import tkinter as tk
import scipy as sc
from scipy.stats import zscore, f

# Functions
def regress(Y, X):
    '''    
    Parameters
    ----------
    Y : floating point array (observations-by-outcomes)
        outcome variables
    X : floating pint array (observation-by-predictors)
        predictors

    Returns
    -------
    floating point array (predictors-by-outcomes)
        beta coefficients
    '''
    
    return np.matmul(
            np.matmul(
            sc.linalg.inv(
            np.matmul(X.transpose(), X)), 
            X.transpose()), Y)

def correct_autocorr(X, W):
    '''
    Parameters
    ----------
    X : floating point array (2D)
        timecourses
    W : floating point array (1D)
        AR(2) model weights

    Returns
    -------
    X_corrected : floating point array (2D)
        timecourses corrected for autocorrelation
    '''
    
    rows, cols = X.shape
    X_corrected = np.zeros((rows, cols))
    for j in range(cols):
        x_shift_0 = X[:, j].reshape(rows,-1)
        x_shift_1 = np.append(0,X[0:-1, j]).reshape(rows, -1)
        x_shift_2 = np.append(np.zeros(2), X[0:-1, j]).reshape(rows, -1)
        x_sliced = np.hstack([x_shift_0, x_shift_1, x_shift_2])
        X_corrected[:, j] = np.matmul(x_sliced, W)
    
    return X_corrected
        
# Classes
class PEA:
    '''
    Phase-encoding analysis tool.

    pea = PEA(parameters) creates an instance of the PEA class.
    parameters is a dictionary with 7 required keys
    - f_sampling: sampling frequency (1 / TR)
    - f_stim    : stimulation frequency
    - n_samples : number of samples (volumes)
    - n_rows    : number of rows (in-plane resolution)
    - n_cols    : number of columns (in-plance resolution)
    - n_slices  : number of slices

    This class has the following function
    - PEA.set_delay(delay)
    - PEA.set_direction(direction)
    - results = PEA.fitting(data)

    typical workflow:
    1. pea = PEA(parameters)
    2. pea.set_delay(delay)
    3. pea.set_direction(direction)
    4. results = pea.fitting(data)
    '''
    
    def __init__(self, parameters):
        self.f_sampling = parameters['f_sampling']
        self.f_stim = parameters['f_stim']
        self.p_sampling = 1 / self.f_sampling
        self.n_samples = parameters['n_samples']
        self.n_rows = parameters['n_rows']
        self.n_cols = parameters['n_cols']
        self.n_slices = parameters['n_slices']
        self.n_total = self.n_rows * self.n_cols * self.n_slices
        self.time = np.arange(0, self.p_sampling * self.n_samples - 1, self.p_sampling) 
        self.delay = 0.0
        self.direction = 1
   

    def set_delay(self, delay):
        '''
        Parameters
        ----------
        delay : float
            delay in traveing wave caused by hemodynamic response
        '''
        self.delay = delay
        
    def set_direction(self, direction):
        '''
        provide a direction of motion for the class.
        This can be either numeric (-1,1) or in form of a string ('cw','ccw')
        for clockwise and counterclockwise rotation, respectively.
        Contracting rings are also considered to move clockwise (-1)
        while expanding rings are considered to move counterclockwise (1).
        '''
            
        if direction=='cw':
                self.direction = -1
        elif direction=='ccw':
                self.direction = 1
        elif (type(direction)==int) | (type(direction)==float):
            self.direction = int(direction)


    def fitting(self, data):
        '''
        identifies phase and amplitude at stimulation frequency for
        each voxel and returns a dictionary with the following keys
         - phase
         - amplitude
         - f_statistic
         - p_value

        The dimension of each field corresponds to the dimensions of the data.

        required inputs are
         - data  : a matrix of empirically observed BOLD timecourses
                   whose rows correspond to time (volumes).
        '''

        print('performing phase encoding analysis\n')

        
        F = np.exp(self.direction * 2j * np.pi * self.f_stim * (self.time-self.delay))
        X = zscore(np.array([np.real(F), np.imag(F)]).transpose())

        data = zscore(np.reshape(
                        data[0:self.n_samples,:,:,:].astype(float), 
                        self.n_samples,
                        self.n_total), 
                        axis = 0)
        beta = regress(data, X)
        Y_ = np.matmul(X, beta)
        residuals = data - Y_
        
        std_signal = np.std(data, axis = 0)
        results = {'phase': np.zeros(self.n_total),
                   'amplitude': np.zeros(self.n_total),
                   'f_stat': np.zeros(self.n_total),
                   'p_value': np.zeros(self.n_total)}
            
        df1 = 1.0
        df2 = self.n_samples-2.0

        for v in range(self.n_total):
            if (std_signal[v]>0):
                
                    
                # estimate and correct for autocorrelation
                T = np.array([np.append(0,residuals[0:-1, v]), 
                              np.append(np.zeros(2), residuals[0:-2, v])]).transpose()
                W = np.append(1, regress(residuals[:, v], T))
                
                X_corrected = correct_autocorr(X, W)
                D_corrected = correct_autocorr(data, W)
                #---------------------------------------------------------------------
                    
                beta_corrected = regress(D_corrected, X_corrected)
                Y_ = np.matmul(X_corrected, beta_corrected)
                mu = np.mean(D_corrected, axis = 0);
                MSM = np.dot(Y_ - mu, Y_ - mu) / df1
                MSE = np.dot(Y_ - D_corrected, Y_ - D_corrected) / df2
                beta_complex = beta_corrected[0] + beta_corrected[1] * 1j 
                results['phase'][v] = np.angle(beta_complex)
                results['amplitude'][v] = np.abs(beta_complex)
                results['f_stat'][v] = MSM / MSE;
                results['p_value'][v] = max(1-f.cdf(MSM / MSE, df1, df2), 1e-20)

            
        results['phase'] = np.reshape(results['phase'],
                                      self.n_rows,
                                      self.n_cols,
                                      self.n_slices)
        results['amplitude'] = np.reshape(results['amplitude'],
                                          self.n_rows,
                                          self.n_cols,
                                          self.n_slices)
        results['f_stat'] = np.reshape(results['f_stat'],
                                       self.n_rows,
                                       self.n_cols,
                                       self.n_slices)
        results['p_value'] = np.reshape(results['p_value'],
                                        self.n_rows,
                                        self.n_cols,
                                        self.n_slices)
            

        