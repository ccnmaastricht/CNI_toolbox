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

import sys
import numpy as np
from scipy.linalg import inv
from scipy.stats import zscore
from scipy.fft import fft, ifft
from cni_toolbox.gadgets import two_gamma, regress

class RRT:
    ''' 
    Ridge regression tool.
    
    rrt = RRT(parameters) creates an instance of the RRT class.
    parameters is a dictionary with 5 required keys
      - f_sampling: sampling frequency (1/TR)
      - n_samples : number of samples (volumes)
      - n_rows    : number of rows (in-plane resolution)
      - n_cols    : number of columns (in-plance resolution)
      - n_slices  : number of slices
    
    optional inputs are
      - hrf       : either a column vector containing a single hemodynamic
                    response used for every voxel;
                    or a matrix with a unique hemodynamic response along
                    its columns for each voxel.
                    By default the canonical two-gamma hemodynamic response
                    function is generated internally based on the scan parameters.
    
    This class has the following functions
    %
      - hrf = RRT.get_hrf()
      - X = RRT.get_design()
      - RRT.set_hrf(hrf)
      - RRT.set_design(X)
      - RRT.optimize_lambda(data,range)
      - results = RRT.perform_ridge(data)
    
    Typical workflow:
    1. rrt = RRT(params);
    2. rrt.set_design(X);
    3. rrt.optimize_lambda(data,range);
    4. results = rrt.perform_ridge(data);
    '''
    
    def __init__(self, parameters, hrf = None):
        self.f_sampling = parameters['f_sampling']
        self.p_sampling = 1 / self.f_sampling
        self.n_samples = parameters['n_samples']
        self.n_rows = parameters['n_rows']
        self.n_cols = parameters['n_cols']
        self.n_slices = parameters['n_slices']
        self.n_total = self.n_rows * self.n_cols * self.n_slices
        
        if hrf != None:
            self.l_hrf = hrf.shape[0]
            if hrf.ndim>2:
                hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
                self.hrf_fft = fft(np.vstack((hrf,
                                      np.zeros((self.n_samples,
                                                self.n_total)))),
                                   axis = 0)
            else:
                self.hrf_fft = fft(np.append(hrf, 
                                     np.zeros(self.n_samples)),
                                   axis = 0 )
        else:
            self.l_hrf = int(32 * self.f_sampling)
            timepoints = np.arange(0, 
                                   self.p_sampling * (self.n_samples +
                                                      self.l_hrf) - 1, 
                                   self.p_sampling)
            self.hrf_fft = fft(two_gamma(timepoints), axis = 0)
        
    
    def get_hrf(self):
        '''
        Returns
        -------
        hrf : floating point array
            hemodynamic response function(s) used by the class
        '''
        if self.hrf_fft.ndim>1:
                hrf = ifft(np.zqueeze(
                    np.reshape(self.hrf,
                                 (self.l_hrf,
                                  self.n_rows,
                                  self.n_cols,
                                  self.n_slices))),
                    axis = 0)[0:self.l_hrf, :]
        else:
            hrf = ifft(self.hrf_fft, axis = 0)[0:self.l_hrf]
        
        return hrf
    
    
    def get_design(self):
        '''
        Returns
        -------
        floating point array
            design matrix used by the class

        '''
        
        return self.X[0:self.n_samples, :]
    
    def set_hrf(self, hrf):
        '''
        Parameters
        ----------
        hrf : floating point array
            hemodynamic response function
        '''
        self.l_hrf = hrf.shape[0]
        if hrf.ndim>2:
            hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
            self.hrf_fft = fft(np.vstack((hrf,
                                     np.zeros((self.n_samples,
                                                self.n_total)))),
                               axis = 0)
        else:
            self.hrf_fft = fft(np.append(hrf, 
                                     np.zeros(self.n_samples)),
                               axis = 0)
            
    def set_design(self, X, convolve = False):
        '''
        provide a n-by-p design matrix to the class with n samples
        and p predictors.
        
        optional inputs is
         - convolve: boolean
             indicates whether the design matrix needs to be convolved 
             with the hemodynamic response function (default = False)
        '''
        
        if X.ndim>1:
            self.n_predictors = X.shape[1]
        else:
            self.n_predictors = 1
        
        if convolve:
            X_fft = fft(np.vstack((X,
                                  np.zeros((self.l_hrf, self.n_predictors)))),
                        axis = 0)
            X = np.abs(ifft(X_fft *
                             np.expand_dims(self.hrf_fft,
                                            axis = 1), axis = 0))

        self.X = zscore(X[0:self.n_samples,], axis = 0)
        
    def optimize_penalty(self, data, candidates, folds = 4,
                        mask = None):
        '''
        performs k-fold cross-validation to find an optimal value
        for the penalty parameter.
 
        required inputs are
         - data: floating point array
            empirically observed BOLD timecourses
            whose rows correspond to time (volumes).
         - candidates: foating point array
             candidate values for the penalty parameter

        optional inputs are
         - folds: integer
             number of folds (k)
         - mask: boolean array 
             binary mask for selecting voxels (default = None)

        '''
        
        if mask.all()==None:
            mask = np.ones(self.n_total).astype(bool)
        else:
            mask = np.reshape(mask,self.n_total)
        
        data = np.reshape(data.astype(float), 
                        (self.n_samples,
                        self.n_total))
        data = zscore(data[:, mask], axis = 0)
        
        fit = np.zeros(candidates.size)
        
        s_sample = self.n_samples // folds
        
        for i in range(candidates.size):
            for k in range(folds):
                
                n_sub = k * s_sample + s_sample
                trn_X = self.X[:n_sub, :]
                trn_data = data[:n_sub, :]
                
                tst_X = self.X[n_sub::, :]
                tst_data = data[n_sub::, :]
                
                beta = regress(trn_data, trn_X, candidates[i])
                
                Y = np.matmul(tst_X, beta)
                
                mag_d = np.sqrt(np.sum(tst_data**2, axis = 0))
                mag_y = np.sqrt(np.sum(Y**2, axis = 0))
                
                corr_fit = np.mean(np.sum(Y * tst_data, axis=0) / (mag_y * mag_d))
                fit[i] += (corr_fit - fit[i]) / (k + 1)
                                
        idx = np.argmax(fit)
        
        self.penalty = candidates[idx]
                

    def perform_ridge(self, data, mask = None, penalty = None):
        '''
        performs ridge regression and returns a dictionary 
        with the following keys
         - corr_fit:
         - beta: 

        The dimension of corr_fit corresponds to the dimensions of
        the data. beta has an additional dimension with num_predictors
        elements.
            %
        required input is
         - data : floating point array
            empirically observed BOLD timecourses
            whose rows correspond to time (volumes).

        optional inputs are
         - lambda: float
             penalty parameter
        - mask: boolean array 
             binary mask for selecting voxels (default = None)
        '''
        
        results = {'corr_fit': np.zeros(self.n_total),
                   'beta': np.zeros((self.n_samples, self.n_predictors))}
        if penalty==None:
            penalty = self.penalty
        
        if mask.all()==None:
            mask = np.ones(self.n_total).astype(bool)
        else:
            mask = np.reshape(mask,self.n_total)
        
        data = np.reshape(data.astype(float), 
                        (self.n_samples,
                        self.n_total))
        data = zscore(data[:, mask], axis = 0)
        
        mask = np.reshape(mask,self.n_total)
        
        beta = regress(data, self.X, penalty)
        Y = np.matmul(self.X, beta)
                
        mag_d = np.sqrt(np.sum(data**2, axis = 0))
        mag_y = np.sqrt(np.sum(Y**2, axis = 0))
                
        corr_fit = np.sum(Y * data, axis=0) / (mag_y * mag_d)
        
        results['corr_fit'][mask] = corr_fit
        results['beta'][mask, :] = beta        
        
        results['corr_fit'] = np.squeeze(
            np.reshape(
                results['corr_fit'],
                (self.n_rows,
                 self.n_cols,
                 self.n_slices)))
        
        results['beta'] = np.squeeze(
            np.reshape(results['beta'],
                       (self.n_rows,
                        self.n_cols,
                        self.n_slices,
                        self.n_predictors)))
        
        return results
   