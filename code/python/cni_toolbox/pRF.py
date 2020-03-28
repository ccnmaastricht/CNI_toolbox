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

import re
import sys
import cv2
import glob
import numpy as np 
import tkinter as tk
from tkinter import filedialog
from scipy.stats import zscore
from scipy.fft import fft, ifft
from cni_toolbox.auxiliary import two_gamma, gaussian

class pRF:
    '''
    Population receptive field (pRF) mapping tool.

    prf = pRF(params) creates an instance of the pRF class.
    Parameters is a dictionary with 7 required keys
      - f_sampling: sampling frequency (1/TR)
      - n_samples : number of samples (volumes)
      - n_rows    : number of rows (in-plane resolution)
      - n_cols    : number of columns (in-plance resolution)
      - n_slices  : number of slices
      - w_stimulus: width of stimulus images in pixels
      - h_stimulus: height of stimulus images in pixels

    Optional inputs are
      - hrf       : either a column vector containing a single hemodynamic
                    response used for every voxel;
                    or a matrix with a unique hemodynamic response along
                    its columns for each voxel.
                    By default the canonical two-gamma hemodynamic response
                    function is generated internally based on the scan parameters.

    This class has the following functions

      - hrf = pRF.get_hrf()
      - stimulus = pRF.get_stimulus()
      - tc = pRF.get_timecourses()
      - pRF.set_hrf(hrf)
      - pRF.set_stimulus(stimulus)
      - pRF.import_stimulus()
      - pRF.create_timecourses()
      - results = pRF.mapping(data)

    Typical workflow:
    1. prf = pRF(params)
    2. prf.import_stimulus()
    3. prf.create_timecourses()
    4. results = prf.mapping(data)
    ''' 
    
    def __init__(self, parameters, hrf = None):
        self.f_sampling = parameters['f_sampling']
        self.p_sampling = 1 / self.f_sampling
        self.n_samples = parameters['n_samples']
        self.n_rows = parameters['n_rows']
        self.n_cols = parameters['n_cols']
        self.n_slices = parameters['n_slices']
        self.n_total = self.n_rows * self.n_cols * self.n_slices
        self.w_stimulus = parameters['w_stimulus']
        self.h_stimulus = parameters['h_stimulus']
        
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
    
    def get_stimulus(self):
        '''
        Returns
        -------
        floating point array (height-width-by-time)
            stimulus used by the class
        '''

        stimulus = np.reshape(self.stimulus[:, 0:self.n_samples],
                          (self.h_stimulus,
                           self.w_stimulus,
                           self.n_samples))
        
        return stimulus
    
    def get_timecourses(self):
        '''
        Returns
        -------
        floating point array (time-by-gridsize)
            predicted timecourses
        '''
        
        return ifft(self.tc_fft, axis = 0)[0:self.n_samples, :]
    
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
            
    def set_stimulus(self, stimulus):
        
        if stimulus.ndim==3:
            self.stimulus = np.reshape(stimulus,
                    (self.w_stimulus * self.h_stimulus,
                    self.n_samples))
        else:
            self.stimulus = stimulus
            
    def import_stimulus(self):
        
        stimulus_directory = ''.join(tk.filedialog.askdirectory(
            title = 'Please select the stimulus directory'))
        root = tk.Tk()
        root.destroy()
        
        files = glob.glob('%s/*.png' % stimulus_directory)
        l = re.search(r"\d", files[0]).start()
        prefix = files[0][0:l]
        
        self.stimulus = np.zeros((self.h_stimulus,
                                  self.w_stimulus,
                                  self.n_samples + self.l_hrf))
        
        for idx, f in enumerate(files):
            number = int(''.join([str(s) for s in f if s.isdigit()]))
            img = cv2.imread(f)
            self.stimulus[:, :, number] = img[:, :, 0]
            
            i = int(idx / self.n_samples * 21)
            sys.stdout.write('\r')
            sys.stdout.write("loading stimulus [%-20s] %d%%" 
                             % ('='*i, 5*i))
            
        self.stimulus = np.reshape(self.stimulus,
                                   (self.h_stimulus * self.w_stimulus,
                                    self.n_samples + self.l_hrf))
                
