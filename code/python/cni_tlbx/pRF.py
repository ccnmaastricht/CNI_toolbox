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
import cv2
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.stats import zscore
from scipy.fft import fft, ifft
from cni_tlbx.gadgets import two_gamma, gaussian, stimpad


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
      - h_stimulus: height of stimulus in pixels
      - w_stimulus: width of stimulus in pixels

    Optional inputs are
      - hrf       : either a vector containing a single hemodynamic response used for every voxel;
                    or a tensor with a unique hemodynamic response along its columns for each voxel.
                    By default the canonical two-gamma hemodynamic response function is used.

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

    def __init__(self, parameters, hrf=None):
        self.f_sampling = parameters['f_sampling']
        self.p_sampling = 1 / self.f_sampling
        self.n_samples = parameters['n_samples']
        self.n_rows = parameters['n_rows']
        self.n_cols = parameters['n_cols']
        self.n_slices = parameters['n_slices']
        self.n_total = self.n_rows * self.n_cols * self.n_slices
        self.h_stimulus = parameters['h_stimulus']
        self.w_stimulus = parameters['w_stimulus']
        self.r_stimulus = np.maximum(self.h_stimulus, self.w_stimulus)

        if hrf is not None:
            self.l_hrf = hrf.shape[0]
            if hrf.ndim > 1:
                hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
                self.hrf_fft = fft(np.vstack((hrf,
                                              np.zeros((self.n_samples,
                                                        self.n_total)))),
                                   axis=0)
            else:
                self.hrf_fft = fft(np.append(hrf,
                                             np.zeros(self.n_samples)),
                                   axis=0)
        else:
            self.l_hrf = int(34 * self.f_sampling)
            timepoints = np.arange(0,
                                   self.p_sampling * (self.n_samples +
                                                      self.l_hrf),
                                   self.p_sampling)
            self.hrf_fft = fft(two_gamma(timepoints), axis=0)

    def get_hrf(self):
        '''
        Returns
        -------
        hrf: floating point array
            hemodynamic response function(s) used by the class
        '''
        if self.hrf_fft.ndim > 1:
            hrf = ifft(np.zqueeze(
                np.reshape(self.hrf[0:self.l_hrf, :],
                           (self.l_hrf,
                            self.n_rows,
                            self.n_cols,
                            self.n_slices))),
                       axis=0)
        else:
            hrf = ifft(self.hrf_fft, axis=0)[0:self.l_hrf]

        return np.abs(hrf)

    def get_stimulus(self):
        '''
        Returns
        -------
        floating point array (height-width-by-time)
            stimulus used by the class
        '''

        stimulus = np.reshape(self.stimulus[:, 0:self.n_samples],
                              (self.r_stimulus,
                               self.r_stimulus,
                               self.n_samples))

        return stimulus

    def get_timecourses(self):
        '''
        Returns
        -------
        floating point array (time-by-grid size)
            predicted timecourses
        '''

        return np.abs(ifft(self.tc_fft, axis=0)[0:self.n_samples, :])

    def set_hrf(self, hrf):
        '''
        Parameters
        ----------
        hrf : floating point array
            hemodynamic response function
        '''
        self.l_hrf = hrf.shape[0]
        if hrf.ndim > 1:
            hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
            self.hrf_fft = fft(np.vstack((hrf,
                                          np.zeros((self.n_samples,
                                                    self.n_total)))),
                               axis=0)
        else:
            self.hrf_fft = fft(np.append(hrf,
                                         np.zeros(self.n_samples)),
                               axis=0)

    def set_stimulus(self, stimulus):
        '''
        Parameters
        ----------
        stimulus : floating point array
        '''
        if stimulus.ndim < 3:
            stimulus = np.reshape(stimulus,
                (self.h_stimulus, self.w_stimulus, self.n_samples))

        stimulus = stimpad(stimulus)
        self.stimulus = np.reshape(
            stimulus, (self.r_stimulus**2, self.n_samples))
        self.stimulus = np.hstack((self.stimulus,
                                   np.zeros((self.r_stimulus**2, self.l_hrf))))

    def import_stimulus(self):
        root = tk.Tk()
        stimulus_directory = ''.join(filedialog.askdirectory(
            title='Please select the stimulus directory'))
        root.destroy()

        files = glob.glob('%s/*.png' % stimulus_directory)

        stimulus = np.zeros((self.h_stimulus,
                                  self.w_stimulus,
                                  self.n_samples))

        for f in files:
            number = int(''.join([str(s) for s in f if s.isdigit()]))
            img = cv2.imread(f)
            stimulus[:, :, number] = img[:, :, 0]

        mn = np.min(stimulus)
        mx = np.max(stimulus)
        stimulus = (stimulus - mn) / (mx - mn)
        stimulus = stimpad(stimulus)

        self.stimulus = np.reshape(stimulus,
                                   (self.r_stimulus**2,
                                    self.n_samples))

        self.stimulus = np.hstack((self.stimulus,
                                   np.zeros((self.r_stimulus**2, self.l_hrf))))

    def create_timecourses(self,use_slope=True, max_radius=10.0, num_xy=30,
                           min_slope=0.1, max_slope=1.2,
                           min_sigma=0.1, max_sigma=1,
                           num_size=10, css_exponent=1,
                           sampling='log'):
        '''
        creates predicted timecourses based on the effective stimulus and candidate receptive fields.
        Candidate receptive fields are generated for a grid of location (x,y) and size parameters.

        optional inputs are
         - use_slope: boolean
            use size-eccentricity slope?    (default = False)
         - max_radius: float
             radius of the field of fiew     (default = 10.0)
         - num_xy: integer
             steps in x and y direction      (default = 30)
         - min_slope: float
             lower bound of RF size slope    (default = 0.1)
         - max_slope: float
             upper bound of RF size slope    (default = 1.2)
         - num_size: integer
             steps from lower to upper bound (default = 10)
         - css_exponent: float
             compressive spatial summation   (default = 1)
         - sampling: string
             eccentricity sampling
             > 'log '(default)
             > 'linear'
        '''
        self.use_slope = use_slope
        if use_slope:
            self._create_timecourses_slope(max_radius, num_xy, min_slope,
                                           max_slope, num_size, css_exponent,
                                           sampling)
        else:
            self._create_timecourses_sigma(max_radius, num_xy, min_sigma,
                                           max_sigma, num_size, css_exponent,
                                           sampling)


    def mapping(self, data, threshold=100, mask=[]):
        '''
        identifies the best fitting timecourse for each voxel and
        returns a dictionary with the following keys
         - corr_fit
         - mu_x
         - mu_y
         - sigma
         - eccentricity
         - polar_angle

        The dimension of each field corresponds to the dimensions
        of the data.

        Required inputs are
        - data : floating point array
            empirically observed BOLD timecourses
            whose rows correspond to time (volumes).


        Optional inputs are
         - threshold: float
             minimum voxel intensity          (default = 100.0)
         - mask: boolean array
             binary mask for selecting voxels (default = [])
        '''
        if self.use_slope:
            return self._mapping_slope(data, threshold, mask)
        else:
            return self._mapping_sigma(data, threshold, mask)


    def get_timecourses(self):
        '''
        Returns
        -------
        floating point array (time-by-grid size)
            predicted timecourses
        '''

        return np.abs(ifft(self.tc_fft, axis=0)[0:self.n_samples, :])

    def set_hrf(self, hrf):
        '''
        Parameters
        ----------
        hrf : floating point array
            hemodynamic response function
        '''
        self.l_hrf = hrf.shape[0]
        if hrf.ndim > 1:
            hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
            self.hrf_fft = fft(np.vstack((hrf,
                                          np.zeros((self.n_samples,
                                                    self.n_total)))),
                               axis=0)
        else:
            self.hrf_fft = fft(np.append(hrf,
                                         np.zeros(self.n_samples)),
                               axis=0)

    def set_stimulus(self, stimulus):
        '''
        Parameters
        ----------
        stimulus : floating point array
        '''
        if stimulus.ndim < 3:
            stimulus = np.reshape(stimulus,
                (self.h_stimulus, self.w_stimulus, self.n_samples))

        stimulus = stimpad(stimulus)
        self.stimulus = np.reshape(
            stimulus, (self.r_stimulus**2, self.n_samples))
        self.stimulus = np.hstack((self.stimulus,
                                   np.zeros((self.r_stimulus**2, self.l_hrf))))

    def import_stimulus(self):
        root = tk.Tk()
        stimulus_directory = ''.join(filedialog.askdirectory(
            title='Please select the stimulus directory'))
        root.destroy()

        files = glob.glob('%s/*.png' % stimulus_directory)

        stimulus = np.zeros((self.h_stimulus,
                                  self.w_stimulus,
                                  self.n_samples))

        for f in files:
            number = int(''.join([str(s) for s in f if s.isdigit()]))
            img = cv2.imread(f)
            stimulus[:, :, number] = img[:, :, 0]

        mn = np.min(stimulus)
        mx = np.max(stimulus)
        stimulus = (stimulus - mn) / (mx - mn)
        stimulus = stimpad(stimulus)

        self.stimulus = np.reshape(stimulus,
                                   (self.r_stimulus**2,
                                    self.n_samples))

        self.stimulus = np.hstack((self.stimulus,
                                   np.zeros((self.r_stimulus**2, self.l_hrf))))

    def _create_timecourses_slope(self, max_radius, num_xy,
                           min_slope, max_slope,
                           num_size, css_exponent,
                           sampling):

        print('\ncreating timecourses')

        self.n_points = (num_xy**2) * num_size

        r_ones = np.ones(self.r_stimulus)

        h_values = np.linspace(max_radius, -max_radius, self.r_stimulus)
        w_values = np.linspace(-max_radius, max_radius, self.r_stimulus)
        x_coordinates = np.outer(r_ones, w_values)
        y_coordinates = np.outer(h_values, r_ones)

        x_coordinates = np.reshape(x_coordinates, self.r_stimulus**2)
        y_coordinates = np.reshape(y_coordinates, self.r_stimulus**2)

        idx_all = np.arange(self.n_points)
        self.idx = np.array([idx_all // (num_size * num_xy),
                             (idx_all // num_size) % num_xy,
                             idx_all % num_size]).transpose()

        n_lower = int(np.ceil(num_xy / 2))
        n_upper = int(np.floor(num_xy / 2))
        if sampling == 'log':
            self.ecc = np.exp(
                np.linspace(
                    np.log(0.1),
                    np.log(max_radius),
                    num_xy))
        elif sampling == 'linear':
            self.ecc = np.linspace(0.1, max_radius, num_xy)

        self.pa = np.linspace(0, (num_xy - 1) / num_xy * 2 * np.pi, num_xy)
        self.slope = np.linspace(min_slope, max_slope, num_size)

        W = np.zeros((self.n_points,
                      self.r_stimulus**2))

        for p in range(self.n_points):
            x = np.cos(self.pa[self.idx[p, 0]]) * self.ecc[self.idx[p, 1]]
            y = np.sin(self.pa[self.idx[p, 0]]) * self.ecc[self.idx[p, 1]]
            sigma = self.ecc[self.idx[p, 1]] * self.slope[self.idx[p, 2]]
            W[p, :] = gaussian(x, y, sigma, x_coordinates, y_coordinates)

            i = int(p / self.n_points * 19)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%"
                             % ('=' * i, 5 * i))

        tc = (np.matmul(W, self.stimulus).transpose())**css_exponent
        sdev_tc = np.std(tc, axis = 0)
        idx_remove = np.where(sdev_tc == 0)
        tc = np.delete(tc, idx_remove, axis = 1)
        self.idx = np.delete(self.idx, idx_remove, axis = 0)

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
        self.tc_fft = fft(tc, axis=0)


    def _create_timecourses_sigma(self, max_radius, num_xy,
                           min_sigma, max_sigma,
                           num_size, css_exponent,
                           sampling):

        print('\ncreating timecourses')

        self.n_points = (num_xy**2) * num_size

        r_ones = np.ones(self.r_stimulus)

        h_values = np.linspace(max_radius, -max_radius, self.r_stimulus)
        w_values = np.linspace(-max_radius, max_radius, self.r_stimulus)
        x_coordinates = np.outer(r_ones, w_values)
        y_coordinates = np.outer(h_values, r_ones)

        x_coordinates = np.reshape(x_coordinates, self.r_stimulus**2)
        y_coordinates = np.reshape(y_coordinates, self.r_stimulus**2)

        idx_all = np.arange(self.n_points)
        self.idx = np.array([idx_all // (num_size * num_xy),
                             (idx_all // num_size) % num_xy,
                             idx_all % num_size]).transpose()

        n_lower = int(np.ceil(num_xy / 2))
        n_upper = int(np.floor(num_xy / 2))
        if sampling == 'log':
            self.ecc = np.exp(
                np.linspace(
                    np.log(0.1),
                    np.log(max_radius),
                    num_xy))
        elif sampling == 'linear':
            self.ecc = np.linspace(0.1, max_radius, num_xy)

        self.pa = np.linspace(0, (num_xy - 1) / num_xy * 2 * np.pi, num_xy)
        self.sigma = np.linspace(min_sigma, max_sigma * max_radius, num_size)

        W = np.zeros((self.n_points,
                      self.r_stimulus**2))

        for p in range(self.n_points):
            x = np.cos(self.pa[self.idx[p, 0]]) * self.ecc[self.idx[p, 1]]
            y = np.sin(self.pa[self.idx[p, 0]]) * self.ecc[self.idx[p, 1]]
            W[p, :] = gaussian(x, y, self.sigma[self.idx[p, 2]],
                               x_coordinates, y_coordinates)

            i = int(p / self.n_points * 19)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%"
                             % ('=' * i, 5 * i))

        tc = (np.matmul(W, self.stimulus).transpose())**css_exponent
        sdev_tc = np.std(tc, axis = 0)
        idx_remove = np.where(sdev_tc == 0)
        tc = np.delete(tc, idx_remove, axis = 1)
        self.idx = np.delete(self.idx, idx_remove, axis = 0)

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
        self.tc_fft = fft(tc, axis=0)

    def _mapping_slope(self, data, threshold, mask):

     print('\nmapping receptive fields')

     data = np.reshape(data.astype(float),
                       (self.n_samples,
                        self.n_total))

     mean_signal = np.mean(data, axis = 0)
     sdev_signal = np.std(data, axis = 0)


     if np.size(mask) == 0:
         mask = mean_signal >= threshold

     mask = np.reshape(mask, self.n_total)
     mask = mask.astype(bool) & (sdev_signal > 0)

     data = zscore(data[:, mask], axis=0)

     voxel_index = np.where(mask)[0]
     n_voxels = voxel_index.size

     mag_d = np.sqrt(np.sum(data**2, axis=0))

     results = {'corr_fit': np.zeros(self.n_total),
                'mu_x': np.zeros(self.n_total),
                'mu_y': np.zeros(self.n_total),
                'sigma': np.zeros(self.n_total)}

     if self.hrf_fft.ndim == 1:
         tc = np.transpose(
             zscore(
                 np.abs(
                     ifft(self.tc_fft *
                          np.expand_dims(self.hrf_fft,
                                         axis=1), axis=0)), axis=0))
         tc = tc[:, 0:self.n_samples]
         mag_tc = np.sqrt(np.sum(tc**2, axis=1))
         for m in range(n_voxels):
             v = voxel_index[m]

             CS = np.matmul(tc, data[:, m]) / (mag_tc * mag_d[m])
             idx_remove = idx_remove = (np.isinf(CS))| (np.isnan(CS))
             CS[idx_remove] = 0

             results['corr_fit'][v] = np.max(CS)
             idx_best = np.argmax(CS)

             results['mu_x'][v] = np.cos(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['mu_y'][v] = np.sin(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['sigma'][v] = self.ecc[self.idx[idx_best, 1]] * \
                 self.slope[self.idx[idx_best, 2]]

             i = int(m / n_voxels * 21)
             sys.stdout.write('\r')
             sys.stdout.write("[%-20s] %d%%"
                              % ('=' * i, 5 * i))

     else:
         for m in range(n_voxels):
             v = voxel_index[m]

             tc = np.transpose(
                 zscore(
                     np.abs(
                         ifft(self.tc_fft *
                              np.expand_dims(self.hrf_fft[:, v],
                                             axis=1), axis=0)), axis=0))

             tc = tc[:, 0:self.n_samples]
             mag_tc = np.sqrt(np.sum(tc**2, axis=1))

             CS = np.matmul(tc, data[:, m]) / (mag_tc * mag_d[m])
             idx_remove = (CS == np.Inf) | (CS == np.NaN)
             CS[idx_remove] = 0

             results['corr_fit'][v] = np.max(CS)
             idx_best = np.argmax(CS)

             results['mu_x'][v] = np.cos(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['mu_y'][v] = np.sin(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['sigma'][v] = self.ecc[self.idx[idx_best, 1]] * \
                 self.slope[self.idx[idx_best, 2]]

             i = int(m / n_voxels * 21)
             sys.stdout.write('\r')
             sys.stdout.write("[%-20s] %d%%"
                              % ('=' * i, 5 * i))

     for key in results:
         results[key] = np.squeeze(
             np.reshape(results[key],
                        (self.n_rows,
                         self.n_cols,
                         self.n_slices)))

     results['eccentricity'] = np.abs(results['mu_x'] +
                                      results['mu_y'] * 1j)
     results['polar_angle'] = np.angle(results['mu_x'] +
                                       results['mu_y'] * 1j)

     return results


    def _mapping_sigma(self, data, threshold, mask):

     print('\nmapping receptive fields')

     data = np.reshape(data.astype(float),
                       (self.n_samples,
                        self.n_total))

     mean_signal = np.mean(data, axis = 0)
     sdev_signal = np.std(data, axis = 0)


     if np.size(mask) == 0:
         mask = mean_signal >= threshold

     mask = np.reshape(mask, self.n_total)
     mask = mask.astype(bool) & (sdev_signal > 0)

     data = zscore(data[:, mask], axis=0)

     voxel_index = np.where(mask)[0]
     n_voxels = voxel_index.size

     mag_d = np.sqrt(np.sum(data**2, axis=0))

     results = {'corr_fit': np.zeros(self.n_total),
                'mu_x': np.zeros(self.n_total),
                'mu_y': np.zeros(self.n_total),
                'sigma': np.zeros(self.n_total)}

     if self.hrf_fft.ndim == 1:
         tc = np.transpose(
             zscore(
                 np.abs(
                     ifft(self.tc_fft *
                          np.expand_dims(self.hrf_fft,
                                         axis=1), axis=0)), axis=0))
         tc = tc[:, 0:self.n_samples]
         mag_tc = np.sqrt(np.sum(tc**2, axis=1))
         for m in range(n_voxels):
             v = voxel_index[m]

             CS = np.matmul(tc, data[:, m]) / (mag_tc * mag_d[m])
             idx_remove = idx_remove = (np.isinf(CS))| (np.isnan(CS))
             CS[idx_remove] = 0

             results['corr_fit'][v] = np.max(CS)
             idx_best = np.argmax(CS)

             results['mu_x'][v] = np.cos(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['mu_y'][v] = np.sin(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['sigma'][v] = self.sigma[self.idx[idx_best, 2]]

             i = int(m / n_voxels * 21)
             sys.stdout.write('\r')
             sys.stdout.write("[%-20s] %d%%"
                              % ('=' * i, 5 * i))

     else:
         for m in range(n_voxels):
             v = voxel_index[m]

             tc = np.transpose(
                 zscore(
                     np.abs(
                         ifft(self.tc_fft *
                              np.expand_dims(self.hrf_fft[:, v],
                                             axis=1), axis=0)), axis=0))

             tc = tc[:, 0:self.n_samples]
             mag_tc = np.sqrt(np.sum(tc**2, axis=1))

             CS = np.matmul(tc, data[:, m]) / (mag_tc * mag_d[m])
             idx_remove = (CS == np.Inf) | (CS == np.NaN)
             CS[idx_remove] = 0

             results['corr_fit'][v] = np.max(CS)
             idx_best = np.argmax(CS)

             results['mu_x'][v] = np.cos(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['mu_y'][v] = np.sin(self.pa[self.idx[idx_best, 0]]) * \
                 self.ecc[self.idx[idx_best, 1]]
             results['sigma'][v] = self.sigma[self.idx[idx_best, 2]]

             i = int(m / n_voxels * 21)
             sys.stdout.write('\r')
             sys.stdout.write("[%-20s] %d%%"
                              % ('=' * i, 5 * i))

     for key in results:
         results[key] = np.squeeze(
             np.reshape(results[key],
                        (self.n_rows,
                         self.n_cols,
                         self.n_slices)))

     results['eccentricity'] = np.abs(results['mu_x'] +
                                      results['mu_y'] * 1j)
     results['polar_angle'] = np.angle(results['mu_x'] +
                                       results['mu_y'] * 1j)

     return results
