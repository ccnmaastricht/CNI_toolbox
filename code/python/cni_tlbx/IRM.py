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
from scipy.stats import zscore
from scipy.fft import fft, ifft
from cni_tlbx.gadgets import two_gamma

class IRM:
    '''
    Input-referred model (IRM) mapping tool.

    irm = IRM(parameters) creates an instance of the IRM class.
    parameters is a dictionary with 5 required keys
      - f_sampling: sampling frequency (1/TR)
      - n_samples : number of samples (volumes)
      - n_rows    : number of rows (in-plane resolution)
      - n_cols    : number of columns (in-plance resolution)
      - n_slices  : number of slices

    optional inputs are
      - hrf       : either a column vector containing a single hemodynamic
                    response used for every voxel;
                    or a tensor with a unique hemodynamic response along
                    its columns for each voxel.
                    By default the canonical two-gamma hemodynamic response
                    function is generated internally based on the scan parameters.

    This class has the following functions

      - hrf = IRM.get_hrf()
      - stimulus = IRM.get_stimulus()
      - tc = IRM.get_timecourses()
      - IRM.set_hrf(hrf)
      - IRM.set_stimulus(stimulus)
      - IRM.create_timecourses()
      - results = IRM.mapping(data)


    Typical workflow:
    1. irm = IRM(params)
    2. irm.set_stimulus()
    3. irm.create_timecourse(FUN,xdata)
    4. results = irm.mapping(data)
    '''

    def __init__(self, parameters, hrf = None):
        self.f_sampling = parameters['f_sampling']
        self.p_sampling = 1 / self.f_sampling
        self.n_samples = parameters['n_samples']
        self.n_rows = parameters['n_rows']
        self.n_cols = parameters['n_cols']
        self.n_slices = parameters['n_slices']
        self.n_total = self.n_rows * self.n_cols * self.n_slices

        if hrf is not None:
            self.l_hrf = hrf.shape[0]
            if hrf.ndim > 1:
                hrf = np.reshape(hrf, (self.l_hrf, self.n_total))
                self.hrf_fft = fft(np.vstack((hrf,
                                              np.zeros((self.n_samples,
                                                        self.n_total)))),
                                   axis = 0)
            else:
                self.hrf_fft = fft(np.append(hrf,
                                             np.zeros(self.n_samples)),
                                   axis = 0)
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
        if self.hrf_fft.ndim>1:
                hrf = ifft(np.zqueeze(
                    np.reshape(self.hrf[0:self.l_hrf, :],
                                 (self.l_hrf,
                                  self.n_rows,
                                  self.n_cols,
                                  self.n_slices))),
                    axis = 0)
        else:
            hrf = ifft(self.hrf_fft, axis = 0)[0:self.l_hrf]

        return np.abs(hrf)

    def get_stimulus(self):
        '''
        Returns
        -------
        floating point array (1D)
            stimulus used by the class
        '''

        return self.stimulus

    def get_timecourses(self):
        '''
        Returns
        -------
        floating point array (time-by-grid size)
            predicted timecourses
        '''

        return np.abs(ifft(self.tc_fft, axis = 0)[0:self.n_samples, :])

    def set_hrf(self, hrf):
        '''
        Parameters
        ----------
        hrf: floating point array
            hemodynamic response function
        '''
        self.l_hrf = hrf.shape[0]
        if hrf.ndim>1:
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
        '''
        Parameters
        ----------
        stimulus: floating point array (1D)
            stimulus used by the class
        '''
        self.stimulus = stimulus

    def create_timecourses(self, FUN, xdata):
        '''
        creates predicted timecourses based on the stimulus protocol
        and a range of parameters for an input referred model.

        Required inputs are
        - FUN: function handle
            defining the input referred model
        - xdata: dictionary with p elements (p = number of parameters).
            Each element needs to contain a column vector of variable
            length with parameter values to be explored.
        '''
        print('\ncreating timecourses')

        self.xdata = xdata
        self.n_predictors = len(xdata)

        n_observations = np.zeros(self.n_predictors)
        for idx, key in enumerate(self.xdata):
            n_observations[idx] = np.size(self.xdata[key])
        self.n_points = np.prod(n_observations).astype(int)

        idx_all = np.arange(self.n_points)
        self.idx = np.zeros((self.n_points, self.n_predictors))
        for p in range(self.n_predictors):
            self.idx[:, p] = (idx_all // (np.prod(n_observations[p+1::]))) \
                % n_observations[p]
        self.idx = self.idx.astype(int)

        tc = np.zeros((self.n_samples + self.l_hrf,
                       self.n_points))

        x = np.zeros(self.n_predictors)

        for j in range(self.n_points):
            for p, key in enumerate(self.xdata):
                x[p] = self.xdata[key][self.idx[j, p]]
            tc[0:self.n_samples, j] = FUN(self.stimulus, x)
            i = int(j / self.n_points * 21)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%"
                   % ('='*i, 5*i))

        sdev_tc = np.std(tc, axis = 0)
        idx_remove = np.where(sdev_tc == 0)
        tc = np.delete(tc, idx_remove, axis = 1)
        self.idx = np.delete(self.idx, idx_remove, axis = 0)
        self.tc_fft = fft(tc, axis = 0)

    def mapping(self, data, threshold = 100, mask = []):
        '''
        identifies the best fitting timecourse for each voxel and
        returns a dictionary with keys corresponding to the
        parameters specified in xdata plus a key 'corr_fit'
        storing the fitness of the solution.

        Required inputs are
        - data: floating point array
            empirically observed BOLD timecourses
            whose rows correspond to time (volumes).

        Optional inputs are
         - threshold: float
             minimum voxel intensity          (default = 100.0)
         - mask: boolean array
             binary mask for selecting voxels (default = []])
        '''
        print('\nmapping model parameters')

        data = np.reshape(data.astype(float),
                        (self.n_samples,
                        self.n_total))

        mean_signal = np.mean(data, axis = 0)
        sdev_signal = np.std(data, axis = 0)

        if np.size(mask)==0:
            mask = mean_signal >= threshold

        mask = np.reshape(mask,self.n_total)
        mask = mask.astype(bool) & (sdev_signal > 0)

        data = zscore(data[:, mask], axis = 0)

        voxel_index = np.where(mask)[0]
        n_voxels = voxel_index.size

        mag_d = np.sqrt(np.sum(data**2, axis = 0))

        results = {'corr_fit': np.zeros(self.n_total)}
        for key in self.xdata:
            results[key] = np.zeros(self.n_total)

        if self.hrf_fft.ndim==1:
            tc = np.transpose(
                zscore(
                    np.abs(
                        ifft(self.tc_fft *
                             np.expand_dims(self.hrf_fft,
                                            axis = 1), axis = 0)), axis = 0))
            tc = tc[:, 0:self.n_samples]
            mag_tc = np.sqrt(np.sum(tc**2, axis = 1))

            for m in range(n_voxels):
                v = voxel_index[m]

                CS = np.matmul(tc, data[:, m]) / (mag_tc * mag_d[m])
                idx_remove = (np.isinf(CS))| (np.isnan(CS))
                CS[idx_remove] = 0

                results['corr_fit'][v] = np.max(CS)
                idx_best = np.argmax(CS)

                for pos, key in enumerate(self.xdata):
                    results[key][v] = self.xdata[key][self.idx[idx_best, pos]]

                i = int(m / n_voxels * 21)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%"
                    % ('='*i, 5*i))

        else:
            for m in range(n_voxels):
                v = voxel_index[m]

                tc = np.transpose(
                    zscore(
                        np.abs(
                            ifft(self.tc_fft *
                                 np.expand_dims(self.hrf_fft[:, v],
                                                axis = 1), axis = 0)), axis = 0))

                tc = tc[:, 0:self.n_samples]
                mag_tc = np.sqrt(np.sum(tc**2, axis = 1))

                CS = np.matmul(tc, data[:, v]) / (mag_tc * mag_d[m])
                idx_remove = (CS == np.Inf)| (CS == np.NaN);
                CS[idx_remove] = 0

                results['corr_fit'][v] = np.max(CS)
                idx_best = np.argmax(CS)

                for pos, key in enumerate(self.xdata):
                    results[key][v] = self.xdata[key][self.idx[idx_best, pos]]

                i = int(m / n_voxels * 21)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%"
                    % ('='*i, 5*i))


        for key in results:
            results[key] = np.squeeze(
            np.reshape(results[key],
                       (self.n_rows,
                        self.n_cols,
                        self.n_slices)))

        return results
