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
from scipy.stats import zscore, f
from cni_tlbx.gadgets import regress, correct_autocorr

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

    This class has the following functions
    - PEA.set_delay(delay)
    - PEA.set_direction(direction)
    - results = PEA.fitting(data)

    Typical workflow:
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
        set delay caused by hemodynamic response

        Parameters
        ----------
        delay: float
            delay in traveing wave caused by hemodynamic response
        '''
        self.delay = delay

    def set_direction(self, direction):
        '''
        provide a direction of stimulus motion

        Parameters
        ----------
        direction: string or integer
            clockwise rotation / contraction = (-1, or 'cw')
            counterclockwise rotation / expansion = (1 or 'ccw')
        '''

        if direction=='cw':
                self.direction = -1
        elif direction=='ccw':
                self.direction = 1
        elif (type(direction)==int) | (type(direction)==float):
            self.direction = int(direction)


    def fitting(self, data, mask = [], threshold = 100):
        '''
        performs analysis; fitting sine and cosine at stimulation frequency to data

        Parameters
        ----------
        data: floating point array
            empirically observed BOLD timecourses
            whose rows correspond to time (volumes).

        Returns
        -------
        results : dictionary with the following keys
                - phase
                - amplitude
                - f_statistic
                - p_value
        '''
        print('\nperforming analysis')

        F = np.exp(self.direction * 2j * np.pi * self.f_stim * (self.time-self.delay))
        X = zscore(np.array([np.real(F), np.imag(F)]).transpose())

        data = np.reshape(
                        data.astype(float),
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

        for m in range(n_voxels):
            v = voxel_index[m]

            # estimate and correct for autocorrelation
            T = np.array([np.append(0,residuals[0:-1, m]),
                              np.append(np.zeros(2), residuals[0:-2, m])]).transpose()
            W = np.append(1, -regress(residuals[:, m], T))

            X_corrected = correct_autocorr(X, W)
            D_corrected = correct_autocorr(data[:,m], W)
            #---------------------------------------------------------------------

            beta_corrected = regress(D_corrected, X_corrected)
            Y_ = np.matmul(X_corrected, beta_corrected)
            mu = np.mean(D_corrected, axis = 0);
            MSM = np.dot((Y_ - mu).transpose(), Y_ - mu) / df1
            MSE = np.dot((Y_ - D_corrected).transpose(), Y_ - D_corrected) / df2
            beta_complex = beta_corrected[0] + beta_corrected[1] * 1j
            results['phase'][v] = np.angle(beta_complex)
            results['amplitude'][v] = np.abs(beta_complex)
            results['f_stat'][v] = MSM / MSE;
            results['p_value'][v] = max(1-f.cdf(MSM / MSE, df1, df2), 1e-20)


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
