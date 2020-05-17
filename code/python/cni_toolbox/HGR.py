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
from scipy.stats import zscore
from scipy.fft import fft, ifft
from cni_toolbox.gadgets import two_gamma, regress

class HGR:
    '''
    Hashed Gaussian Regression (HGR) tool.

    hgr = HGR(parameters) creates an instance of the HGR class.

    parameters is a structure with 7 required fields
      - f_sampling : sampling frequency (1/TR)
      - r_stimulus : width & height of stimulus images in pixels
      - n_features : number of features (hashed Gaussians)
      - n_gaussians: number of Gaussians per feature
      - n_voxels   : total number of voxels in data
      - fwhm       : full width at half maximum of Gaussians
      - eta        : learning rate (inverse of regularization parameter)

    optional inputs are
      - l_kernel   : length of convolution kernel (two-gamma hresults)

    this class has the following functions

      - [mask, corr_fit] = HGR.get_best_voxels(data, stimulus);
      - gamma = HGR.get_features()
      - results = HGR.get_parameters()
      - theta = HGR.get_weights()
      - tc = HGR.get_timecourses()
      - HGR.set_parameters(parameters)
      - HGR.reset();
      - HGR.ridge(data, stimulus)
      - HGR.update(data, stimulus)

    use help HGR.function to get more detailed help on any specific function
    (e.g. help HGR.ridge)

    typical offline workflow:
    1. hgr = HGR(parameters);
    2. hgr.ridge(data, stimulus);
    3. hgr.get_parameters();
    '''

    def __init__(self, parameters, l_kernel = 34):
        self.p_sampling = 1 / parameters['f_sampling']
        self.r_stimulus = parameters['r_stimulus']
        self.n_pixels = self['r_stimulus']**2
        self.n_features = parameters['n_features']
        self.n_gaussians = parameters['n_gaussians']
        self.n_voxels = parameters['n_voxels']
        self.fwhm = parameters['fwhm'] * self.r_stimulus
        self.eta = parameters['eta'] / self.n_features
        self.lambda = 1 / self.eta;

        self.theta = np.zeros((self.n_features, self.n_voxels))
        self.step = 1;
        self.mean = np.zeros(self.n_features)
        self.previous_mean = np.zeros(self.n_features)
        self.M2 = np.ones(self.n_features)
        self.sigma = np.zeros(self.n_features)

        timepoints = np.arange(0, self.l_kernel, self.p_sampling)
        self.kernel = two_gamma(timepoints)

        self.__create_gamma__();

    def update(data, stimulus):
        '''

        '''
        phi = np.matmul(stimulus, self.gamma)
        phi_conv = self.__convolution_step__(phi)
        self.phi = self.__zscore_step__(phi_conv)
        self.theta = self.theta + self.eta * \
            (np.matmul(self.phi.transpose(), data) -
            np.matmul(
            np.matmul(self.phi.transpose(), self.phi),
            self.theta))

    def ridge(data, stimulus):
        '''

        '''
        self.phi = zscore(
            self.convolution(
            np.matmul(stimulus,self.gamma)))
        self.theta = regress(data, self.phi, l = self.lambda)
