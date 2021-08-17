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
from cni_tlbx.gadgets import two_gamma, gaussian, regress, online_processor

class HGR:
    '''
    Hashed Gaussian Regression (HGR) tool.

    hgr = HGR(parameters) creates an instance of the HGR class.

    parameters is a dictionary with 7 required keys
      - f_sampling : sampling frequency (1/TR)
      - r_stimulus : width & height of stimulus images in pixels
      - n_features : number of features (hashed Gaussians)
      - n_gaussians: number of Gaussians per feature
      - n_voxels   : total number of voxels in data
      - fwhm       : full width at half maximum of Gaussians
      - eta        : learning rate (inverse of regularization parameter)

    optional inputs are
      - l_kernel   : int
        length of convolution kernel (two-gamma hresults)

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
        self.n_pixels = self.r_stimulus**2
        self.n_features = parameters['n_features']
        self.n_gaussians = parameters['n_gaussians']
        self.n_voxels = parameters['n_voxels']
        self.l_hrf = l_kernel
        self.fwhm = parameters['fwhm'] * self.r_stimulus
        self.eta = parameters['eta'] / self.n_features
        self.regularization = 1 / self.eta

        self.theta = np.zeros((self.n_features, self.n_voxels))


        timepoints = np.arange(0, self.l_hrf, self.p_sampling)
        self.hrf = two_gamma(timepoints)

        self.data_processor = online_processor(self.n_voxels,
                                sampling_rate = self.p_sampling,
                                l_kernel = l_kernel)
        self.phi_processor = online_processor(self.n_features,
                                sampling_rate = self.p_sampling,
                                l_kernel = l_kernel)

        self.__create_gamma__();

    def update(self, data, stimulus):
        '''
        performs a single gradient descent update based on current
        time point's data and stimulus. Online convolution and
        z-normalization is handled internally.

        required inputs are
         - data: floating point array
            observed BOLD activation levels per voxel
         - stimulus: floating point array
            pixel intensities
        '''

        phi = np.matmul(stimulus, self.gamma)
        phi = self.phi_processor.convolve(phi)
        phi = self.phi_processor.update(phi).reshape(1,-1)
        y = self.data_processor.update(data).reshape(1,-1)
        self.theta += self.eta * (np.dot(
            phi.transpose(), y) -
            np.matmul(
            np.outer(phi, phi),
            self.theta))

    def ridge(self, data, stimulus):
        '''
        performs ridge regression with stimulus encoded by hashed
        Gaussians as predictors.

        required inputs are
        - data: floating pint 2D array (time-by-voxels)
            observed BOLD timecourses
        - stimulus: floating point 2D array (time-by-pixels)
            stimulus matrix
        '''

        phi = zscore(
            self.__convolution__(
            np.matmul(stimulus, self.gamma)))
        self.theta = regress(data, phi, l = self.regularization)

    def get_features(self):
        '''
        Returns
        -------
        floating point 2D array (pixels-by-features)
            hashed Gaussian features
        '''

        return self.gamma

    def get_weights(self):
        '''
        Returns
        -------
        loating point 2D array (features-by-voxels)
            learned regression weights
        '''

        return self.theta

    def get_parameters(self, n_batch = 10000,
        max_radius = 10, alpha = 1, mask = []):
        '''
        estimates population receptive field (2D Gaussian) parameters
        based on raw receptive fields given by features and their
        regression weights.

        returns a dictionary with the following keys
        - corr_fit
        - mu_x
        - mu_y
        - sigma
        - eccentricity
        - polar_angle

        Each key is a column vector with number of voxels elements

        optional inputs are
        - n_batch: float
            batch size                       (default = 10000)
        - max_radius: integer
            radius of measured visual field  (default =    10)
        - alpha: float
            shrinkage parameter              (default =     1)
        - mask: boolean array
            binary mask for selecting voxels (default =  None)
        '''

        print('\nestimating pRF parameters')

        if np.size(mask) == 0:
            mask = np.ones(self.n_voxels)

        mask = mask.astype(bool)
        idx = np.arange(self.n_voxels)
        idx = idx[mask]
        n_mask = sum(mask)

        results = {'mu_x': np.zeros(self.n_voxels),
                   'mu_y': np.zeros(self.n_voxels),
                   'sigma': np.zeros(self.n_voxels)}

        xy = np.linspace(-max_radius, max_radius, self.r_stimulus)
        [x_coordinates, y_coordinates] = np.meshgrid(xy,xy)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = -y_coordinates.flatten()

        s = np.linspace(1e-1, 1.5 * max_radius, 50)
        r = np.linspace(0, np.sqrt(2 * max_radius**2), 50)
        [S, R] = np.meshgrid(s,r)
        S = S.flatten()
        R = R.flatten()
        I = np.zeros(2500)

        for i in range(2500):
            x = np.cos(np.pi / 4) * R[i]
            y = np.sin(np.pi / 4) * R[i]
            W = gaussian(x, y, S[i], x_coordinates, y_coordinates)
            mx = np.max(W)
            mn = np.min(W)
            val_range = mx - mn
            W = (W - mn) / val_range
            I[i] = np.mean(W)

        P = np.hstack((I.reshape(-1,1), R.reshape(-1,1)))
        beta = regress(S, P)

        for v in np.arange(0, n_mask - n_batch, n_batch):
            i = int(v / n_mask * 21)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%"
                            % ('=' * i, 5 * i))

            batch = idx[v: v + n_batch]
            im = np.matmul(self.gamma, self.theta[:, batch])
            pos = np.argmax(im, axis = 0)
            mx = np.max(im, axis = 0)
            mn = np.min(im, axis = 0)
            val_range = mx - mn
            im = ((im - mn) / val_range)**alpha
            m_image = np.mean(im, axis = 0)

            cx = np.floor(pos / self.r_stimulus)
            cy = pos % self.r_stimulus
            results['mu_x'][batch] = cx / self.r_stimulus * max_radius * 2 - max_radius
            results['mu_y'][batch] = -cy / self.r_stimulus * max_radius * 2 - max_radius
            R = np.sqrt(results['mu_x'][batch]**2 + results['mu_y'][batch]**2)
            P = np.hstack((m_image.reshape(-1,1), R.reshape(-1,1)))
            results['sigma'][batch] = np.matmul(P, beta)

        exist = 'v' in locals()
        if exist==False:
            batch = idx
        else:
            batch = idx[v:]

        i = 20
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%"
                        % ('=' * i, 5 * i))

        im = np.matmul(self.gamma, self.theta[:, batch])
        pos = np.argmax(im, axis = 0)
        mx = np.max(im, axis = 0)
        mn = np.min(im, axis = 0)
        val_range = mx - mn
        im = ((im - mn) / val_range)**alpha
        m_image = np.mean(im, axis = 0).transpose()

        cx = np.floor(pos / self.r_stimulus)
        cy = pos % self.r_stimulus
        results['mu_x'][batch] = cx / self.r_stimulus * max_radius * 2 - max_radius
        results['mu_y'][batch] = -cy / self.r_stimulus * max_radius * 2 - max_radius
        R = np.sqrt(results['mu_x'][batch]**2 + results['mu_y'][batch]**2)
        P = np.hstack((m_image.reshape(-1,1), R.reshape(-1,1)))
        results['sigma'][batch] = np.matmul(P, beta)

        return results

    def get_timecourses(self, stimulus):
        '''
        Parameters
        ----------
        stimulus: floating point 2D array (time-by-pixels)

        Returns
        -------
        floating point 2D array (time-by-voxels)
            predicted timecourses
        '''

        phi = zscore(
            self.__convolution__(
            np.matmul(stimulus, self.gamma)))

        return np.matmul(phi, self.theta)

    def set_parameters(self, parameters):
        '''
        parameters is a dictionary with 7 required keys
          - f_sampling : sampling frequency (1/TR)
          - r_stimulus : width & height of stimulus images in pixels
          - n_features : number of features (hashed Gaussians)
          - n_gaussians: number of Gaussians per feature
          - n_voxels   : total number of voxels in data
          - fwhm       : full width at half maximum of Gaussians
          - eta        : learning rate (inverse of regularization parameter)
        '''

        self.p_sampling = 1 / parameters['f_sampling']
        self.r_stimulus = parameters['r_stimulus']
        self.n_pixels = self['r_stimulus']**2
        self.n_features = parameters['n_features']
        self.n_gaussians = parameters['n_gaussians']
        self.n_voxels = parameters['n_voxels']
        self.fwhm = parameters['fwhm'] * self.r_stimulus
        self.eta = parameters['eta'] / self.n_features
        self.regularization = 1 / self.eta;

        self.__create_gamma__()

    def reset(self):
        '''
        resets all internal states of the class

        use this function prior to conducting real time estimation
        for a new experimental run
        '''

        timepoints = np.arange(0, self.l_hrf, self.p_sampling)
        self.hrf = two_gamma(timepoints)
        self.data_processor.reset()
        self.phi_processor.reset()

    def get_best_voxels(self, data, stimulus,
                        type = 'percentile',
                        cutoff = 95.,
                        n_splits = 4):
        '''
        uses blocked cross-validation to obtain the best fitting voxels

        required inputs are
         - data: floating point 2D array
            empirically observed BOLD timecourses (time-by-voxels)
         - stimulus: floating point 2D array
            flattened stimulus matrix (time-by-pixels)

        optional inputs are
         - type: string
            cutoff type
            > 'percentile' (default)
            > 'threshold'
            > 'number'
         - cutoff: float
            cutoff value          (default = 95)
         - n_splits: int
            number of data splits (default =  4)

        Returns
        -------
        mask: boolean array
            mask indicating best voxels
        corr_fit: floating point array
            cross-validated correlation fit
        '''

        n_time = data.shape[0]
        n_steps = n_splits - 1
        n_samples = np.floor(n_time / n_splits)

        corr_fit = np.zeros((self.n_voxels, n_splits))

        for i in range(n_steps):
            bound = int((i + 1) * n_samples)
            train_data = zscore(data[:bound, :])
            train_stim = stimulus[:bound]
            test_data = zscore(data[bound:, :])
            test_stim = stimulus[bound:, :]

            self.ridge(train_data, train_stim)
            Y = self.get_timecourses(test_stim)
            mag_Y = np.sqrt(np.sum(Y**2, axis=0))
            mag_data = np.sqrt(np.sum(test_data**2, axis=0))
            corr_fit[:, i] = np.sum(Y * test_data, axis=0) / (mag_Y * mag_data)

        corr_fit = np.mean(corr_fit, axis=1)

        if type=='percentile':
            threshold = np.percentile(corr_fit, cutoff)
        elif type=='threshold':
            threshold = cutoff
        elif type=='number':
            corr_fit[np.isnan(corr_fit)] = -1
            corr_fit[corr_fit==np.Inf] = -1
            val = -np.sort(-corr_fit)
            threshold = val[cutoff]
        else:
            raise ValueError('Wrong type: Choose either ''percentile'', ''number'' or ''threshold''')

        mask = corr_fit>=threshold

        return mask, corr_fit


    def __create_gamma__(self):
        '''
        creates hashed Gaussian features
        '''

        r = np.arange(self.r_stimulus)
        [x_coordinates, y_coordinates] = np.meshgrid(r, r)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()
        sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.gamma = np.zeros((self.n_pixels, self.n_features))
        pix_id = np.linspace(0, self.n_pixels, self.n_features * self.n_gaussians)

        x = np.floor(pix_id / self.r_stimulus)
        y = pix_id % self.r_stimulus

        for i in range(self.n_features):
            for j in range(self.n_gaussians):
                self.gamma[:,i] += gaussian(x[i * self.n_gaussians + j],
                    y[i * self.n_gaussians + j], sigma, x_coordinates, y_coordinates)

            self.gamma[:, i] /= np.sum(self.gamma[:, i])


    def __convolution__(self, x):
        '''
        Parameters
        ----------
        x: floating point 2D array (time-by-channels)

        Returns
        -------
        floating point 2D array (time-channels)
            x convolved with hemodynamic response function
        '''

        n_samples = x.shape[0]
        kernel = np.append(self.hrf, np.zeros(n_samples))
        x = np.vstack((x, np.zeros((np.ceil(self.l_hrf / self.p_sampling).astype(int),
                                    self.n_features))))
        x_conv = np.abs(
                    ifft(fft(x, axis=0) *
                         np.expand_dims(fft(kernel),
                                        axis=1), axis=0))
        return x_conv[:n_samples]
