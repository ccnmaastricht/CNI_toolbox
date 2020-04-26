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
from scipy.special import gamma


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
    X: floating point array (2D)
        timecourses
    W: floating point array (1D)
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
