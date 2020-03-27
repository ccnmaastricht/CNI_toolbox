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
import scipy as sc


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