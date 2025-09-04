#compute the partial correlation using the inverse covariance approach.
#The partial correlation matrix is the negative of the off-diagonal elements of the inverse covariance,
#divided by the squared root of the corresponding diagonal elements.     
#https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
#in this approach, for two nodes X and Y, the partial correlation conditions on all the nodes except X and Y.

# Function adapted from Ruben Sanchez-Romero's CombinedFCToolBox 
# (https://github.com/ColeLab/CombinedFC/tree/master/CombinedFCToolBox) 
import numpy as np
from scipy import linalg
from sklearn.covariance import log_likelihood,empirical_covariance

def parCorrInvCov(data):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
    OUTPUT:
        M : matrix of partial correlation coefficients
    '''
    #compute the covariance matrix of the dataset and invert it. This is known as the precision matrix.
    #use the (Moore-Penrose) pseudo-inverse of a matrix.
    empCov = np.cov(data,rowvar=False)
    prec = linalg.pinv(empCov)
    #transform the precision matrix into partial correlation coefficients
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    parCorr = -prec * denom * denom.T
    #make the diagonal zero.
    np.fill_diagonal(parCorr,0)

    return parCorr,prec