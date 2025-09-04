import numpy as np

def intraclassCorrelationCoefficient(data):
    
    nEdges = data.shape[0]
    nSubjs = data.shape[1]
    k = data.shape[2]
    
    targetMeans = np.mean(data,axis=2)
    totalMean = np.mean(data,axis=(1,2))
    
    betweenTargetsSumSquares = np.sum((targetMeans - np.repeat(totalMean[:,np.newaxis],nSubjs,axis=1))**2,axis=1) * k
    betweenTargetsMeanSquares = betweenTargetsSumSquares / (nSubjs-1)
    
    withinTargetsSumSquares = np.sum((data - np.repeat(targetMeans[:,:,np.newaxis],k,axis=2))**2,axis=(1,2))
    withinTargetsMeanSquares = withinTargetsSumSquares / (nSubjs*(k-1))
    
    ICC = (betweenTargetsMeanSquares - withinTargetsMeanSquares) / (betweenTargetsMeanSquares + (k-1)*withinTargetsMeanSquares)
    
    return ICC