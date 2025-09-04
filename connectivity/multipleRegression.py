# Function adapted from Ruben Sanchez-Romero's CombinedFCToolBox 
# (https://github.com/ColeLab/CombinedFC/tree/master/CombinedFCToolBox) 
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def multipleRegression(data,targetNodesToRun=None,sigThresh=False,alpha=0.01):
    '''
    Runs node-wise regression to compute FC matrix of a dataset
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        targetNodesToRun : list of target nodes (0 to nNodes-1) for which to fit an FC model (i.e., rows in the FC matrix), with all variables in the provided dataset used as source nodes (i.e., columns in the FC matrix); default ('all') is to use all nodes in the dataset as both targets and sources, to construct an [nNodes x nNodes] FC matrix
        sigThresh : whether to threshold coefficients by statistical significance
        alpha : alpha value to use for significance threshold 
        
    OUTPUT:
        FC : multiple regression connectivity matrix
    '''
    nTRs = data.shape[0]
    nNodes = data.shape[1]
    if isinstance(targetNodesToRun,str) and (targetNodesToRun == 'all'):
        targetNodesToRun = np.arange(nNodes)
    else:
        targetNodesToRun = np.array(targetNodesToRun)
    
    # Z-score the data
    data_scaled = stats.zscore(data,axis=0)
    
    # Pre-allocate arrays
    conn = np.zeros((nNodes,nNodes))
    
    # Loop through target nodes (only nodes to keep, i.e. cortex only)
    for target in targetNodesToRun:
        sources = list(range(nNodes)) # include all nodes in model (i.e. cortex and subcortex)
        sources.remove(target)
        X = data_scaled[:,sources].copy()
        y = data_scaled[:,target].copy()
        # Run the regression for target node
        reg = LinearRegression().fit(X,y)
        
        if sigThresh == False:
            # Fill in coefficients
            conn[target,sources] = reg.coef_
        
        elif sigThresh == True:
            # Parameters estimated = intercept and the beta coefficients
            params = np.append(reg.intercept_,reg.coef_)
            nParams = len(params)
            # Obtain predicted target data
            y_pred = reg.predict(X)
            # Append a column of 1s (for intercept) to the regressors dataset
            newX = np.append(np.ones((nTRs,1)),X,axis=1)
            # see chapter 12 and 13 of Devore's Probability textbook
            # mean squared errors MSE = SSE/(n-k-1), where k is the number of covariates (pg.519 Devore's)
            MSE = (np.sum(np.square(y - y_pred)))/(nTRs - nParams)
            # Compute variance of parameters (intercept and betas)
            paramsVar = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
            # Compute standard deviation
            paramsStd = np.sqrt(paramsVar)
            # Transform parameters into t-statistics under the null of B = 0
            Bho = 0 #beta under the null
            tStats = (params - Bho)/paramsStd
            # p-value for a t-statistic in a two-sided one sample t-test
            pVals = 2*(1 - stats.t.cdf(np.abs(tStats),df=nTRs-1))
            
            # Remove the intercept p-value
            pVals = np.delete(pVals,0)
            # Record the Betas with p-values < alpha
            conn[target,sources] = np.multiply(reg.coef_,pVals<alpha)
        
    return conn