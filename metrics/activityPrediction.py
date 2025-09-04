import numpy as np
from scipy import stats

def activityPrediction(activity,conn,outputPred=False):
    '''
    Uses a functional connectivity matrix to predict the (held-out) activity of each node from the activities of all other nodes. Returns R^2 and Pearson's r as measures of the similarity between predicted and actual timeseries, with higher similarity indicating a more accurate connectivity model. Performs activity flow modelling when the activity is task activations.
    INPUT:
        activity : held-out timeseries ([nNodes x nDatapoints], or [nNodes x Datapoints x nSubjects])
        conn : connectivity matrix being tested
    OUTPUT:
        R2 : the Coefficient of Determination (R^2) between predicted and actual activity
        pearson : the Pearson correlation (r) between predicted and actual activity
    '''

    if activity.ndim == 2:
        activity = activity[:,:,np.newaxis]
        conn = conn[:,:,np.newaxis]

    nSubjs = activity.shape[2]
    nNodes = activity.shape[0]
    nodesList = np.arange(nNodes)

    R2 = np.zeros((nSubjs))
    pearson = np.zeros((nSubjs))

    # Predict the activities of each node (j) from the activities of all other nodes (i)
    # prediction_j = sum(activity_i * connectivity_ij)
    prediction = np.zeros((activity.shape))
    for n in range(nNodes):
        otherNodes = nodesList!=n
        for s in range(nSubjs):
            X = activity[otherNodes,:,s]
            y = activity[n,:,s]
            betas = conn[n,otherNodes,s]
            yPred = np.sum(X*betas[:,np.newaxis],axis=0)
            prediction[n,:,s] = yPred
    # Calculate R^2 and Pearson's r between the actual and predicted timeseries
    for s in range(nSubjs):
        sumSqrReg = np.sum((activity[:,:,s]-prediction[:,:,s])**2)
        sumSqrTot = np.sum((activity[:,:,s]-np.mean(activity[:,:,s]))**2)
        R2[s] = 1 - (sumSqrReg/sumSqrTot)
        pearson[s] = np.corrcoef(activity[:,:,s].flatten(),prediction[:,:,s].flatten())[0,1]

    if outputPred:
        return R2,pearson,prediction
    else:
        return R2,pearson