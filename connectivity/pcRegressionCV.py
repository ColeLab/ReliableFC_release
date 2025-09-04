import os
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Kirsten Peterson, Jan. 2024

# Peterson, K. L., Sanchez-Romero, R., Mill, R. D., & Cole, M. W. (2023). Regularized partial correlation provides reliable functional connectivity estimates while correcting for widespread confounding. In bioRxiv (p. 2023.09.16.558065). https://doi.org/10.1101/2023.09.16.558065

def pcRegressionCV(data, numsPCs=None, kFolds=10, optMethod='R2', saveFiles=0, outDir='', foldsToRun='all', targetNodesToRun='all',  nodewiseHyperparams=False, foldsScheme='blocked'): 
    '''
    Runs node-wise lasso regression to compute the L1-regularized multiple regression FC matrix of a dataset, using cross validation to select the optimal L1 hyperparameters. Currectly, model performance for each hyperparameter value is scored as the R^2 between held-out activity predicted for each node using the training data connectivity matrix and the actual held-out activity. 
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1s : list of L1 (lambda1) hyperparameter values to test (all values must be >0); scales the lasso penalty term; the default L1s may not be suitable for all datasets and are only intended as a starting point
        kFolds : number of cross-validation folds to use during hyperparameter selection; an FC model is fit k times, each time using all folds but one; we recommend using a higher number (e.g., at least 10) so that the number of timepoints being fit is close to that of the full dataset (e.g., at least 90% as many timepoints), since the optimal hyperparameter value depends on data quantity
        optMethod : method for choosing the optimal hyperparameter (only 'R2' currently available), or None to skip
        saveFiles : whether to save intermediate and output variables to .npy files (0 = no; 1 = savefinal optimal FC matrix, optimal hyperparameter value, and model fit metrics (i.e., R^2) for each hyperparameter and fold; 2 = save CV fold connectivity coefficients too)
        outDir : if saveFiles>=1, the directory in which to save the files
        foldsToRun : list of CV folds (0 to kFolds-1) for which to compute model coefficients during hyperparameter selection; default ('all') is to run for all kFolds folds, but running only a subset of folds can reduce computation
        targetNodesToRun : list of target nodes (0 to nNodes-1) for which to fit an FC model (i.e., rows in the FC matrix), with all variables in the provided dataset used as source nodes (i.e., columns in the FC matrix); default ('all') is to use all nodes in the dataset as both targets and sources, to construct an [nNodes x nNodes] FC matrix
        nodewiseHyperparams : whether to select a different optimal hyperparameter for each target node's regression model
        
    OUTPUT:
        FC : lasso connectivity matrix, using optimal L1 value from list of L1s
        cvResults : dictionary with 'bestParam' for the optimal L1 value, 'L1s' for the list of tested L1 values, and 'R2' containing scores for every L1 value and cross-validation fold
    '''
    
    if numsPCs is None:
        numsPCs = np.arange(10,np.amin([data.shape[0]-1,data.shape[1]])+1,10) # max number of PCs is min(nNodes-1, nTRs)
        # We'd recommend checking the optimal hyperparameters for a few subjects and then narrowing down the range
    else:
        numsPCs = np.array(numsPCs)
    
    if not ((optMethod == 'R2') or (optMethod is None)):
        raise ValueError(f'optMethod "{optMethod}" does not match available methods. Available options are "R2" or None (compute CV matrices but skip model selection).')
    
    if isinstance(foldsToRun,str) and (foldsToRun == 'all'):
        foldsToRun = np.arange(kFolds)
    else:
        foldsToRun = np.array(foldsToRun)
        
    # Divide timepoints into folds
    nTRs = data.shape[1]
    kFoldsTRs = np.full((kFolds,nTRs),False)
    if foldsScheme=='interleaving':
        k = 0
        for t in range(nTRs):
            kFoldsTRs[k,t] = True
            k += 1
            if k >= kFolds:
                k = 0
    elif foldsScheme=='blocked':
        TRsPerFold = nTRs/kFolds
        t1 = 0
        for k in range(kFolds):
            t2 = int(np.round((k+1)*TRsPerFold))
            kFoldsTRs[k,t1:t2] = True
            t1 = t2
    
    # Define arrays to hold performance metrics
    nNodes = data.shape[0]
    if nodewiseHyperparams:
        scores = np.full((len(numsPCs),kFolds,nNodes),np.nan)
    else:
        scores = np.full((len(numsPCs),kFolds,1),np.nan)
    
    # If saving intermediate files
    if saveFiles >= 1:
        
        # Where to save files
        if outDir == '':
            outDir = os.getcwd()
            print(f'Directory for output files not provided, saving to current directory:\n{outDir}')
        if not os.path.exists(outDir):
            os.mkdir(outDir)
            
        outfileCVModel = {}
        outfileCVScores = {}
        
        # Loop through folds
        for k in foldsToRun:
            outfileCVModel[k] = {}
            outfileCVScores[k] = {}
            
            # Loop through numbers of PCs - find which need to be run
            numsPCsToRun = np.full(numsPCs.shape,True)
            for c,numPCs in enumerate(numsPCs):
                outfileCVModel[k][numPCs] = f'{outDir}/numPCs-{numPCs}_kFolds-{k+1}of{kFolds}.npy'
                if nodewiseHyperparams:
                    outfileCVScores[k][numPCs] = f'{outDir}/numPCs-{numPCs}_kFolds-{k+1}of{kFolds}_{optMethod}_nodewise.npy'
                else:
                    outfileCVScores[k][numPCs] = f'{outDir}/numPCs-{numPCs}_kFolds-{k+1}of{kFolds}_{optMethod}.npy'
            
                # If performance metrics were already saved for this fold and number of PCs, load and move on
                if os.path.exists(outfileCVScores[k][numPCs]):
                    scores[c,k] = np.load(outfileCVScores[k][numPCs])
                    numsPCsToRun[c] = False
                    
                # If FC matrix was saved for this fold and L1, calculate performance metric and move on
                elif os.path.exists(outfileCVModel[k][numPCs]):
                    pcRegCV = np.load(outfileCVModel[k][numPCs])
                    if optMethod == 'R2':
                        r2,r = activityPrediction(data[:,kFoldsTRs[k]],pcRegCV,nodewise=nodewiseHyperparams)
                        scores[c,k] = r2.squeeze()
                        np.save(outfileCVScores[k][numPCs],scores[c,k])
                    numsPCsToRun[c] = False
                    
            # Estimate the PC regression FC matrices for missing numbers of PCs
            if np.sum(numsPCsToRun) > 0:
                pcRegCV = pcRegression(data[:,~kFoldsTRs[k]],numsPCs[numsPCsToRun],targetNodesToRun=targetNodesToRun)
                    
                # Save CV matrices if saveFiles = 2
                if saveFiles == 2:
                    for c,numPCs in enumerate(numsPCs[numsPCsToRun]):
                        np.save(outfileCVModel[k][numPCs],pcRegCV[:,:,c].squeeze())
                      
                if optMethod == 'R2':
                    # Calculate R^2
                    r2,r = activityPrediction(np.repeat(data[:,kFoldsTRs[k],np.newaxis],np.sum(numsPCsToRun),axis=2),pcRegCV,nodewise=nodewiseHyperparams)
                    scores[numsPCsToRun,k] = r2.T
                
                # Save performance metrics
                for c,numPCs in enumerate(numsPCs[numsPCsToRun]):
                    np.save(outfileCVScores[k][numPCs],r2[:,c].squeeze())
    
    # If not saving intermediate files
    else:
        # Loop through folds
        for k in foldsToRun:
            print(k)
            # Estimate the lasso FC matrices
            pcRegCV = pcRegression(data[:,~kFoldsTRs[k]],numsPCs,targetNodesToRun=targetNodesToRun)
                
            if optMethod == 'R2':
                # Calculate R^2
                r2,r = activityPrediction(np.repeat(data[:,kFoldsTRs[k],np.newaxis],pcRegCV.shape[2],axis=2),pcRegCV,nodewise=nodewiseHyperparams)
                scores[:,k] = r2.T 
    
    # Find the best param according to each performance metric 
    scoresKFoldMean = np.nanmean(scores,axis=1)
    if optMethod == 'R2':
        bestParam = numsPCs.T[np.argmax(scoresKFoldMean,axis=0)]
    
    # Estimate the regularized partial correlation using all data and the optimal hyperparameters
    pcRegOpt = pcRegression(data,bestParam,targetNodesToRun=targetNodesToRun,nodewiseHyperparams=nodewiseHyperparams).squeeze()
    
    if saveFiles >= .5:
        if nodewiseHyperparams:
            np.save(f'{outDir}/bestNumPCs_opt-{optMethod}_nodewise.npy',bestParam)
            np.save(f'{outDir}/pcRegression_opt-{optMethod}_nodewise.npy',pcRegOpt)
        else:
            np.save(f'{outDir}/bestNumPCs_opt-{optMethod}.npy',bestParam)
            np.save(f'{outDir}/pcRegression_opt-{optMethod}.npy',pcRegOpt)
    
    cvResults = {'bestParam': bestParam, optMethod: scores, 'numbersOfPCs': numsPCs}
    
    return pcRegOpt,cvResults

#-----
def pcRegression(data,numsPCs,zscore=True,targetNodesToRun='all',nodewiseHyperparams=False):
    '''
    Calculates the PC regression coefficients of a dataset. Runs sklearn's Lasso function and several other necessary steps.
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        numPCs : hyperparameter value; single value for all target nodes, or list with a value for each node
        zscore : whether to zscore the input data, default True
        targetNodesToRun : 'all', or subset to use as target nodes 
        nodewiseHyperparams : whether to use different hyperparameter for each target node's regression model
    
    OUTPUT:
        conn : regularized multiple regression coefficients (i.e., the FC matrix)
    '''
    
    nNodes = data.shape[0]
    if isinstance(targetNodesToRun,str) and (targetNodesToRun == 'all'):
        targetNodesToRun = np.arange(nNodes)
    else:
        targetNodesToRun = np.array(targetNodesToRun)
    
    # Check numPCs
    if nodewiseHyperparams:
        if not (isinstance(numsPCs,list) or isinstance(numsPCs,np.ndarray)):
            raise ValueError(f'If nodewiseHyperparams=True, then L1 must be a list or np.ndarray. L1 is {type(numsPCs)}.')
        elif not (len(numsPCs)==nNodes):
            raise ValueError(f'If nodewiseHyperparams=True, then L1 must contain as many elements as there are nodes (data dimension 0). L1 has length {len(numsPCs)} while data has shape {data.shape}.')
        numPCsByNode = np.array(numsPCs).copy()
        numsPCs = np.array([numPCsByNode[0]])
    else:
        if (numsPCs.ndim==0 or isinstance(numsPCs,float) or isinstance(numsPCs,int)):
            numsPCs = [numsPCs]        
        numsPCs = np.array(numsPCs)

    # Pre-allocate FC matrix        
    conn = np.full((nNodes,nNodes,len(numsPCs)),np.nan)
    
    # Z-score the data
    # (recommended when applying regularization)
    if zscore:
        data = stats.zscore(data,axis=1)
    
    # Loop through target nodes (only nodes to keep, i.e. cortex only)
    allNodes = np.arange(nNodes)
    for target in targetNodesToRun:
        print('\t',target)
        sources = allNodes[allNodes!=target] # include all nodes in model (i.e. cortex and subcortex)
        
        X = data[sources,:].copy()
        y = data[target,:].copy()
            
        # Get node-specific hyperparameter
        if nodewiseHyperparams: 
            numsPCs = np.array([numPCsByNode[target]])
        
        for c,numPCs in enumerate(numsPCs):
            # Get PCA components of predictors
            pca = PCA(n_components=numPCs)
            X_reduced = pca.fit_transform(X.T)
        
            # Run the regression for target node
            reg = LinearRegression().fit(X_reduced,y.T)
        
            # Fill in coefficients
            conn[target,sources,c] = pca.inverse_transform(reg.coef_)
            conn[target,target,c] = 0
    
    return np.squeeze(conn)

#-----
def activityPrediction(activity,conn,zscore=True,nodewise=False):
    '''
    Uses a functional connectivity matrix to predict the (held-out) activity of each node from the activities of all other nodes. Returns R^2 and Pearson's r as measures of the similarity between predicted and actual timeseries, with higher similarity indicating a more accurate connectivity model.
    INPUT:
        activity : held-out timeseries ([nNodes x nDatapoints], or [nNodes x Datapoints x nSubjects])
        conn : connectivity matrix being tested
        zscore : whether to zscore the input activity data, default True
        nodewise : whether to calculate separate scores for each node
    OUTPUT:
        R2 : the Coefficient of Determination (R^2) between predicted and actual activity
        pearson : the Pearson correlation (r) between predicted and actual activity
    '''
    if zscore:
        activity = stats.zscore(activity,axis=1)
    
    if activity.ndim == 2:
        activity = activity[:,:,np.newaxis]
    if conn.ndim == 2:
        conn = conn[:,:,np.newaxis]
    
    nSubjs = activity.shape[2]
    nNodes = activity.shape[0]
    
    # Predict the activities of each node (j) from the activities of all other nodes (i)
    # prediction_j = sum(activity_i * connectivity_ij)
    prediction = np.full((activity.shape),np.nan)
    nodesList = np.arange(nNodes)
    for n in nodesList:
        otherNodes = nodesList!=n
        for s in range(nSubjs):
            X = activity[otherNodes,:,s]
            betas = conn[n,otherNodes,s]
            yPred = np.sum(X*betas[:,np.newaxis],axis=0)
            prediction[n,:,s] = yPred
        
    # Calculate R^2 and Pearson's r between the actual and predicted timeseries
    if nodewise:
        sumSqrReg = np.sum((activity-prediction)**2,axis=1)
        sumSqrTot = np.sum((activity-np.mean(np.mean(activity,axis=1),axis=0))**2,axis=1)
        R2 = 1 - (sumSqrReg/sumSqrTot)
          
        pearson = np.full((nNodes,nSubjs),np.nan)
        for s in range(nSubjs):    
            for n in range(nNodes):
                pearson[n,s] = np.corrcoef(activity[n,:,s],prediction[n,:,s])[0,1]

    else:
        sumSqrReg = np.sum(np.sum((activity-prediction)**2,axis=1),axis=0)
        sumSqrTot = np.sum(np.sum((activity-np.mean(np.mean(activity,axis=1),axis=0))**2,axis=1),axis=0)
        R2 = 1 - (sumSqrReg/sumSqrTot)
        R2 = np.reshape(R2,(1,nSubjs))
        
        pearson = np.full((1,nSubjs),np.nan)
        for s in range(nSubjs):
            pearson[0,s] = np.corrcoef(activity[:,:,s].flatten(),prediction[:,:,s].flatten())[0,1]
    
    return R2,pearson
