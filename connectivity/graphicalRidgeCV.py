import os
import numpy as np
from scipy import stats
from scipy import linalg
from sklearn.covariance import log_likelihood,empirical_covariance

#cmdLoadR = ''
cmdLoadR = 'export PATH=$HOME/pcre/10.40/bin:$PATH; export C_INCLUDE_PATH=$HOME/pcre/10.40/include; export CPLUS_INCLUDE_PATH=$HOME/pcre/10.40/include; export LIBRARY_PATH=$HOME/pcre/10.40/lib; export LD_LIBRARY_PATH=$HOME/pcre/10.40/lib; export MANPATH=$HOME/pcre/10.40/share/man:$MANPATH; '
cmdLoadR = cmdLoadR + 'export PATH=$HOME/R/4.1.0/bin:$PATH; export C_INCLUDE_PATH=$HOME/R/4.1.0/include:$C_INCLUDE_PATH; export CPLUS_INCLUDE_PATH=$HOME/R/4.1.0/include:$CPLUS_INCLUDE_PATH; export LIBRARY_PATH=$HOME/R/4.1.0/lib:$LIBRARY_PATH; export LD_LIBRARY_PATH=$HOME/R/4.1.0/lib:$HOME/R/4.1.0/lib64/R/lib:$LD_LIBRARY_PATH; export MANPATH=$HOME/R/4.1.0/share/man:$MANPATH; '

# Kirsten Peterson, Sept. 2023

# Peterson, K. L., Sanchez-Romero, R., Mill, R. D., & Cole, M. W. (2023). Regularized partial correlation provides reliable functional connectivity estimates while correcting for widespread confounding. In bioRxiv (p. 2023.09.16.558065). https://doi.org/10.1101/2023.09.16.558065

def graphicalRidgeCV(data,L2s=None,kFolds=10,optMethod='loglikelihood',saveFiles=0,outDir='',foldsScheme='blocked'):
    '''
    Runs graphical lasso to compute the L1-regularized partial correlation matrix of a dataset, using cross-validation to select the optimal L1 hyperparameter. Currently, model performance for each hyperparameter value is scored as: the loglikelihood between the training data precision (regularized inverse covariance) matrix and held-out data empirical (unregularized) covariance matrix; or the R^2 between held-out activity (time series) predicted for each node using the training data connectivity matrix and the actual held-out activity (time series).
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1s : list of L1 (lambda1) hyperparameter values to test (all values must be >0); scales the lasso penalty term; the default L1s may not be suitable for all datasets and are only intended as a starting point
        kFolds : number of cross-validation folds to use during hyperparameter selection (FC model is fit k times, each time using all folds but one)
        optMethod : method for choosing the optimal hyperparameter ('loglikelihood' or 'R2')
        saveFiles : whether to save intermediate and output variables to .npy files (0 = no, 1 = save R^2 and negative loglikelihood results, 2 = save connectivity and precision matrices too
        outDir : if saveFiles>=1, the directory in which to save the files
    OUTPUT:
        parCorr : graphical lasso (L1-regularized partial correlation) connectivity matrix, using optimal L1 value from list of L1s
        cvResults : dictionary with 'bestParam' for the optimal L1 value from input list of L1s ('L1s') and 'loglikelihood' or 'R2' containing scores for every cross validation fold and L1 value
    '''
    
    if not ((optMethod == 'loglikelihood') or (optMethod == 'R2')):
        raise ValueError(f'optMethod "{optMethod}" does not match available methods. Available options are "loglikelihood" and "R2".')

    if L2s is None:
        # Test log-scaled range of L2s (from 0.316 to 0.001)
        L2s = np.arange(.5,-2.1,-.1) 
        L2s = 10**L2s
        L2s = np.round(L2s,20)
        # We recommend checking the optimal hyperparameters for a few subjects and then narrowing down the range
    else:
        L2s = np.array(L2s)

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
    
    # Where to save files
    if outDir == '':
        outDir = os.getcwd()
        print(f'Directory for output files not provided, saving to current directory:\n{outDir}')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    # Define arrays to hold performance metrics
    scores = np.zeros((len(L2s),kFolds))
    
    # If saving intermediate files
    if saveFiles >= 1:
        
        # Loop through L2s
        for l,L2 in enumerate(L2s):
            outfileCVScores = f'{outDir}/L2-{L2}_{optMethod}.npy'

            # If performance metrics were already saved for this L2, load them and move on
            if os.path.exists(outfileCVScores):
                scores[l,:] = np.load(outfileCVScores)
                continue

            # Loop through folds
            for k in range(kFolds):
                outfileParCorr = f'{outDir}/L2-{L2}_kFolds-{k+1}of{kFolds}_partialCorr.npy'
                outfilePrec = f'{outDir}/L2-{L2}_kFolds-{k+1}of{kFolds}_precison.npy'

                # Check if partial corr and precision matrices were already created for this fold
                if os.path.exists(outfileParCorr) and os.path.exists(outfilePrec):
                    parCorr = np.load(outfileParCorr)
                    prec = np.load(outfilePrec)
                else:
                    # Estimate the regularized partial correlation and precision (intermediate) matrices
                    parCorr,prec = graphicalRidge(data[:,~kFoldsTRs[k]],L2,tmpDir=outDir)

                    # Save partial corr and precision matrices if saveFiles = 2
                    if saveFiles == 2:
                        np.save(outfileParCorr,parCorr)
                        np.save(outfilePrec,prec)

                if optMethod == 'loglikelihood':
                    # Calculate negative loglikelihood
                    empCov_test = np.cov(stats.zscore(data[:,kFoldsTRs[k]],axis=1),rowvar=True)
                    scores[l,k] = -log_likelihood(empCov_test,prec)

                elif optMethod == 'R2':
                    # Calculate R^2
                    scores[l,k],r = activityPrediction(stats.zscore(data[:,kFoldsTRs[k]],axis=1),parCorr)

            # Save performance metrics for this L2
            np.save(outfileCVScores,scores[l,:])

    # If not saving intermediate files
    else:
        # Loop through L2s
        for l,L2 in enumerate(L2s):
            # Loop through folds
            for k in range(kFolds):
                # Estimate the regularized partial correlation and precision (intermediate) matrices
                parCorr,prec = graphicalRidge(data[:,~kFoldsTRs[k]],L2,tmpDir=outDir)

                if optMethod == 'loglikelihood':
                    # Calculate negative loglikelihood
                    empCov_test = np.cov(stats.zscore(data[:,kFoldsTRs[k]],axis=1),rowvar=True)
                    scores[l,k] = -log_likelihood(empCov_test,prec)

                elif optMethod == 'R2':
                    # Calculate R^2
                    scores[l,k],r = activityPrediction(stats.zscore(data[:,kFoldsTRs[k]],axis=1),parCorr)

    # Find the best param according to each performance metric
    meanScores = np.mean(scores,axis=1)
    if optMethod == 'loglikelihood':
        bestParam = L2s[meanScores==np.amin(meanScores)][0]
    elif optMethod == 'R2':
        bestParam = L2s[meanScores==np.amax(meanScores)][0]

    # Estimate the regularized partial correlation using all data and the optimal hyperparameters
    parCorr,prec = graphicalRidge(data,bestParam,tmpDir=outDir)

    if saveFiles >= 1:
        np.save(f'{outDir}/bestL2_opt-{optMethod}.npy',bestParam)
        np.save(f'{outDir}/graphicalRidge_opt-{optMethod}.npy',parCorr)

    cvResults = {'bestParam': bestParam, optMethod: scores, 'L2s': L2s}

    return parCorr,cvResults

#-----
def graphicalRidge(data,L2,tmpDir='',tmpFileBase=''):
    # Where to save temporary files
    if tmpFileBase == '':
        if tmpDir == '':
            tmpDir = os.getcwd()
        tmpFileBase = f'{tmpDir}/tmp{np.random.randint(0,high=1000000,size=1)[0]:06}'
        print(f'Name for temporary files not provided, saving to current directory:\n{tmpFileBase}_...')
        
    nNodes = data.shape[0]
    
    # Z-score the data
    data_scaled = stats.zscore(data,axis=1)
    
    # Calculate and save empirical covariance matrix
    covFile = f'{tmpFileBase}_cov.csv'
    empCov = np.cov(data_scaled,rowvar=True)
    
    if L2 > 0:
        np.savetxt(covFile,empCov,delimiter=',')
    
        # Run the ridge regularization in R
        precFile = f'{tmpFileBase}_prec_L2-{L2}.csv'
        command = cmdLoadR + "Rscript -e '"
        command = command + f'library(rags2ridges); '
        command = command + f'empCov <- read.csv("{covFile}",header=FALSE); '
        command = command + f'empCov <- as.matrix(empCov); '
        command = command + f'rownames(empCov) <- 1:{nNodes}; '
        command = command + f'colnames(empCov) <- 1:{nNodes}; '
        command = command + f'prec <- ridgeP(empCov,lambda={L2}); '
        command = command + f'write.csv(prec,"{precFile}",row.names=FALSE)'
        command = command + "'"
        os.system(command)
            
        prec = np.loadtxt(precFile,delimiter=',',skiprows=1)
        os.system(f'rm {covFile}')
        os.system(f'rm {precFile}')
            
    elif L2 == 0:
        # Unregularized partial correlation
        prec = linalg.pinv(empCov)
        
    # Transform precision matrix into regularized partial correlation matrix
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    conn = -prec * denom * denom.T
    np.fill_diagonal(conn,0)
    
    return conn, prec

#-----
def activityPrediction(activity,conn):
    '''
    Uses a functional connectivity matrix to predict the (held-out) activity of each node from the activities of all other nodes. Returns R^2 and Pearson's r as measures of the similarity between predicted and actual timeseries, with higher similarity indicating a more accurate connectivity model.
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

    return R2,pearson