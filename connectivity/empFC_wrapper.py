import numpy as np
import h5py
import os
import pandas as pd
import sys
from scipy import stats
from graphicalLassoCV import graphicalLassoCV,graphicalLasso
from graphicalRidgeCV import graphicalRidgeCV,graphicalRidge
from pcRegressionCV import pcRegressionCV,pcRegression
from parCorrInvCov import parCorrInvCov
from multipleRegression import multipleRegression
dataDir = '/projects/f_mc1689_1/ReliableFC/data'
scriptsDir = '/projects/f_mc1689_1/ReliableFC/docs/scripts'
sys.path.append(scriptsDir)
from getRestData import getRestData

# Order of parcels by network
parcType = 'cortex'
parcOrderFile = f'{scriptsDir}/cabnp/{parcType}_community_order.txt'
parcOrder = pd.read_csv(parcOrderFile, sep='\t', header=None)[0] - 1
nNodes = parcOrder.shape[0]

path = f'{dataDir}/empirical'

overwrite = False
CV = True
kFolds = 10
optMethod = 'R2'

sesList = [1,2]

paramsOpt = {}
paramsFull = {}
paramsOpt['graphicalLasso'] = np.round(np.arange(.005,.085,.005),3)
paramsFull['graphicalLasso'] = np.round(np.hstack([np.arange(.001,.005,.001),np.arange(.005,.205,.005)]),3)

paramsOpt['graphicalRidge'] = np.round(np.arange(0.1,1.3,.1),3)
paramsFull['graphicalRidge'] = np.round(np.hstack([np.arange(.02,.1,.02),np.arange(.1,3.1,.1)]),3)

paramsOpt['pcRegression'] = np.round(np.arange(10,130,5),3)
paramsFull['pcRegression'] = np.round(np.hstack([[1,5,10,15],np.arange(20,361,10)]),3)
paramsFull['pcRegression'][-1] = 359


def empFC_wrapper(subj,method,sesList=sesList):
    for ses in sesList:
        print(subj,ses)
        # Import subject rest data
        data = getRestData(subj,ses=ses)
    
        # Select nodes by parcellation type and put into network order
        data = data[parcOrder]
        ntrsNum = data.shape[1]
        
        outDir = f'{path}/connectivity/{method}/{subj}_ses-{ses}'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
                        
        if (method=='pairwiseCorr') or (method=='partialCorr'):
            if method=='pairwiseCorr':
                fc = np.corrcoef(data)
                np.fill_diagonal(fc,0)
            elif method=='partialCorr':
                if ntrsNum >= data.shape[0]:
                    fc,prec = parCorrInvCov(data.T)
                else:
                    fc = np.full((data.shape[0],data.shape[0]),np.nan)
                    
                if not os.path.exists(f'{path}/connectivity/graphicalLasso/{subj}_ses-{ses}'):
                    os.makedirs(f'{path}/connectivity/graphicalLasso/{subj}_ses-{ses}')
                if not os.path.exists(f'{path}/connectivity/graphicalRidge/{subj}_ses-{ses}'):
                    os.makedirs(f'{path}/connectivity/graphicalRidge/{subj}_ses-{ses}')
                np.save(f'{path}/connectivity/graphicalLasso/{subj}_ses-{ses}/FC_param-0.0.npy',fc)
                np.save(f'{path}/connectivity/graphicalRidge/{subj}_ses-{ses}/FC_param-0.0.npy',fc)
                
            np.save(f'{outDir}/{method}.npy',fc)
            np.save(f'{outDir}/FC.npy',fc)
            continue
        
        skip = False
        if CV:
            if ~overwrite:
                if not os.path.exists(f'{outDir}/FC.npy'):
                    print(f'{outDir}/FC.npy exists')
                    skip = True
            
        if CV and (~skip):
            if method == 'graphicalLasso':
                fc,cvResults = graphicalLassoCV(data,L1s=paramsOpt[method],kFolds=kFolds,optMethod=optMethod,saveFiles=1,outDir=outDir,foldsScheme='blocked')
            elif method == 'graphicalRidge':
                fc,cvResults = graphicalRidgeCV(data,L2s=paramsOpt[method],kFolds=kFolds,optMethod=optMethod,saveFiles=1,foldsScheme='blocked',outDir=outDir)
            elif method == 'pcRegression':
                params = paramsOpt[method][paramsOpt[method]<=(ntrsNum*((kFolds-1)/kFolds))]
                fc,cvResults = pcRegressionCV(data,numsPCs=params,kFolds=kFolds,optMethod=optMethod,saveFiles=2,outDir=outDir,foldsScheme='blocked',nodewiseHyperparams=False)
                tmpScores = np.full((len(paramsOpt[method]),kFolds),np.nan)
                tmpScores[paramsOpt[method]<=(ntrsNum*((kFolds-1)/kFolds))] = cvResults[optMethod][:,:,0]
                cvResults[optMethod] = tmpScores
                                      
            np.save(f'{outDir}/FC.npy',fc)
            np.save(f'{outDir}/params.npy',paramsOpt[method])
            np.save(f'{outDir}/bestParam.npy',cvResults['bestParam'])
            np.save(f'{outDir}/{optMethod}s.npy',cvResults[optMethod])
               
        #continue
        for param in paramsFull[method]:
            print(param)
            if os.path.exists(f'{outDir}/FC_param-{param}.npy'):
                print(f'{outDir}/FC_param-{param}.npy exists')
                continue
            
            if method == 'graphicalLasso':
                fc,prec = graphicalLasso(data,param)
            elif method == 'graphicalRidge':
                fc,prec = graphicalRidge(data,param,tmpDir=outDir)
            elif method == 'pcRegression':
                if param == 359:
                    fc = multipleRegression(data.T)
                    fc = (fc+fc.T)/2
                else:
                    fc = pcRegression(data,param,nodewiseHyperparams=False)
                    fc = (fc+fc.T)/2
            
            np.save(f'{outDir}/FC_param-{param}.npy',fc)
               
                 
