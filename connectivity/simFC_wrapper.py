import numpy as np
import h5py
import os
import glob
from graphicalLassoCV import graphicalLassoCV,graphicalLasso
from graphicalRidgeCV import graphicalRidgeCV,graphicalRidge
from pcRegressionCV import pcRegressionCV,pcRegression
from parCorrInvCov import parCorrInvCov
from multipleRegression import multipleRegression
dataDir = '/projects/f_mc1689_1/ReliableFC/data'

nSes = 1
sesList = np.arange(nSes)
ses = 0
ntrs = 250
ntrsNum = 250
ntrsList = [50,100,200,300,400,500,1000,10000]
noiseLevelsList = [.25,.5,1]
#ntrsList = [250]
#noiseLevelsList = [.5]
TR = 'none'
k = 4
nModules = 5
path = f'{dataDir}/simulations/BANet_k-{k}_nModules-{nModules}/noiseAndTRs'#_boldConv'

overwrite = True
#CV = False
CV = True
kFolds = 10
optMethod = 'R2'
checkPrevBestParam = False#True

paramsOpt = {}
paramsFull = {}
#paramsOpt['graphicalLasso'] = np.round(np.arange(.0,.155,.005),3)
paramsOpt['graphicalLasso'] = np.round(np.arange(.0,.455,.005),3)
paramsFull['graphicalLasso'] = np.round(np.hstack([np.arange(.002,.01,.002),np.arange(.01,.41,.01)]),3)

#paramsOpt['graphicalRidge'] = np.round(np.arange(.0,4.1,.1),3)
paramsOpt['graphicalRidge'] = np.round(np.arange(.0,22.1,.1),3)
paramsFull['graphicalRidge'] = np.round(np.hstack([np.arange(.02,.1,.02),np.arange(.1,4.1,.1)]),3)

paramsOpt['pcRegression'] = np.round(np.arange(10,105,5),3)
paramsOpt['pcRegression'][-1] = 99
paramsFull['pcRegression'] = np.round(np.hstack([np.arange(1,10,1),np.arange(10,101,5)]),3)
paramsFull['pcRegression'][-1] = 99

paramsOpt['pairwiseCorr'] = []
paramsOpt['partialCorr'] = []
paramsFull['pairwiseCorr'] = []
paramsFull['partialCorr'] = []

def simFC_wrapper(sim,method,sesList=sesList,noiseLevelsList=noiseLevelsList,ntrsList=ntrsList):
    h5filename = f'{path}/sim-{sim}.h5'
    for ses in sesList:
        for ntrs in ntrsList:
            ntrsNum = ntrs
            for noiseLevel in noiseLevelsList:
                print(sim,ses,ntrs)
                outDir = f'{path}/{method}/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}'
                if not os.path.exists(outDir):
                    os.makedirs(outDir)
        
                with h5py.File(h5filename,'r') as h5file:
                    data = h5file[f'ses-{ses}/TR-{TR}/noise-{noiseLevel}/nTRs-{ntrs}/data'][:]
            
                if (method=='pairwiseCorr') or (method=='partialCorr'):
                    if method=='pairwiseCorr':
                        fc = np.corrcoef(data)
                        np.fill_diagonal(fc,0)
                    elif method=='partialCorr':
                        fc,prec = parCorrInvCov(data.T)
                        if not os.path.exists(f'{path}/graphicalLasso/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}'):
                            os.makedirs(f'{path}/graphicalLasso/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}')
                        if not os.path.exists(f'{path}/graphicalRidge/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}'):
                            os.makedirs(f'{path}/graphicalRidge/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}')
                        np.save(f'{path}/graphicalLasso/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}/FC_param-0.0.npy',fc)
                        np.save(f'{path}/graphicalRidge/{sim}_{ses}_TR-{TR}_noiseLevel-{noiseLevel}_nTRs-{ntrs}/FC_param-0.0.npy',fc)
                
                    np.save(f'{outDir}/FC.npy',fc)
                    np.save(f'{outDir}/{method}.npy',fc)
                    continue
        
                skip = False
                if CV:
                    if ~overwrite:
                        cvResults = {}
                        if os.path.exists(f'{outDir}/{method}_opt-{optMethod}.npy'):
                            skip = True
                        
                if CV and (~skip):
                    if checkPrevBestParam:
                        if os.path.exists(f'{outDir}/bestParam.npy'):
                            prevBestParam = np.load(f'{outDir}/bestParam.npy')
                            prevParams = np.load(f'{outDir}/params.npy')
                            if (prevBestParam > np.amin(prevParams)) and (prevBestParam < np.amax(prevParams)):
                                continue
                    
                    if method == 'graphicalLasso':
                        fc,cvResults = graphicalLassoCV(data,L1s=paramsOpt[method],kFolds=kFolds,optMethod=optMethod,saveFiles=1,outDir=outDir,foldsScheme='blocked')
                    elif method == 'graphicalRidge':
                        fc,cvResults = graphicalRidgeCV(data,L2s=paramsOpt[method],kFolds=kFolds,optMethod=optMethod,saveFiles=1,foldsScheme='blocked',outDir=outDir)
                    elif method == 'pcRegression':
                        cvResults = {}
                        params = paramsOpt[method][paramsOpt[method]<=(ntrsNum*((kFolds-1)/kFolds))]
                        fc,cvResults = pcRegressionCV(data,numsPCs=params,kFolds=kFolds,optMethod=optMethod,saveFiles=1,outDir=outDir,foldsScheme='blocked',nodewiseHyperparams=False)
                        tmpScores = np.full((len(paramsOpt[method]),kFolds),np.nan)
                        tmpScores[paramsOpt[method]<=(ntrsNum*((kFolds-1)/kFolds))] = cvResults[optMethod][:,:,0]
                        cvResults[optMethod] = tmpScores
                    
                    np.save(f'{outDir}/FC.npy',fc)
                    np.save(f'{outDir}/params.npy',paramsOpt[method])
                    np.save(f'{outDir}/bestParam.npy',cvResults['bestParam'])
                    np.save(f'{outDir}/{optMethod}s.npy',cvResults[optMethod])
        
                continue
        
                for param in paramsFull[method]:
                    print(param)
                    
                    #if ~overwrite:
                    if os.path.exists(f'{outDir}/FC_param-{param}.npy'):
                        continue
                    
                    if method == 'graphicalLasso':
                        fc,prec = graphicalLasso(data,param)
                    elif method == 'graphicalRidge':
                        fc,prec = graphicalRidge(data,param,tmpDir=outDir)
                    elif method == 'pcRegression':
                        if param == 99:
                            fc = multipleRegression(data.T)
                        else:
                            fc = pcRegression(data,param,nodewiseHyperparams=False)
            
                    np.save(f'{outDir}/FC_param-{param}.npy',fc)
                    
                 
