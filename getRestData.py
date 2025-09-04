import h5py
import numpy as np

# Project directories
dataDir = '/projects/f_mc1689_1/ReliableFC/data/empirical'

# fMRI rest run names
#restRuns = ['rfMRI_REST1_AP','rfMRI_REST2_AP','rfMRI_REST1_PA','rfMRI_REST2_PA']
restRuns = ['rfMRI_REST1_PA','rfMRI_REST1_AP','rfMRI_REST2_PA','rfMRI_REST2_AP']

def getRestData(subj,ses=0,withGSR=False):
    restData = []
    
    if ses == 1:
        runsList = ['rfMRI_REST1_PA','rfMRI_REST1_AP']
    elif ses == 2:
        runsList = ['rfMRI_REST2_PA','rfMRI_REST2_AP']
    
    if not withGSR:
        model = '24pXaCompCorXVolterra'
    elif withGSR:
        model = '36p'
    
    # Subject's final, clean dataset
    h5file = h5py.File(f'{dataDir}/postproc/{subj}_V1_MR_glmOutput_cortexsubcortex_data.h5','r')
    
    for run in restRuns:
        # Extract the run data, [:] loads it as an array
        restData_run = h5file[f'{run}/nuisanceReg_resid_{model}'][:]
        
        # Concatenate the data from all restRuns
        restData.extend(restData_run.T)
    
    # CLOSE THE h5 FILE
    h5file.close()
    
    # Save the nDatapoints x nNodes dataset as an array
    restData = np.asarray(restData).T
        
    return restData
