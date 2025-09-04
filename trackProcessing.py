import pandas as pd
import numpy as np
import os

dataDir = '/projects/f_mc1689_1/ReliableFC/data'
scriptsDir = '/projects/f_mc1689_1/ReliableFC/docs/scripts'
trackerFile = f'{scriptsDir}/trackProcessing.tsv'
subjectsFile = f'{scriptsDir}/subjects_discovery.tsv'

fmriRunsList = ['rfMRI_REST1_AP','rfMRI_REST2_AP','rfMRI_REST1_PA','rfMRI_REST2_PA','tfMRI_CARIT_PA'] #,'tfMRI_FACENAME_PA','tfMRI_VISMOTOR_PA']

columns = ['data downloaded','diffusion preprocessing','cifti parcellation','brain masks','nuisance regressors','nuisance regression','tractography','DONE']

def update():
    
    # Get complete subject list
    subjectsDF = pd.read_csv(subjectsFile,sep='\t',index_col='subjects')
    subjList = subjectsDF.index
        
    trackerDF = pd.DataFrame(index=subjList,columns=columns)
        
    for subj in subjList:
        preproDir = f'{dataDir}/preprocessed/{subj}_V1_MR'
        rawDir = f'{dataDir}/downloads/imagingcollection01/{subj}_V1_MR/unprocessed'
        
        # Check that all necessary data has been downloaded
        missing = False
        checkpath1 = f'{preproDir}/MNINonLinear/wmparc.nii.gz'
        checkpath2 = f'{rawDir}/Diffusion/{subj}_V1_MR_dMRI_dir98_AP.nii.gz'
        checkpath3 = f'{preproDir}/T1w/{subj}_V1_MR/mri/T1.mgz'
        if not (os.path.isfile(checkpath1) and os.path.isfile(checkpath2) and os.path.isfile(checkpath3)):   
            missing = True
        for run in fmriRunsList:
            checkpath4 = f'{preproDir}/MNINonLinear/Results/{run}/{run}_Atlas_MSMAll.dtseries.nii'
            checkpath5 = f'{preproDir}/MNINonLinear/Results/{run}/{run}.nii.gz'
            if not (os.path.isfile(checkpath4) and os.path.isfile(checkpath5)):
                missing = True
        if not missing:
            trackerDF.loc[subj,'data downloaded'] = 1
        else:
            trackerDF.loc[subj,'data downloaded'] = 0
        
        # Check if diffusion preprocessing is complete
        missing = False
        checkpath = f'{preproDir}/T1w/Diffusion/data.nii.gz'
        if not os.path.isfile(checkpath):    
            missing = True
        if not missing:
            trackerDF.loc[subj,'diffusion preprocessing'] = 1
        else:
            trackerDF.loc[subj,'diffusion preprocessing'] = 0
        
        # Check for cifti parcellation (CAB-NP)
        missing = False
        for run in fmriRunsList:
            checkpath = f'{preproDir}/MNINonLinear/Results/{run}/{run}_Atlas_MSMAll_cab-np.ptseries.nii'
            if not os.path.isfile(checkpath):
                missing = True
        if not missing:
            trackerDF.loc[subj,'cifti parcellation'] = 1
        else:
            trackerDF.loc[subj,'cifti parcellation'] = 0
        
        # Check if brain masks were created (for nuisance regression)
        missing = False
        checkpath = f'{preproDir}/masks/{subj}_V1_MR_ventricles_func_eroded.nii.gz'
        if not os.path.isfile(checkpath):
            missing = True
        if not missing:
            trackerDF.loc[subj,'brain masks'] = 1
        else:
            trackerDF.loc[subj,'brain masks'] = 0
        
        # Check if nuisance regressors were created
        missing = False
        checkpath = f'{dataDir}/preprocessed/nuisanceRegressors/{subj}_V1_MR_nuisanceRegressors.h5'
        if not os.path.isfile(checkpath):
            missing = True
        elif not os.path.getsize(checkpath) > 840000: #1000000:
            missing = True
        if not missing:
            trackerDF.loc[subj,'nuisance regressors'] = 1
        else:
            trackerDF.loc[subj,'nuisance regressors'] = 0
            
        # Check if nuisance regression was performed
        missing = False
        checkpath = f'{dataDir}/preprocessed/postproc/{subj}_V1_MR_glmOutput_cortexsubcortex_data.h5'
        if not os.path.isfile(checkpath):
            missing = True
        elif not os.path.getsize(checkpath) > 16000000: #18000000:
            missing = True
        if not missing:
            trackerDF.loc[subj,'nuisance regression'] = 1
        else:
            trackerDF.loc[subj,'nuisance regression'] = 0
        
        # Check for tractography (processing with DSI Studio)
        missing = False
        checkpath = f'{preproDir}/T1w/Diffusion/DSIStudio/tracts.trk.gz.cabnp_cortex.ncount.pass.connectogram.txt'
        if not os.path.isfile(checkpath):    
            missing = True
        if not missing:
            trackerDF.loc[subj,'tractography'] = 1
        else:
            trackerDF.loc[subj,'tractography'] = 0
        
    trackerDF.to_csv(trackerFile,sep='\t',na_rep='n/a',index_label='subjects')
    os.chmod(trackerFile, mode=0o775)


def getSubjects(nextStep,prereqs=[]):
    trackerDF = pd.read_csv(trackerFile,sep='\t',index_col='subjects')
    subjList = []
    for subj in trackerDF.index:
        processSubject = False
        if nextStep == 'None' or trackerDF.loc[subj,nextStep] == 0:
            processSubject = True
            for step in prereqs:
                if trackerDF.loc[subj,step] == 0:
                    processSubject = False
        if processSubject:
            subjList.append(subj)
    return subjList