import numpy as np
import os
import sys

'''
Downsamples hcp preprocessed dtseries into CIFTI parcels then converts to .tsv
Assumes you have access to connectome workbench command line
'''
# Variables
projectDir = '/projects/f_mc1689_1/ReliableFC'
PARC = 'cab-np'
OVERWRITE = 0
VERBOSE = 0
runsList = ['rfMRI_REST1_PA','rfMRI_REST1_AP','rfMRI_REST2_PA','rfMRI_REST2_AP','tfMRI_CARIT_PA']#,'tfMRI_FACENAME_PA','tfMRI_VISMOTOR_PA']

def CIFTIdownsample(subjList,runsList=runsList):

    if PARC == 'cab-np':
        parcellationFile = '/projects/f_mc1689_1/AnalysisTools/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

    for subj in subjList:
        print('Subject=',subj)
        
        for run in runsList:
            print('\t',run)
            subjDir = f'{projectDir}/data/preprocessed/{subj}_V1_MR/MNINonLinear/Results/{run}/'
        
            #input file
            inputFile  = f'{subjDir}/{run}_Atlas_MSMAll.dtseries.nii'
            if VERBOSE:
                print(f'\t\tInput filename: {inputFile}')
        
            #output file
            outputFile  = f'{subjDir}/{run}_Atlas_MSMAll_{PARC}.ptseries.nii'
            if VERBOSE:
                print(f'\t\tOutput filename: {outputFile}')
        
            # check if the input data exists
            if os.path.isfile(inputFile):
                # check if the output data exist
                if os.path.isfile(outputFile) and OVERWRITE == 0:
                    print('\t\t*** The output data already exist - skipping')
                
                else:
                    # downsample the data
                    print('\t\tDownsampling grayordinate data to parcels')
                    cmd = f'wb_command -cifti-parcellate {inputFile} {parcellationFile} COLUMN {outputFile} -method MEAN'
                    if VERBOSE:
                        print('\t\t\t',cmd)
                    os.system(cmd)

            else:
                print('\t\t*** The input data does not seem to exist for this task - skipping')