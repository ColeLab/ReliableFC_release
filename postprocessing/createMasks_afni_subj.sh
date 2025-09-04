#!/bin/bash
# Taku Ito
# 05/07/18

# Script that produces nifti masks for later nuisance regression
#### IMPORT AFNI 
export PATH=$PATH:/projects/f_mc1689_1/HCP_v2_prereqs/afni

#########################
basedir="/projects/f_mc1689_1/ReliableFC"
datadir="${basedir}/data/preprocessed"

#########################
subjNums=${1}

##
for subj in $subjNums
do
    subj=${subj}_V1_MR
    
    echo "Creating gray, white, ventricle, whole brain masks for subject ${subj}..."
    
    subjmaskdir=${datadir}/${subj}/masks
    functionalVolumeData=${datadir}/${subj}/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA.nii.gz

    if [ ! -e $subjmaskdir ]; then mkdir $subjmaskdir; fi 

    # HCP standard to parcel out white v gray v ventricle matter
    segparc=${datadir}/${subj}/MNINonLinear/wmparc.nii.gz

    # Change to subjmaskdir
    pushd $subjmaskdir

    ###############################
    ### Create whole brain masks
    echo "Creating whole brain mask for subject ${subj}..."
    3dcalc -overwrite -a $segparc -expr 'ispositive(a)' -prefix ${subj}_wholebrainmask.nii.gz
    # Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_wholebrainmask.nii.gz -prefix ${subj}_wholebrainmask_func.nii.gz
    # Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_wholebrainmask_func_dil1vox.nii.gz ${subj}_wholebrainmask_func.nii.gz
    
   

    ###############################
    ### Create gray matter masks
    echo "Creating gray matter masks for subject ${subj}..." 
    # Indicate the mask value set for wmparc.nii.gz
    # Gray matter mask set
    maskValSet="8 9 10 11 12 13 16 17 18 19 20 26 27 28 47 48 49 50 51 52 53 54 55 56 58 59 60 96 97 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_gmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_gmMask.nii.gz -prefix ${subj}_gmMask_func.nii.gz
    #Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_gmMask_func_dil1vox.nii.gz ${subj}_gmMask_func.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
       
   
    
    ###############################
    ### Create white matter masks
    echo "Creating white matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # White matter mask set
    maskValSet="250 251 252 253 254 255 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3015 3016 3017 3018 3019 3020 3021 3022 3023 3024 3025 3026 3027 3028 3029 3030 3031 3032 3033 3034 3035 4000 4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011 4012 4013 4014 4015 4016 4017 4018 4019 4020 4021 4022 4023 4024 4025 4026 4027 4028 4029 4030 4031 4032 4033 4034 4035 5001 5002"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_wmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_wmMask.nii.gz -prefix ${subj}_wmMask_func.nii.gz
    #Subtract graymatter mask from white matter mask (avoiding negative #s)
    3dcalc -a ${subj}_wmMask_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_wmMask_func_eroded.nii.gz -overwrite
    
    rm -f ${subj}mask_temp.nii.gz
          

    
    ###############################
    ### Create ventricle masks
    echo "Creating ventricle matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # Ventricle mask set
    maskValSet="4 43 14 15"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_ventricles.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_ventricles.nii.gz -prefix ${subj}_ventricles_func.nii.gz
    #Subtract graymatter mask from ventricles (avoiding negative #s)
    3dcalc -a ${subj}_ventricles_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_ventricles_func_eroded.nii.gz -overwrite
    rm -f ${subjNum}mask_temp.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
    
    chmod 775 -R $subjmaskdir
    
    popd

done 



