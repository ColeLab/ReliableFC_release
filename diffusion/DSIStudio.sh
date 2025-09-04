module purge
module load singularity/3.6.0

module use /projects/community/modulefiles/
module load connectome_wb/1.3.2-kholodvl

FSLDIR=/projects/f_mc1689_1/AnalysisTools/fsl
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh
####

subj=$1
subj=${subj}_V1_MR

projectDir=/projects/f_mc1689_1/ReliableFC
preproDir=${projectDir}/data/preprocessed/${subj}
dsisDir=${preproDir}/T1w/Diffusion/DSIStudio
cabnpDir=${projectDir}/docs/scripts/cabnp

DSIStudioContainer=/projects/community/containers/dsistudio/dsistudio_latest.sif


## Prepare DSIStudio directory
if [ ! -d ${dsisDir} ]; then mkdir -p ${dsisDir}/parcellations; fi

# Copy over necessary diffusion files
prereqFiles="data.nii.gz bvals bvecs nodif_brain_mask.nii.gz"
for file in ${prereqFiles}; do
    if [ ! -e ${dsisDir}/${file} ]; then
        cp ${preproDir}/T1w/Diffusion/${file} ${dsisDir}/${file}
    fi
done


## Make parcellation in individual subject's volume space
if [ ! -e ${dsisDir}/parcellations/cabnp_cortex.nii.gz ]; then

    # Left cortex surface labels to volume
    wb_command -label-to-volume-mapping ${cabnpDir}/cabnp_cortex_L.label.gii \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.L.pial_MSMAll.32k_fs_LR.surf.gii \
    ${preproDir}/T1w/Diffusion/data.nii.gz \
    ${dsisDir}/parcellations/cabnp_cortex_L.nii.gz \
    -ribbon-constrained \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.L.white_MSMAll.32k_fs_LR.surf.gii \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.L.pial_MSMAll.32k_fs_LR.surf.gii

    # Right cortex surface labels to volume
    wb_command -label-to-volume-mapping ${cabnpDir}/cabnp_cortex_R.label.gii \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.R.pial_MSMAll.32k_fs_LR.surf.gii \
    ${preproDir}/T1w/Diffusion/data.nii.gz \
    ${dsisDir}/parcellations/cabnp_cortex_R.nii.gz \
    -ribbon-constrained \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.R.white_MSMAll.32k_fs_LR.surf.gii \
    ${preproDir}/T1w/fsaverage_LR32k/${subj}.R.pial_MSMAll.32k_fs_LR.surf.gii
    
    # Combine left cortex and right cortex in volume file
    wb_command -volume-merge ${dsisDir}/parcellations/cabnp_cortex.nii.gz \
    -volume ${dsisDir}/parcellations/cabnp_cortex_L.nii.gz \
    -volume ${dsisDir}/parcellations/cabnp_cortex_R.nii.gz
    wb_command -volume-reduce ${dsisDir}/parcellations/cabnp_cortex.nii.gz MAX ${dsisDir}/parcellations/cabnp_cortex.nii.gz
    
    # Subcortex transform from MNI to subject space
    applywarp --ref=${preproDir}/T1w/Diffusion/data.nii.gz \
    --in=${cabnpDir}/cabnp_subcortex.nii \
    --warp=${preproDir}/MNINonLinear/xfms/standard2acpc_dc.nii.gz \
    --out=${dsisDir}/parcellations/cabnp_subcortex.nii.gz \
    --interp=nn

    # Combine left cortex, right cortex, and subcortex in one volume file
    # (I currently don't recommend using subcortical parcels with DSIStudio)
    wb_command -volume-label-import ${dsisDir}/parcellations/cabnp_subcortex.nii.gz "" ${dsisDir}/parcellations/cabnp_subcortex.nii.gz
    wb_command -volume-merge ${dsisDir}/parcellations/cabnp_cortex_subcortex.nii.gz \
    -volume ${dsisDir}/parcellations/cabnp_cortex_L.nii.gz \
    -volume ${dsisDir}/parcellations/cabnp_cortex_R.nii.gz \
    -volume ${dsisDir}/parcellations/cabnp_subcortex.nii.gz
    wb_command -volume-reduce ${dsisDir}/parcellations/cabnp_cortex_subcortex.nii.gz MAX ${dsisDir}/parcellations/cabnp_cortex_subcortex.nii.gz
fi


## Run DSI Studio commands
cd ${dsisDir}

# Generate SRC files
if [ ! -e ./data.src.gz ]; then
    singularity exec ${DSIStudioContainer} \
    dsi_studio --action=src --source=data.nii.gz --output=data.src.gz
fi

# Image Reconstruction
if [ ! -e ./data.src.gz.gqi.1.25.fib.gz ]; then
    singularity exec ${DSIStudioContainer} \
    dsi_studio --action=rec --source=data.src.gz --mask=nodif_brain_mask.nii.gz --method=4 --param0=1.25
fi

# Fiber tracking
# Runge-Kutta fourth order (--method=1) and smaller step size (--step_size=0.125) use more accurate numerical method, 
# give small improvements with little extra run time
# --connectivity_type=end gives poor results, only use 'pass' 
# --connectivity_value=count and ncount give similar results
if [ ! -e ./tracts.trk.gz ]; then
    singularity exec ${DSIStudioContainer} \
    dsi_studio --action=trk --source=data.src.gz.gqi.1.25.fib.gz --output=tracts.trk.gz --export=qa,dti_fa,stat,tdi \
    --thread_count=6 --fiber_count=100000 --method=1 --step_size=0.125 \
    --connectivity=parcellations/cabnp_cortex.nii.gz --connectivity_type=pass --connectivity_value=count,ncount
fi


chmod -Rf 775 ${dsisDir}
chgrp -Rf g_mc1689_1 ${dsisDir}

chmod -Rf 775 ${projectDir}/docs/scripts/diffusion/batchScripts
chgrp -Rf g_mc1689_1 ${projectDir}/docs/scripts/diffusion/batchScripts
