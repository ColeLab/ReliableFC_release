set -e

## Get subject number
subj=$1
subj=${subj}_V1_MR
echo ${subj}

## Specify directories
projectDir=/projects/f_mc1689_1/ReliableFC
rawDir=${projectDir}/data/downloads/imagingcollection01/${subj}/unprocessed/Diffusion
preproDir=${projectDir}/data/preprocessed

## Set up HCP environment
envScript=${projectDir}/docs/scripts/SetUpHCPPipeline.sh
source ${envScript}

## Diffusion parameters
PosData="${rawDir}/${subj}_dMRI_dir98_PA.nii.gz@${rawDir}/${subj}_dMRI_dir99_PA.nii.gz"
NegData="${rawDir}/${subj}_dMRI_dir98_AP.nii.gz@${rawDir}/${subj}_dMRI_dir99_AP.nii.gz"
PEdir=2 #Use 1 for Left-Right Phase Encoding, 2 for Anterior-Posterior
EchoSpacing=0.69 #Echo Spacing or Dwelltime of dMRI image, set to NONE if not used. 
# Dwelltime = 1/(BandwidthPerPixelPhaseEncode * # of phase encoding samples): 
# DICOM field (0019,1028) = BandwidthPerPixelPhaseEncode, DICOM field (0051,100b) AcquisitionMatrixText first value (# of phase encoding samples).  
# On Siemens, iPAT/GRAPPA factors have already been accounted for.
Gdcoeffs="NONE"


## Run the diffusion node of HCP pipeline
# Extra parameters provide extra motion correction
${HCPPIPEDIR}/DiffusionPreprocessing/DiffPreprocPipeline.sh \
    --posData="${PosData}" --negData="${NegData}" \
    --path="${preproDir}" --subject="${subj}" \
    --echospacing="${EchoSpacing}" --PEdir=${PEdir} \
    --gdcoeffs="${Gdcoeffs}" \
    --cuda-version=${cudaVersion} \
    --extra-eddy-arg=--repol \
    --extra-eddy-arg=--ol_type=both \
    --extra-eddy-arg=--mporder=8 \
    --extra-eddy-arg=--s2v_niter=8 \
    --extra-eddy-arg=--slspec=${preproDir}/slspec.txt \
    --extra-eddy-arg=--niter=5 \
    --extra-eddy-arg=--fwhm=10,0,0,0,0 \




