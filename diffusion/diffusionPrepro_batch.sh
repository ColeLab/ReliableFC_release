#!/bin/bash

# This script runs a bash function on a large number of subjects (executing the HCP Preprocessing Pipelines) using the supercomputer queuing system

# Directory containing scripts to run
scriptsDir="/projects/f_mc1689_1/ReliableFC/docs/scripts/diffusion"

# Where batch scripts for each subject are written
batchDir="${scriptsDir}/batchScripts"

# List of subjects
subjList=$(<${scriptsDir}/subjectsToProcess.tsv)

cd ${batchDir}

i=0

# Make and execute a batch script for each subject
for subj in ${subjList}
do
    ((i++))
    if [[ $i%2 -eq 0 ]]; then node=pascal009; else node=pascal008; fi
    echo $node
    
    batchFilename=diffPrepro-${subj}.sh
    
    echo "#!/bin/bash" > $batchFilename
    echo "#SBATCH --nodes=1" >> $batchFilename
    echo "#SBATCH --ntasks=1" >> $batchFilename
    echo "#SBATCH --partition=p_ps848" >> $batchFilename
    ###echo "#SBATCH --partition=gpu" >> $batchFilename
    echo "#SBATCH --time=10:00:00" >> $batchFilename
    echo "#SBATCH --job-name=diffPrepro-${subj}" >> $batchFilename
    echo "#SBATCH --output=slurm.diffPrepro-${subj}.out" >> $batchFilename
    echo "#SBATCH --error=slurm.diffPrepro-${subj}.err" >> $batchFilename
    echo "#SBATCH --cpus-per-task=4" >> $batchFilename
    echo "#SBATCH --mem=40000" >> $batchFilename
    echo "#SBATCH --gres=gpu:1" >> $batchFilename
    echo "#SBATCH --nodelist=${node}" >> $batchFilename
    # 'Export all' purportedly exports all environmental variables to the job environment:
    echo "#SBATCH --export=ALL" >> $batchFilename
    
    echo "cd $scriptsDir" >> $batchFilename
    #echo "module --show_hidden avail cuda" >> $batchFilename
    
    # Command to execute HCP preproc script - all modules!
    echo "${scriptsDir}/diffusionPrepro_HCPPipeline.sh ${subj}" >> $batchFilename
    
    #Submit the job
    sbatch $batchFilename
    
done

