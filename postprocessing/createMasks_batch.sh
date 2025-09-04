# Write subject batch script to run createMasks_afni_subj.sh on Amarel

subj=$1

scriptsDir="/projects/f_mc1689_1/ReliableFC/docs/scripts/postprocessing"
batchDir="${scriptsDir}/batchScripts/"

cd ${batchDir}

batchFilename=createMasks-${subj}.sh

echo "#!/bin/bash" > $batchFilename
echo "#SBATCH --nodes=1" >> $batchFilename
echo "#SBATCH --ntasks=1" >> $batchFilename
echo "#SBATCH --partition=main" >> $batchFilename
echo "#SBATCH --time=0:30:00" >> $batchFilename
echo "#SBATCH --job-name=${subj}-masks" >> $batchFilename
echo "#SBATCH --output=slurm.createMasks-${subj}.out" >> $batchFilename
echo "#SBATCH --error=slurm.createMasks-${subj}.err" >> $batchFilename
echo "#SBATCH --cpus-per-task=4" >> $batchFilename
echo "#SBATCH --mem=8000" >> $batchFilename
echo "#SBATCH --export=ALL" >>$batchFilename

echo "${scriptsDir}/createMasks_afni_subj.sh ${subj}" >> $batchFilename

# Submit the job
sbatch $batchFilename