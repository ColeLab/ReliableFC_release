# Write subject batch script to run DSIStudio.sh on Amarel

subj=$1

scriptsDir="/projects/f_mc1689_1/ReliableFC/docs/scripts/diffusion"
batchDir="${scriptsDir}/batchScripts/"

cd ${batchDir}

batchFilename=DSI-${subj}.sh

echo "#!/bin/bash" > $batchFilename
echo "#SBATCH --nodes=1" >> $batchFilename
echo "#SBATCH --ntasks=1" >> $batchFilename
echo "#SBATCH --partition=main" >> $batchFilename
echo "#SBATCH --time=1:00:00" >> $batchFilename
echo "#SBATCH --job-name=${subj}" >> $batchFilename
echo "#SBATCH --output=slurm.DSI-${subj}.out" >> $batchFilename
echo "#SBATCH --error=slurm.DSI-${subj}.err" >> $batchFilename
echo "#SBATCH --cpus-per-task=4" >> $batchFilename
echo "#SBATCH --mem=8000" >> $batchFilename
echo "#SBATCH --export=ALL" >>$batchFilename

echo "${scriptsDir}/DSIStudio.sh ${subj}" >> $batchFilename

# Submit the job
sbatch $batchFilename