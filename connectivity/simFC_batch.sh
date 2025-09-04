sim=$1
method=$2
sesList=$3
nList=$4
tList=$5

scriptsDir="/projects/f_mc1689_1/ReliableFC/docs/scripts/connectivity"
batchDir=${scriptsDir}/batchScripts

cd ${batchDir}

jobname=${method}_sim-${sim}
batchFilename=${batchDir}/${jobname}.sh

echo "#!/bin/bash" > $batchFilename
echo "#SBATCH --nodes=1" >> $batchFilename
echo "#SBATCH --ntasks=1" >> $batchFilename
echo "#SBATCH --partition=main,price" >> $batchFilename
echo "#SBATCH --time=4:00:00" >> $batchFilename
echo "#SBATCH --job-name=${jobname}" >> $batchFilename
echo "#SBATCH --output=slurm.${jobname}.out" >> $batchFilename
echo "#SBATCH --error=slurm.${jobname}.err" >> $batchFilename
echo "#SBATCH --cpus-per-task=1" >> $batchFilename
echo "#SBATCH --mem=24000" >> $batchFilename
echo "#SBATCH --export=ALL" >> $batchFilename

#echo "./pathsForR4.1.0.sh" >> $batchFilename

echo "python3.9 -c 'import sys; sys.path.append("\""${scriptsDir}"\""); import simFC_wrapper; simFC_wrapper.simFC_wrapper(${sim}, "\""${method}"\"",sesList=${sesList},noiseLevelsList=${nList},ntrsList=${tList})'" >> $batchFilename

# Submit the job
sbatch $batchFilename