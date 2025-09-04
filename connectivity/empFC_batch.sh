subj=$1
method=$2

scriptsDir="/projects/f_mc1689_1/ReliableFC/docs/scripts/connectivity"
batchDir=${scriptsDir}/batchScripts

cd ${batchDir}

jobname=${method}_subj-${subj}
batchFilename=${batchDir}/${jobname}.sh

echo "#!/bin/bash" > $batchFilename
echo "#SBATCH --nodes=1" >> $batchFilename
echo "#SBATCH --ntasks=1" >> $batchFilename
echo "#SBATCH --partition=price" >> $batchFilename
echo "#SBATCH --time=10:00:00" >> $batchFilename
echo "#SBATCH --job-name=${jobname}" >> $batchFilename
echo "#SBATCH --output=slurm.${jobname}.out" >> $batchFilename
echo "#SBATCH --error=slurm.${jobname}.err" >> $batchFilename
echo "#SBATCH --cpus-per-task=1" >> $batchFilename
echo "#SBATCH --mem=24000" >> $batchFilename
echo "#SBATCH --export=ALL" >> $batchFilename

#echo "./pathsForR4.1.0.sh" >> $batchFilename

echo "python3.9 -c 'import sys; sys.path.append("\""${scriptsDir}"\""); import empFC_wrapper; empFC_wrapper.empFC_wrapper("\""${subj}"\"", "\""${method}"\"")'" >> $batchFilename

# Submit the job
sbatch $batchFilename