### Set the project name, your department code by default
#PBS -P scai
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -q scai_q
#PBS -l select=1:ncpus=16:ngpus=2:centos=amdepyc
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=6:00:00
#PBS -l software=PYTHON
# After job starts, must goto working directory.
# $PBS_O_WORKDIR is the directory from where the job is fired.
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd /scratch/scai/phd/aiz238140/Shashwat/AACL/code/sota-new/muril-moco

# module () {
#         eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
# }

eval "$(conda shell.bash hook)"
conda activate col775

python train_ddp.py
