#!/bin/bash
#SBATCH --job-name=comgen_ua
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=comgen_ua_%j.out
#SBATCH --error=comgen_ua_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Move to repository in Barkla
cd ~/scratch/imdp/IMDPomega

# Activate your venv
source ~/venv/venv_imdp/bin/activate

# WLS license
export GRB_WLSACCESSID=17ba8d99-8b5c-49af-b9b6-fe77f492bbbf
export GRB_WLSSECRET=fb083a39-dc1b-4315-a505-0d706094bb65
export GRB_LICENSEID=2799552



# Run script
python imdp_runner.py UAV hoa_files
# python ~/scratch/imdp/IMDPomega/imdp_runner.py UAV hoa_files

echo "End time: $(date)"
