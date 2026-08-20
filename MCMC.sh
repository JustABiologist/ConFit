#!/bin/bash
#SBATCH --job-name=confit_MCMC
#SBATCH --output=logs/MCMC_%j.out
#SBATCH --error=logs/MCMC_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nodes=1

# activate your conda env
module load cuda/12.4
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate ConFit

# ensure log directory exists
mkdir -p logs

accelerate launch --mixed_precision fp16 MCMC_ConFIT.py --sampler masla --penalties "28,C,6;29,P,3;39,C,6;37,G,5;33,P,3" --penalty_mode "predictive" --alpha 0.45 --temp 1.0 --k_flips 2 --num_chains 128 --steps 200 --top_n 40 --fasta_out mgrb_avg_glob_normed_P29_P33.fasta --checkpoint /home/gruenf/ConFit/checkpoint/mgrb_avg_glob_normed/seed1
