#!/bin/bash
#SBATCH --job-name=confit_MCMC
#SBATCH --output=logs/MCMC_%j.out
#SBATCH --error=logs/MCMC_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1

# activate your conda env
module load cuda/12.4
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate ConFit

# ensure log directory exists
mkdir -p logs

accelerate launch --mixed_precision fp16 MCMC_ConFIT.py --num_chains 10000 --steps 512 --top_n 128 --fasta_out LARGE_SCALE_VARIANTS-HUGE.fasta --checkpoint /home/gruenf/ConFit/checkpoint/mgrb_filtered_better_smaller_lora/seed1
