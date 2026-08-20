#!/bin/bash
#SBATCH --job-name=confit_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# activate your conda env
module load cuda/12.4
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate ConFit

# ensure log directory exists
mkdir -p logs

# launch training
accelerate launch \
  --config_file config/parallel_config.yaml \
  confit/train.py \
  --config config/training_config.yaml \
  --dataset mgrb_confit_fitness \
  --sample_seed 0 \
  --model_seed 1
