#!/bin/bash
#SBATCH --job-name=confit_hp_array
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-$(($(ls config/training_config_*.yaml | wc -l)-1))

# Activate your conda environment
module load cuda/12.4
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate ConFit

# Ensure log directory exists
mkdir -p logs

# Select config and data based on SLURM array index
CONFIG_FILES=(config/training_config_*.yaml)
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=$(basename "$CONFIG" .yaml | sed 's/training_config_//')
DATASET="${EXP_NAME}"

echo "Starting array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} -> Exp: $EXP_NAME"
\# Launch training with accelerate
accelerate launch \
  --config_file config/parallel_config.yaml \
  confit/train.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --sample_seed 0 \
  --model_seed 1
