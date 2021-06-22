#!/bin/bash
#SBATCH --job-name="test_adadbins"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=16G
#SBATCH --time=19:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/yugay/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/yugay/logs/slurm-%j.out

source /usr/stud/yugay/anaconda3/etc/profile.d/conda.sh 
conda activate adabins

export WANDB_API_KEY=5f2a463f1f161d549f23716919931e4bddf14f6d

srun python train.py --data_path /storage/user/yugay/datasets/MOTSynth --gt_path /storage/user/yugay/datasets/MOTSynth_annotations --data_path_eval /storage/user/yugay/datasets/MOTSynth --gt_path /storage/user/yugay/datasets/MOTSynth_annotations --bs 16 --epochs 100 --input_height 540 --input_width 960 --validate_every 10 
