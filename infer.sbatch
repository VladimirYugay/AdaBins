#!/bin/bash
#SBATCH --job-name="adabins_mots inference"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=32G
#SBATCH --time=300:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/yugay/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/yugay/logs/slurm-%j.out

source /usr/stud/yugay/anaconda3/etc/profile.d/conda.sh 
conda activate adabins

export WANDB_API_KEY=5f2a463f1f161d549f23716919931e4bddf14f6d

python infer_motsynth_depth.py --data_path /storage/remote/atcremers79/yugay/datasets/MOTSynth/frames/ --output_path /storage/remote/atcremers79/yugay/datasets/MOTSynth_annotations/all/ --seq_file_path ../../datasets/MOTSynth_annotations/official_split/test.txt --model_path checkpoints/adabins_mots_best.pt
