#!/bin/bash
#SBATCH --job-name="adadbins for motsynth"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_8000:3
#SBATCH --mem=128G
#SBATCH --time=300:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/yugay/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/yugay/logs/slurm-%j.out

source /usr/stud/yugay/anaconda3/etc/profile.d/conda.sh 
conda activate adabins

export WANDB_API_KEY=5f2a463f1f161d549f23716919931e4bddf14f6d

srun python train.py --data_path /storage/slurm/yugay/datasets/MOTSynth --gt_path /storage/slurm/yugay/datasets/MOTSynth_annotations --data_path_eval /storage/slurm/yugay/datasets/MOTSynth --gt_path /storage/slurm/yugay/datasets/MOTSynth_annotations --bs 24 --epochs 50 --input_height 576 --input_width 960 --validate_every 3 --filenames_file ./train_test_inputs/adabins_motsynth/motsynth_train.txt --filenames_file_eval ./train_test_inputs/adabins_motsynth/motsynth_test.txt --same-lr --name adabins_for_motsynth
