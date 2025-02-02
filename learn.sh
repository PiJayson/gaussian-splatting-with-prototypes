#!/bin/bash
#SBATCH --job-name=gauss_prototype_training
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=student
#SBATCH --output=output.log
#SBATCH --error=error.log


cd $HOME/gaussian-splatting
source ~/miniconda3/bin/activate gaussian_splatting
python train.py -s /shared/sets/datasets/nerf_synthetic/materials --segment_paths single_balls/copper_ball.ply single_balls/gray_ball.ply single_balls/gold_ball.ply --segment_counts 2 8 2 --iterations 5000 --test_iterations 3000 5000 --batch_size 35 --lambda_dssim 0.2 --position_lr_init 0.003 --position_lr_max_steps 5000
