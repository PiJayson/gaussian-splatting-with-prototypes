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
python train.py -s /shared/sets/datasets/nerf_synthetic/materials \
        --segment_paths single_balls/copper_ball.ply single_balls/gray_ball.ply single_balls/gold_ball.ply \
        --segment_counts 2 8 2 \
        --port 53615 \
        --lambda_dssim 0.2 \
        --iterations 4000 \
        --test_iterations 400 4000 \
        --position_lr_init 0.01 \
        --position_lr_final 0.0005 \
        --position_lr_delay_mult 0.0 \
        --position_lr_max_steps 4000 \
        --rotation_lr 0.5 \
        --scaling_lr 0.003 \
        --batch_size 5 \
        --param_position 0 \
        --param_rotation 1 \
        --param_scale 0 \
        --position_noise 0 \
        --rotation_noise 0.5 \
        --scale_noise 0.7 \
        --uniform_scale 0
