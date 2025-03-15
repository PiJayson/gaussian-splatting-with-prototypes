#!/bin/bash
#SBATCH --job-name=gauss_prototype_training
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=student
#SBATCH --output=output.log
#SBATCH --error=error.log


# gs_number - a restrict number of max gaussians in prototype.
# gs_number [1, inf] - max number of gaussians

cd $HOME/gaussian-splatting
source ~/miniconda3/bin/activate gaussian_splatting
python train.py -s ./datasets/nerf_lego \
        --port 53615 \
        --iterations 10000 \
        --test_iterations 1000 10000 \
        --gs_number 1000