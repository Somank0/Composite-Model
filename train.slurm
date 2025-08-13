#!/bin/bash

#SBATCH --job-name=Composite_DRN_EB_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /home/seema/miniconda3/etc/profile.d/conda.sh

conda activate ml


pickle_folder="/storage/seema/sosaha/Pickle_files/CompDRN_4p3M_EB_sample"

composite_DRN_folder="/storage/seema/sosaha/CompositeDRN_V2"

echo "starting job at $(date)"

#$composite_DRN_folder/train --data_folder $pickle_folder --mp_layers 4 --agg_layers 3 --drn_hidden_dim 64 --drn_output_dim 64 --batch_size 500 --learning_rate 0.0001 --num_epochs 200 > $pickle_folder/training.log 2>&1 
$composite_DRN_folder/train --data_folder $pickle_folder --mp_layers 4 --agg_layers 3 --drn_hidden_dim 64 --drn_output_dim 64 --batch_size 500 --learning_rate 0.0001 --num_epochs 200 --predict_only > $pickle_folder/training.log 2>&1 

echo "Job ended at $(date)"
