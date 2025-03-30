#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 8                      # total number of tasks
#SBATCH --mem=64g                  # allocate 1 GB of memory
#SBATCH -J "abhay_train"      # name of the job
#SBATCH -o abhay_train_%j.out # name of the output file
#SBATCH -e abhay_train_%j.err # name of the error file
#SBATCH -p academic                  # partition to submit to
#SBATCH -t 15:00:00               # time limit of 1 hour
#SBATCH --gres=gpu:1              # request 1 GPU

module load python/3.11.10               # These version were chosen for compatability with pytorch
module load cuda/12.4.0/3mdaov5          # load CUDA (adjust if necessary)

python3 -m venv venv    # create virtual environment
source ./venv/bin/activate   # activate virtual environment
pip3 install -r requirements.txt     
python3 converted.py                # run Python script
