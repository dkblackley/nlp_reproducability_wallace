#!/bin/bash
#SBATCH --job-name=nlp_reprod
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --output=gpu_test-%j.out
#SBATCH --error=gpu_test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Load cuda and conda
module load cuda
module load miniconda3
eval "$(conda shell.bash hook)"

conda remove -n poison --all -y || true

# Create base environment with python first
echo "Creating base conda environment..."
conda create -n poison python=3.8 -y

echo "Activating conda environment..."
conda activate poison

# Install GPU PyTorch first (has most packages)
echo "Installing PyTorch with CUDA support..."
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install core packages
echo "Installing core packages..."
conda install -y -c conda-forge transformers spacy

# Install pip packages with corrected versions
echo "Installing remaining packages..."
pip install fsspec==2024.6.1  # Downgraded to meet datasets requirement
pip install huggingface-hub==0.26.2
pip install regex==2024.11.6
pip install safetensors==0.4.5
pip install tokenizers==0.20.3
pip install transformers==4.46.2
pip install wandb  # Added wandb
pip install datasets  # Added datasets explicitly
pip install sentencepiece
pip install scipy
pip install seaborn

# Install spacy model extras
python -m spacy download en_core_web_sm

# Verify CUDA is available
echo "=== CUDA Information ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python3 /scratch/dblackle/tmp/nlp_reproducability_wallace/src/main.py
