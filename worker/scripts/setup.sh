#!/bin/bash
set -e  # exit on error

export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
echo "Etc/UTC" > /etc/timezone

apt-get update 
apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    wget \
    curl \
    bzip2 \
    git \
    tree \
    vim \
    aria2 \
    rsync

# Install Rust.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# # Install the official CUDA keyring .deb
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# rm cuda-keyring_1.0-1_all.deb

# # Now 'apt-get update' should succeed, trusting NVIDIA's repo signatures.
# apt-get update

# 3) Install Miniconda
#    (You can also verify the Miniconda installer with sha256sum if you want extra security.)
wget --no-check-certificate \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh

# 4) "conda init" is typically for interactive shells. In Docker builds,
#    you can often just add /opt/conda/bin to PATH or run /opt/conda/bin/conda directly.
#    But if you do want it available for interactive shells in the container, keep it.
/opt/conda/bin/conda init bash

# 5) Create a new environment and install your packages in one or two steps.
#    Using a single conda command can be faster and keeps the environment fully solvable at once.
#    Also, consider the 'mamba' package (from conda-forge) for faster solves.
#
#    -n worker means environment name is “worker”.
#    -c pytorch -c nvidia -c conda-forge includes those channels.
#    You can pin Python version here as well.
#
/opt/conda/bin/conda create -y -n worker \
    python=3.12 \
    cudatoolkit=11.8 \
    pytorch-cuda=12.1 \
    pytorch \
    torchvision \
    torchaudio \
    -c pytorch -c nvidia -c conda-forge

# 6) Clean up conda cache
/opt/conda/bin/conda clean -a -y

# 7) Install pip packages for your worker environment
/opt/conda/envs/worker/bin/pip install --no-cache-dir -r /backend/requirements.txt

rm -rf /var/lib/apt/lists/*