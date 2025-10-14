#!/bin/bash
set -e  # exit on error

export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
echo "Etc/UTC" > /etc/timezone

apt-get update
apt-get install -y --no-install-recommends ubuntu-keyring
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

# Remove APT caches & temporary files
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
