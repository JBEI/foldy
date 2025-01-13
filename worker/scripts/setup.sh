
# Set the environment variable for non-interactive installation
export DEBIAN_FRONTEND=noninteractive

# Preconfigure the time zone to avoid interactive prompts
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone

# Install basic utilities and dependencies.
apt-get -y update
apt-get -y install tree vim curl wget bzip2 git aria2 rsync
rm -rf /var/lib/apt/lists/*

# Install Miniconda.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh
/opt/conda/bin/conda init bash

# Add Conda to PATH.
export PATH="/opt/conda/bin:$PATH"

# Install backend requirements.
conda create -y -n worker python=3.12
conda install -y -n worker cudatoolkit=11.8 pytorch-cuda=12.1 pytorch torchvision torchaudio -c pytorch -c nvidia -c conda-forge
conda clean -a -y