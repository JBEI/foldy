#!/bin/bash
set -e  # exit on error

export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
echo "Etc/UTC" > /etc/timezone

# Install Rust.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 4) "conda init" is typically for interactive shells. In Docker builds,
#    you can often just add /opt/conda/bin to PATH or run /opt/conda/bin/conda directly.
#    But if you do want it available for interactive shells in the container, keep it.
/opt/conda/bin/conda init bash

# Accept Terms of Service for all channels to avoid interactive prompts
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# 5) Install all GPU dependencies with retry logic
for i in {1..3}; do
  /opt/conda/bin/conda create -y -n worker \
    python=3.12                             \
    pytorch=2.2.*                           \
    torchvision=0.17.*                      \
    torchaudio=2.2.*                        \
    pytorch-cuda=12.1                       \
    gpytorch=1.14                           \
    botorch=0.14.*                          \
    linear_operator=0.6                     \
    pyro-ppl>=1.8.4                         \
    -c pytorch -c nvidia -c gpytorch -c conda-forge && break || {
      echo "Attempt $i failed, retrying in 10 seconds..."
      sleep 10
    }
done

# /opt/conda/bin/conda create -y -n worker \
#     python=3.12 \
#     cudatoolkit=11.8 \
#     pytorch-cuda=12.1 \
#     pytorch \
#     torchvision \
#     torchaudio \
#     -c pytorch -c nvidia -c conda-forge
# 6) Clean up conda cache
/opt/conda/bin/conda clean -afq

# 7) Install UV for faster pip operations
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 8) Install pip packages for your worker environment using UV
uv pip install --python /opt/conda/envs/worker/bin/python pip-tools
uv pip install --python /opt/conda/envs/worker/bin/python --no-cache --no-deps -r /backend/requirements.txt

# Clean conda & pip
/opt/conda/bin/conda clean -a -y
rm -rf /opt/conda/pkgs
rm -rf /root/.cache/pip

# Clean Rust if not needed
rm -rf /root/.cargo /root/.rustup
