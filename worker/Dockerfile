FROM foldyalphafold


WORKDIR /
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
# Install curl for this script, and aria2 and rsync for AF DB downloading.
RUN apt-get -y update && apt-get -y install curl aria2 rsync && \
  curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-401.0.0-linux-x86_64.tar.gz && \
  tar -xf google-cloud-cli-401.0.0-linux-x86_64.tar.gz && \
  ./google-cloud-sdk/install.sh -q && \
  source /google-cloud-sdk/path.bash.inc

RUN mkdir /backend && mkdir -p /worker/docking

# Install backend requirements.
COPY backend/requirements.txt /backend/
RUN conda create -n worker python=3.7
RUN /opt/conda/envs/worker/bin/pip install -r /backend/requirements.txt

# Install docking requirements.
WORKDIR /worker/docking
COPY worker/docking/requirements.txt ./
RUN /opt/conda/bin/conda create -y -n dock conda python==3.7 && \
  /opt/conda/envs/dock/bin/conda install -y -c conda-forge openbabel numpy && \
  /opt/conda/envs/dock/bin/pip install --no-input -r requirements.txt
RUN wget https://ccsb.scripps.edu/adfr/download/1038/ADFRsuite_x86_64Linux_1.0.tar.gz
RUN tar zxvf ADFRsuite_x86_64Linux_1.0.tar.gz && \
  cd ADFRsuite_x86_64Linux_1.0 && \
  echo Y | ./install.sh -d /adfrsuite -c 0

# Install decompress pkls requirements.
RUN /opt/conda/envs/worker/bin/pip install --no-input install scipy

# Install annotation requirements.
WORKDIR /annotations
RUN /opt/conda/bin/conda config --add channels defaults && \
  /opt/conda/bin/conda config --add channels bioconda && \
  /opt/conda/bin/conda config --add channels conda-forge
RUN /opt/conda/bin/conda create -y -n annotations python=3.9 conda java-jdk
# TODO: maybe switch to muscle version 3.8.31, which we have elsewhere,
# but did not install properly here.
RUN /opt/conda/envs/annotations/bin/conda install -y hmmer2 hmmer diamond fasttree prodigal blast muscle=3.8.1551 glimmerhmm

# Copy in code.
COPY backend/ /backend/
COPY worker/*.sh /worker/
COPY worker/*.py /worker/
COPY worker/docking/* /worker/docking/

ENTRYPOINT /opt/conda/envs/worker/bin/flask