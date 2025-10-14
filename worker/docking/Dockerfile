FROM continuumio/miniconda3

RUN apt-get -y update
RUN apt-get -y install less

RUN conda install -y openbabel -c conda-forge
RUN conda install -y numpy

RUN mkdir /code
COPY requirements.txt /code/
RUN pip install --no-input -r /code/requirements.txt

# COPY ADFRsuite_x86_64Linux_1.0.tar.gz /
# RUN tar zxvf ADFRsuite_x86_64Linux_1.0.tar.gz
# RUN cd ADFRsuite_x86_64Linux_1.0 && echo Y | ./install.sh -d /adfrsuite -c 0
RUN wget https://ccsb.scripps.edu/adfr/download/1038/ADFRsuite_x86_64Linux_1.0.tar.gz
RUN tar zxvf ADFRsuite_x86_64Linux_1.0.tar.gz && \
  cd ADFRsuite_x86_64Linux_1.0 && \
  echo Y | ./install.sh -d /adfrsuite -c 0


COPY . /code
