# We use the same CUDA image in all our Dockerfiles.
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

COPY worker/scripts/base_setup.sh /scripts/base_setup.sh
RUN /scripts/base_setup.sh

COPY backend/requirements.txt /backend/
COPY worker/scripts/conda_env_setup.sh /scripts/conda_env_setup.sh
RUN /scripts/conda_env_setup.sh

ENV APP_HOME /backend
WORKDIR $APP_HOME

RUN /opt/conda/envs/worker/bin/pip install gunicorn
# Copy local code to the container image.
COPY backend/ ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD flask rq worker --burst all
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app_dispatch
CMD exec /opt/conda/envs/worker/bin/gunicorn --bind :$PORT --workers 3 --threads 3 --timeout 60 --log-level=debug --keep-alive 30 main:app_dispatch
