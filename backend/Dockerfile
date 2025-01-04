# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM --platform=linux/amd64 python:3.12

ENV APP_HOME /src
WORKDIR $APP_HOME

RUN apt-get update && apt-get install vim -y

# Install production dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY src/ ./


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD flask rq worker --burst all
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app_dispatch
CMD exec gunicorn --bind :$PORT --workers 1 --threads 3 --timeout 0 --log-level=debug --keep-alive 30 main:app_dispatch
