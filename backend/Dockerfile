# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7.13

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY backend/ ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD flask rq worker --burst all
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app_dispatch
CMD exec gunicorn --bind :$PORT --workers 1 --threads 3 --timeout 0 --log-level=debug --keep-alive 30 main:app_dispatch
