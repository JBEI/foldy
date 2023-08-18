
# Developer Deployment / Local Deployment

To deploy Foldy locally for development or testing purposes, you can follow these steps:

### Prerequisites

Make sure you have the following tools installed on your machine:

- Docker (for running the local deployment)
- Conda (for creating the environment and installing dependencies)

### Backend and Databases

1. Start by setting up the backend and databases using Docker Compose. Open a terminal window and navigate to the project directory.

2. Run the following command to bring up the backend and databases:

   ```bash
   docker compose \
     --file deployment/development/docker-compose.yml \
     --project-directory . \
     up
   ```

    This command will start the frontend & backend servers and the required databases. Note that it takes a few minutes for the images to build, and then a few more minutes for the frontend to start up (the logs will say "Starting the development server...", then it takes ~60 seconds to build the frontend, then it will say "No issues found.").

4. Install the databases by running the following command:

   ```bash
   docker compose \
     --file deployment/development/docker-compose.yml \
     --project-directory . \
     exec backend flask db upgrade
   ```

5. Access the Foldy website at http://localhost:3000 in your browser.

That's it! You have successfully deployed Foldy locally for development or testing purposes. Explore the website and make any necessary changes to the codebase.


### Upgrading Database in development

If any changes are made to the database models, execute the following commands to create a revision and migrate database

```bash
docker compose exec \
     --file deployment/development/docker-compose.yml \
     --project-directory . \
     backend flask db stamp $CURRENT_REVISION_NUMBER
docker compose exec \
     --file deployment/development/docker-compose.yml \
     --project-directory . \
     backend flask db migrate
docker compose exec \
     --file deployment/development/docker-compose.yml \
     --project-directory . \
     backend flask db upgrade
```

### Development Tasks

> Protein structure prediction tasks in the development environment are not actually performed.
> Instead a few test cases have been precomputed that auto-complete once queued.