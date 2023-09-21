# Helm Deployment

If you are interested in setting up your own Foldy instance, we would love to hear about it and would be happy to help, just send an email to `jbr@lbl.gov`!

Once you are satisfied with the application, you can deploy the application into production by
following the procedure below.


<figure align="center">
  <img src="../../scripts/architecture.png"  />
  <p align="center"><i>Foldy architecture. Kubernetes cluster resources are deployed and managed by Helm. Other resources are deployed and managed manually, instructions below.</i></p>
</figure>

### Initial Setup

This site is built on Kubernetes (specifically [Google Kubernetes Engine, GKE](https://cloud.google.com/kubernetes-engine)).
A few Google Cloud resources need to be created, included a GKE project, and then all
resources within GKE can be deployed at once. The Kuberenetes config, and its resources,
are expressed using a tool called Helm.

Prior to deployment, you must choose the following variables:

- `GOOGLE_PROJECT_ID`: ID for institution google cloud project. Does not need to be foldy specific. Can be retrieved from google cloud console.
- `GKE_CLUSTER_NAME`: Name of kubernetes foldy cluster, typically 'foldy'
- `GOOGLE_SERVICE_ACCOUNT_ID`: Name of service account that foldy uses, typically 'foldy-sa'
- `GOOGLE_SQL_DB_NAME`: Name of SQL database in gke cluster, typically 'foldy-db'
- `GOOGLE_SQL_DB_PASSWORD`: SQL database password in gke cluster, for example use the following command to generate a secure password:

  ```bash
  python -c 'import secrets; print(secrets.token_urlsafe(32))'
  ```

- `FOLDY_DOMAIN`: Domain name selected for foldy application
- `FOLDY_USER_EMAIL_DOMAIN`: Email domain to allow access, e.g. "lbl.gov" will allow all users with "@lbl.gov" email addresses to access
- `GOOGLE_STORAGE_DIRECTORY`: Name of google cloud bucket, for example 'berkeley-foldy-bucket' however it needs to be unique globally like an email address needs to be unique globally
- `GOOGLE_ARTIFACT_REPO`: Name of google cloud docker image repository, typically 'foldy-repo'
- `GOOGLE_CLOUD_STATIC_IP_NAME`: Name of google cloud static IP resource, typically 'foldy-ip'

These variables will be used throughout this procedure. Once completed, execute the following procedure:

1. Clone this repo

   ```bash
   git clone --recurse-submodules https://github.com/JBEI/foldy.git
   cd foldy
   ```

1. Copy the following templates:

   ```bash
   cp deployment/helm/values_template.yaml deployment/helm/values.yaml
   cp deployment/helm/db_creation_resources_template.yaml deployment/helm/db_creation_resources.yaml
   ```

1. Choose a domain! We named our instance `LBL foldy`, and reserved the domain `foldy.lbl.gov` with our IT folks, and we think it reads pretty well. If you don't have an IT team who can provision a domain name / record for you, you can reserve an address like _ourinstitute_-foldy.com using any commercial hostname provider
1. Enable cloud logging API [for prometheus / metrics](https://cloud.google.com/logging/docs/api/enable-api)
1. Install local tools `gcloud`, `helm`, `kubectl`, and `yq`:
   1. Install Google Cloud CLI [[instructions here](https://cloud.google.com/sdk/docs/install-sdk)]
   2. Install Helm CLI [[instructions here](https://helm.sh/docs/intro/install/)], briefly `brew install helm`
   3. Install Kubectl CLI [[instructions here]](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl)). Briefly, make sure you call `gcloud components install kubectl` and `gcloud components install gke-gcloud-auth-plugin`
   4. Install yq [[instructions here](https://github.com/mikefarah/yq)]. Briefly, on mac, you can call `brew install yq`.
1. Create following google cloud resources

   - **Create foldy service account** which has scopes/permissions to access necessary foldy resources
     - From [google cloud console](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating).
     - Make sure to provide following roles:
       - artifact registry administrator
       - artifact registry reader
       - cloud sql client
       - compute admin
       - logging admin
       - monitoring admin
       - storage admin
       - storage object admin
     - Fill in service account details in `deployment/helm/values.yaml`
   - **Create Kubernetes project**

     ```bash
     gcloud container clusters create $GKE_CLUSTER_NAME --enable-managed-prometheus --region=us-central1-c --workload-pool=$GOOGLE_PROJECT_ID.svc.id.goog
     ```

   - **Enable kubectl**

     ```bash
     gcloud container clusters get-credentials $GKE_CLUSTER_NAME
     ```

   - **Create PostgreSQL DB**:

     ```bash
     gcloud sql instances create ${GOOGLE_SQL_DB_NAME} --tier=db-f1-micro --region=us-central1 --storage-size=100GB --database-version=POSTGRES_13 --root-password=${GOOGLE_SQL_DB_PASSWORD}
     ```

     - Then, through the cloud console, enable private IP at `https://console.cloud.google.com/sql/instances/${GOOGLE_SQL_DB_NAME}`, and note the DB IP address as `GOOGLE_SQL_DB_PRIVATE_IP`
     - Now, fill in `DATABASE_URL` in `deployment/helm/values.yaml` using following example: `postgresql://postgres:${GOOGLE_SQL_DB_PASSWORD}@${GOOGLE_SQL_DB_PRIVATE_IP}/postgres`

   - **Allocate Static IP Address**

     - From the [Cloud Console](https://console.cloud.google.com/networking/addresses/list), reserve an external static IP address
     - Make it IPv4, Regional (us-central1, attached to None)

     ```bash
     gcloud compute addresses create ${GOOGLE_CLOUD_STATIC_IP_NAME} --global
     gcloud compute addresses describe ${GOOGLE_CLOUD_STATIC_IP_NAME} --global
     ```

   - **OAuth Client ID**
     - Create OAuth Client ID for production
       - Using the [Google cloud console](https://console.cloud.google.com/apis/credentials).
       - Application type: Web Application
       - Name: `${GKE_CLUSTER_NAME}-prod`
       - Authorized javascript origins: `https://${FOLDY_DOMAIN}`
       - Authorized redirect URIs: `https://${FOLDY_DOMAIN}/api/authorize`
       - Then paste the ID and secret in the `GOOGLE_CLIENT_{ID,SECRET}` fields in `deployment/helm/values.yaml`
   - **Create gcloud bucket** using [cloud console](https://cloud.google.com/storage/docs/creating-buckets) with following attributes:
     - Name = `${GOOGLE_STORAGE_DIRECTORY}`
     - Multi-region
     - Autoclass storage class
     - Prevent public access
     - No object protection
   - **Create gcloud docker image repo** by running:

     ```bash
     gcloud artifacts repositories create ${GOOGLE_ARTIFACT_REPO} --repository-format=docker --location=us-central1
     ```

   - **Enable permission to push and pull images** from artifact registry with:

     ```bash
     gcloud auth configure-docker us-central1-docker.pkg.dev
     ```

   - **Create node pools** by running: `bash scripts/create_nodepools.sh`

1. Fill out template files

   - Fill in `SECRET_KEY` in `deployment/helm/values.yaml` with random secure string, for example use the following command

   ```bash
   python -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

   - `EMAIL_USERNAME` and `EMAIL_PASSWORD` in `deployment/helm/values.yaml` are optional. They will be used for status notifications, but they must be gmail credentials if specified.
   - Fill in variables in `deployment/helm/values.yaml` with appropriate values

1. Install the Keda helm/kubernetes plugin [with docs](https://keda.sh/docs/2.9/deploy/)

1. Bind service account to GKE

   ```bash
   gcloud iam service-accounts add-iam-policy-binding ${GOOGLE_SERVICE_ACCOUNT_ID}@${GOOGLE_PROJECT_ID}.iam.gserviceaccount.com --role roles/iam.workloadIdentityUser --member "serviceAccount:${GOOGLE_PROJECT_ID}.svc.id.goog[default/foldy-ksa]"
   ```

1. Build and push docker images to your google artifact registry with

   ```bash
   bash scripts/build_and_deploy_containers.sh
   ```

1. Make sure that the `ImageVersion` is properly set in `deployment/helm/values.yaml`, then deploy the kubernetes services using

   ```bash
   helm install foldy deployment/helm
   ```

1. Initialize tables in PostgreSQL database

   ```bash
   kubectl exec service/backend -- env FLASK_APP=main.py flask db upgrade
   ```

1. Fill out `db_creation_resources.yaml` with appropriate variables and download alphafold databases into a persistent volume with

   ```bash
   kubectl apply -f db_creation_resources.yaml
   ```

   Can monitor progress of database download with

   ```bash
   kubectl logs --follow --timestamps --previous create-dbs |less
   ```

   **Note, don't run any jobs until database download has been completed.**

1. Reserve a domain name

   - Can use this command to find static IP address

   ```bash
   gcloud compute addresses describe ${GOOGLE_CLOUD_STATIC_IP_NAME} --global
   ```

   - You can add an ANAME record pointing at the static IP address provisioned above.

_Note, using the `us-central1-c` region is required because most google A100s are located in that region._

### Deploying new code

1. Increment `ImageVersion` in `deployment/helm/values.yaml`
1. Rebuild the docker images:

   ```bash
   scripts/build_and_deploy_containers.sh ${PROJECT_ID} ${GOOGLE_ARTIFACT_REPO} ${IMAGE_VERSION}
   ```

1. Update the helm chart `helm upgrade foldy deployment/helm`