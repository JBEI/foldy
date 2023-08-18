#!/bin/bash

set -e -o xtrace

if [ "$#" -ne 0 ]; then
  echo "Must specify GKE_CLUSTER_NAME and service account, like create_nodepools.sh \$GKE_CLUSTER_NAME \$GOOGLE_SERVICE_ACCOUNT_ID@\$GOOGLE_PROJECT_ID.iam.gserviceaccount.com"
  exit 1
fi

GOOGLE_PROJECT_ID=$(yq eval '.GoogleProjectId' deployment/helm/values.yaml -e)
GKE_CLUSTER_NAME=$(yq eval '.GkeClusterId' deployment/helm/values.yaml -e)
GOOGLE_SERVICE_ACCOUNT_ID=$(yq eval '.ServiceAccount' deployment/helm/values.yaml -e)

SERVICE_ACCOUNT_FULL_ADDRESS="$GOOGLE_SERVICE_ACCOUNT_ID@$GOOGLE_PROJECT_ID.iam.gserviceaccount.com"

echo "Using GKE Cluster ID: $GKE_CLUSTER_ID"
echo "Using Google Service Account: $SERVICE_ACCOUNT_FULL_ADDRESS"

gcloud container node-pools create generalnodes \
  --zone us-central1-c \
  --cluster $GKE_CLUSTER_NAME \
  --num-nodes 0 --min-nodes 0 --max-nodes 3 --enable-autoscaling \
  --spot \
  --machine-type e2-standard-2 \
  --service-account $SERVICE_ACCOUNT_FULL_ADDRESS

# Prevent GKE kube-system pods from running on our compute nodes by adding a taint:
# https://cloud.google.com/kubernetes-engine/docs/how-to/isolate-workloads-dedicated-nodes

gcloud container node-pools create spothighmemnodes \
  --zone us-central1-c \
  --cluster $GKE_CLUSTER_NAME \
  --num-nodes 0 --min-nodes 0 --max-nodes 10 --enable-autoscaling \
  --spot \
  --node-taints computenode=true:NoSchedule \
  --machine-type e2-highmem-8 \
  --service-account $SERVICE_ACCOUNT_FULL_ADDRESS

# Instructions to set up the autoscaling A100 nodepools:
# https://cloud.google.com/kubernetes-engine/docs/how-to/gpus

# We're using A100s with 80GB memory
# https://cloud.google.com/compute/docs/gpus#a100-80gb

gcloud container node-pools create spota100nodes \
  --accelerator type=nvidia-a100-80gb,count=1 \
  --zone us-central1-c \
  --cluster $GKE_CLUSTER_NAME \
  --num-nodes 0 --min-nodes 0 --max-nodes 3 --enable-autoscaling \
  --spot \
  --node-taints computenode=true:NoSchedule \
  --machine-type a2-ultragpu-1g \
  --service-account $SERVICE_ACCOUNT_FULL_ADDRESS
