#!/bin/bash

set -e -o xtrace

if [ "$#" -ne 2 ]; then
  echo "Must specify GKE_CLUSTER_NAME and service account, like create_nodepools.sh \$GKE_CLUSTER_NAME \$GOOGLE_SERVICE_ACCOUNT_ID@\$GOOGLE_PROJECT_ID.iam.gserviceaccount.com"
  exit 1
fi

gcloud container node-pools create generalnodes \
  --zone us-central1-c \
  --cluster $1 \
  --num-nodes 0 --min-nodes 0 --max-nodes 3 --enable-autoscaling \
  --spot \
  --machine-type e2-standard-2 \
  --service-account $2

# Prevent GKE kube-system pods from running on our compute nodes by adding a taint:
# https://cloud.google.com/kubernetes-engine/docs/how-to/isolate-workloads-dedicated-nodes

gcloud container node-pools create spothighmemnodes \
  --zone us-central1-c \
  --cluster $1 \
  --num-nodes 0 --min-nodes 0 --max-nodes 10 --enable-autoscaling \
  --spot \
  --node-taints computenode=true:NoSchedule \
  --machine-type e2-highmem-8 \
  --service-account $2

# Instructions to set up the autoscaling A100 nodepools:
# https://cloud.google.com/kubernetes-engine/docs/how-to/gpus

# We're using A100s with 80GB memory
# https://cloud.google.com/compute/docs/gpus#a100-80gb

gcloud container node-pools create spota100nodes \
  --accelerator type=nvidia-a100-80gb,count=1 \
  --zone us-central1-c \
  --cluster $1 \
  --num-nodes 0 --min-nodes 0 --max-nodes 3 --enable-autoscaling \
  --spot \
  --node-taints computenode=true:NoSchedule \
  --machine-type a2-ultragpu-1g \
  --service-account $2
