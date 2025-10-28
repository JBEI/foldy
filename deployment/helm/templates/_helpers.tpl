{{/*
Expand the name of the chart.
*/}}
{{- define "foldy.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "foldy.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "foldy.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "foldy.labels" -}}
helm.sh/chart: {{ include "foldy.chart" . }}
{{ include "foldy.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "foldy.selectorLabels" -}}
app.kubernetes.io/name: {{ include "foldy.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "foldy.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "foldy.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}


{{/*
Create a ScaledJob to process RQ tasks from the specified queue.

The passed context must include:
  .RqQueueName: the name of the RQ pool to listen on.

Usage example:
{{- include "foldy.createWorkerPool" (merge . (dict "RqQueueName" "emailparrot")) -}}

*/}}

{{- define "foldy.createWorkerPool" }}
# Scaling jobs: https://keda.sh/docs/2.8/concepts/scaling-jobs/
# Prometheus triggers: https://keda.sh/docs/2.8/scalers/prometheus/

apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: worker-{{ required "RqQueueName is required." .RqQueueName }}
  namespace: default
spec:
  # pollingInterval: 30                         # Optional. Default: 30 seconds
  successfulJobsHistoryLimit: 20               # Optional. Default: 100. How many completed jobs should be kept.
  failedJobsHistoryLimit: 20                   # Optional. Default: 100. How many failed jobs should be kept.
  # envSourceContainerName: {container-name}    # Optional. Default: .spec.JobTargetRef.template.spec.containers[0]
  minReplicaCount: 0                          # Optional. Default: 0
  maxReplicaCount: {{ if (eq .RqQueueName "cpu") -}}
    10
  {{- else -}}
    3
  {{- end }}
  rollout:
    strategy: gradual                         # Optional. Default: default. Which Rollout Strategy KEDA will use.
    # propagationPolicy: foreground             # Optional. Default: background. Kubernetes propagation policy for cleaning up existing jobs during rollout.
  scalingStrategy:
    strategy: "accurate"                        # Optional. Default: default. Which Scaling Strategy to use.
    # customScalingQueueLengthDeduction: 1      # Optional. A parameter to optimize custom ScalingStrategy.
    # customScalingRunningJobPercentage: "0.5"  # Optional. A parameter to optimize custom ScalingStrategy.
    # pendingPodConditions:                     # Optional. A parameter to calculate pending job count per the specified pod conditions
    #   - "Ready"
    #   - "PodScheduled"
    #   - "AnyOtherCustomPodCondition"
    # multipleScalersCalculation : "max" # Optional. Default: max. Specifies how to calculate the target metrics when multiple scalers are defined.
  triggers:
  - type: prometheus
    metadata:
      # Specify the namespace in the server hostname, because the querying pod will be in the "keda" namespace:
      # https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/
      serverAddress: http://prom-frontend.default:9090
      metricName: size_{{ required "RqQueueName is required." .RqQueueName }}_queue # Note: name to identify the metric, generated value would be `prometheus-http_requests_total`
      # Filter to only metrics coming from this cluster.
      query: MAX(size_{{ required "RqQueueName is required." .RqQueueName }}_queue{cluster="{{ required "GkeClusterId is required" .Values.GkeClusterId }}"}) #sum(rate(http_requests_total{deployment="my-deployment"}[2m])) # Note: query must return a vector/scalar single element response
      threshold: '0.5'
      # activationThreshold: '5.5'
      # Optional fields:
      # namespace: example-namespace  # for namespaced queries, eg. Thanos
      # cortexOrgId: my-org # Optional. X-Scope-OrgID header for Cortex.
      # ignoreNullValues: false # Default is `true`, which means ignoring the empty value list from Prometheus. Set to `false` the scaler will return error when Prometheus target is lost

  jobTargetRef:
    parallelism: 1                             # [max number of desired pods](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#controlling-parallelism)
    completions: 1                             # [desired number of successfully finished pods](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#controlling-parallelism)
    # activeDeadlineSeconds: 600                 #  Specifies the duration in seconds relative to the startTime that the job may be active before the system tries to terminate it; value must be positive integer
    backoffLimit: 3                            # Specifies the number of retries before marking this job failed. Defaults to 6

    # Uncomment to keep each Job for 24 h after it completes
    ttlSecondsAfterFinished: 86400

    template:
      spec:
        serviceAccountName: foldy-ksa

        restartPolicy: Never
        # allow 25 for clean shutdown. Spot nodes are terminated with only 30s notice...
        terminationGracePeriodSeconds: 25

        volumes:
          - name: foldydbs
            persistentVolumeClaim:
              claimName: foldydbs
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 20Gi

        nodeSelector:
          iam.gke.io/gke-metadata-server-enabled: "true"
        {{- if or (eq .RqQueueName "gpu") (eq .RqQueueName "biggpu") }}
          cloud.google.com/gke-nodepool: spota100nodes
        {{- else if or (eq .RqQueueName "esm") (eq .RqQueueName "boltz") }}
          cloud.google.com/gke-nodepool: ondemanda100nodes
        {{- else if (eq .RqQueueName "cpu") }}
          cloud.google.com/gke-nodepool: spothighmemnodes
        {{- end }}

        # We must allow this job to be run on our compute nodepools.
        tolerations:
        - key: "computenode"
          operator: "Exists"
          effect: "NoSchedule"

        containers:
        - name: master
          image: {{ .Values.GoogleCloudRegion }}-docker.pkg.dev/{{ .Values.GoogleProjectId }}/{{ .Values.ArtifactRepo }}/{{ required "image name is required" .ImageName }}:{{  .Values.ImageVersion }}
          command: ["/opt/conda/envs/worker/bin/python"]
          args: ["/backend/rq_worker_main.py", {{ required "RqQueueName is required." .RqQueueName | quote }}, "--burst", "--max-jobs", "1"]
          env:
          - name: FLASK_APP
            value: rq_worker_main.py
          - name: RUN_ANNOTATE_PATH
            value: /worker/run_annotate.sh
          - name: RUN_DOCK
            value: /worker/run_dock.sh
          - name: NODE_NAME
            valueFrom:
              fieldRef:
                fieldPath: spec.nodeName
          - name: NODE_IP
            valueFrom:
              fieldRef:
                fieldPath: status.hostIP
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
          - name: POD_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
          envFrom:
          - configMapRef:
              name: foldy-configmap
          - secretRef:
              name: foldy-secret
          volumeMounts:
          - mountPath: "/foldydbs"
            name: foldydbs
          - mountPath: /dev/shm
            name: dshm  # for example
          resources:
            {{- if eq .RqQueueName "emailparrot" }}
            requests:
              cpu: 100m
              memory: 100Mi
            {{- else if (eq .RqQueueName "cpu") }}
            requests:
              cpu: 15000m
              memory: 115Gi
            {{- else }}
            requests:
              cpu: 10000m
              memory: 150Gi
              nvidia.com/gpu: 1
            limits:
              nvidia.com/gpu: 1
            {{- end }}

{{ end -}}
