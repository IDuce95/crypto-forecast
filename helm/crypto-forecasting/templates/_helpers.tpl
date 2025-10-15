{{/*
Expand the name of the chart.
*/}}
{{- define "crypto-forecasting.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "crypto-forecasting.fullname" -}}
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
{{- define "crypto-forecasting.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "crypto-forecasting.labels" -}}
helm.sh/chart: {{ include "crypto-forecasting.chart" . }}
{{ include "crypto-forecasting.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "crypto-forecasting.selectorLabels" -}}
app.kubernetes.io/name: {{ include "crypto-forecasting.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "crypto-forecasting.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "crypto-forecasting.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Backend selector labels
*/}}
{{- define "crypto-forecasting.backend.selectorLabels" -}}
{{ include "crypto-forecasting.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Frontend selector labels
*/}}
{{- define "crypto-forecasting.frontend.selectorLabels" -}}
{{ include "crypto-forecasting.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Airflow selector labels
*/}}
{{- define "crypto-forecasting.airflow.selectorLabels" -}}
{{ include "crypto-forecasting.selectorLabels" . }}
app.kubernetes.io/component: airflow
{{- end }}

{{/*
Database URL
*/}}
{{- define "crypto-forecasting.databaseURL" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password .Release.Name .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.app.env.DATABASE_URL }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "crypto-forecasting.redisURL" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://%s-redis-master:6379/1" .Release.Name }}
{{- else }}
{{- .Values.app.env.REDIS_URL }}
{{- end }}
{{- end }}

{{/*
API URL for frontend
*/}}
{{- define "crypto-forecasting.apiURL" -}}
{{- if .Values.ingress.enabled }}
{{- $host := index .Values.ingress.hosts 0 }}
{{- printf "https://%s/api" $host.host }}
{{- else }}
{{- printf "http://%s-backend:5000" (include "crypto-forecasting.fullname" .) }}
{{- end }}
{{- end }}
