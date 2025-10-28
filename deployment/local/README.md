# Foldy Local Deployment

Run Foldy locally with a single command - no git clone required!

## 🚀 Quick Deployment

<details>
<summary>Unix/Linux/macOS (CPU Only)</summary>

```bash
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```
*Requires: [Docker](#requirements)*

</details>

<details>
<summary>Unix/Linux/macOS with GPU</summary>

```bash
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  FOLDY_GPU_RUNTIME=nvidia \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```
*Requires: [Docker + GPU Support](#gpu-support)*

</details>

<details>
<summary>Windows PowerShell (CPU Only)</summary>

> We recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and running the Linux commands above.

```powershell
$env:FOLDY_STORAGE_DIRECTORY="$env:USERPROFILE\foldy-data"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml" -OutFile temp-compose.yml
docker-compose -f temp-compose.yml up -d
```
*Requires: [Docker](#requirements)*

</details>

<details>
<summary>Windows PowerShell with GPU</summary>

> We recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and running the Linux commands above.

```powershell
$env:FOLDY_STORAGE_DIRECTORY="$env:USERPROFILE\foldy-data"
$env:FOLDY_GPU_RUNTIME="nvidia"
$env:NVIDIA_VISIBLE_DEVICES="all"
$env:NVIDIA_DRIVER_CAPABILITIES="compute,utility"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml" -OutFile temp-compose.yml
docker-compose -f temp-compose.yml up -d
```
*Requires: [Docker + GPU Support](#gpu-support)*

</details>

**That's it!** Foldy will be available at **http://localhost:3000**

---

## Requirements

### Basic Requirements
- Docker and Docker Compose installed
- Internet connection for initial setup

### GPU Support

To use GPU acceleration, you need:
1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Container Toolkit** installed on your system

#### Installing NVIDIA Container Toolkit

[Instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Verify GPU support:**
```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi
```

---

## Configuration

### Required Environment Variables
- `FOLDY_STORAGE_DIRECTORY` - Path where Foldy will store all persistent data

### Optional Environment Variables
- `FOLDY_VERSION=latest` - Docker image version to use
- `FOLDY_DOCKERHUB_USER=keasling` - DockerHub username for images
- `FOLDY_GPU_RUNTIME=nvidia` - Enable GPU acceleration (requires NVIDIA Container Toolkit)
- `NVIDIA_VISIBLE_DEVICES=all` - Which GPUs to use (when GPU enabled)
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - NVIDIA driver capabilities (when GPU enabled)

## Data Storage

Your `FOLDY_STORAGE_DIRECTORY` will contain:

```
foldy-data/
├── postgres_data/    # PostgreSQL database files
├── blob_storage/     # Protein structures, results, uploads
└── boltz_cache/      # Cached Boltz model files
```

## Management Commands

```bash
# Stop Foldy
docker-compose down

# View logs
docker-compose logs -f

# Update to latest version
docker-compose pull && docker-compose up -d

# Restart specific service
docker-compose restart backend

# Database shell
docker-compose exec db psql -U user -d postgres

# Backend shell
docker-compose exec backend bash
```

## Troubleshooting

### Services fail to start
1. Check Docker is running: `docker info`
2. Ensure `FOLDY_STORAGE_DIRECTORY` exists and is writable
3. Try full restart: `docker-compose down && docker-compose up -d`

### Database connection issues
- Wait for PostgreSQL to initialize (may take a minute)
- Check logs: `docker-compose logs db`
- Verify data directory permissions

### GPU issues
- Verify GPU support: `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi`
- Check NVIDIA Container Toolkit installation
- Ensure GPU drivers are up to date

---

# 🛠 Developer Guide

## Setup

### 1. DockerHub Access
```bash
# Login to DockerHub (required for pushing images)
docker login

# Verify access to keasling organization
docker search keasling
```

### 2. Repository Setup
```bash
# Clone repository
git clone https://github.com/JBEI/foldy.git
cd foldy

# Ensure you're on the right branch
git checkout main
git pull origin main
```

## Release Process

### 1. Test Development Build
```bash
# Test with development compose first
docker-compose -f deployment/development/docker-compose.yml up -d
# ... verify everything works ...
docker-compose -f deployment/development/docker-compose.yml down
```

### 2. Build and Push Release
```bash
cd deployment/local

# Build and push to DockerHub (replace with your version)
./deployment/local/build_and_deploy_containers.sh v2.0.8

# Build with multiple tags (version + latest)
./deployment/local/build_and_deploy_containers.sh v2.0.8 latest

# Or if you want to host your own images, you can specify overrides:
DOCKERHUB_USER=myusername BACKEND_URL=https://myfoldy.com \
  ./deployment/local/build_and_deploy_containers.sh v2.0.8 latest
```

## Docker Images Built

The script builds and pushes these images:

- `keasling/foldy-frontend:VERSION`
- `keasling/foldy-backend:VERSION`
- `keasling/foldy-worker-esm:VERSION`
- `keasling/foldy-worker-boltz:VERSION`


## Version Tagging Strategy

- Use semantic versioning: `v1.2.3`
- `latest` tag should be updated with each push
- Keep `latest` stable - use feature branches for experimental builds
