# Foldy Local Deployment

Run Foldy locally with a single command - no git clone required!

## 🚀 Quick Deployment

### Unix/Linux/macOS (CPU Only)
```bash
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```
*Requires: [Docker](#requirements)*

### Unix/Linux/macOS with GPU
```bash
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  FOLDY_GPU_RUNTIME=nvidia \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```
*Requires: [Docker + GPU Support](#gpu-support)*

### Windows PowerShell (CPU Only)

> We recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and running the Linux commands above.

```powershell
$env:FOLDY_STORAGE_DIRECTORY="$env:USERPROFILE\foldy-data"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml" -OutFile temp-compose.yml
docker-compose -f temp-compose.yml up -d
```
*Requires: [Docker](#requirements)*

### Windows PowerShell with GPU

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

**Linux (Ubuntu/Debian):**
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows:**
- Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- Install [NVIDIA Driver](https://www.nvidia.com/drivers/) for your GPU
- Docker Desktop automatically includes GPU support when NVIDIA drivers are present

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
- `FOLDY_DOCKERHUB_USER=jbrlbl` - DockerHub username for images
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

# Verify access to jbrlbl organization
docker search jbrlbl
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
./build_and_deploy_containers.sh v2.0.8

# Build with multiple tags (version + latest)
./build_and_deploy_containers.sh v2.0.8 latest

# Or use environment variables for custom settings
DOCKERHUB_USER=myusername BACKEND_URL=https://myfoldy.com \
  ./build_and_deploy_containers.sh v2.0.8 latest
```

### 3. Test Production Images
```bash
# Test the newly pushed images
FOLDY_STORAGE_DIRECTORY=/tmp/foldy-test FOLDY_VERSION=v2.1.0 \
  docker-compose up -d

# Verify services are healthy
docker-compose ps
docker-compose logs

# Clean up test
docker-compose down
rm -rf /tmp/foldy-test
```

### 4. Update Version References
```bash
# Update any documentation or scripts that reference the version
# Consider updating the default FOLDY_VERSION in docker-compose.yml if needed
```

## Build Script Options

The `build_and_deploy_containers.sh` script accepts these environment variables:

- `DOCKERHUB_USER` - DockerHub username (default: jbrlbl)
- `DOCKERHUB_REPO_PREFIX` - Image name prefix (default: foldy)
- `BACKEND_URL` - Backend URL for frontend build (default: http://localhost:8080)
- `INSTITUTION` - Institution name for frontend (default: "Foldy Local")

## Docker Images Built

The script builds and pushes these images:

- `jbrlbl/foldy-frontend:VERSION`
- `jbrlbl/foldy-backend:VERSION`
- `jbrlbl/foldy-worker-esm:VERSION`
- `jbrlbl/foldy-worker-boltz:VERSION`

## Testing Checklist

Before releasing, verify:

- [ ] All services start successfully
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend API responds internally
- [ ] Database migrations run successfully
- [ ] Workers can process jobs (test with sample protein)
- [ ] File uploads work correctly
- [ ] Data persists after restart

## Version Tagging Strategy

- Use semantic versioning: `v1.2.3`
- `latest` tag is automatically updated with each push
- Keep `latest` stable - use feature branches for experimental builds
- Document breaking changes in release notes

## Troubleshooting Builds

### Build fails with "no space left on device"
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

### Push fails with authentication error
```bash
# Re-login to DockerHub
docker logout
docker login
```

### Image too large
```bash
# Check image sizes
docker images | grep foldy

# Optimize Dockerfiles if needed
# Consider multi-stage builds to reduce final image size
```
