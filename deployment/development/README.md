# Development Environment Setup

This guide provides complete instructions for setting up a local development environment for Foldy.

## Prerequisites

Ensure you have the following tools installed:

- **Git** - Version control
- **Docker** - For backend services and databases
- **Python 3.12** - Backend development
- **Node.js** (via nvm) - Frontend development

## Quick Start

For experienced developers, here's the minimal setup:

```bash
# 1. Python environment
python3.12 -m venv .venv
source .venv/bin/activate
cd backend && pip install -e ".[dev]"

# 2. Pre-commit hooks
pre-commit install --install-hooks
pre-commit install -t pre-push

# 3. Node.js environment
cd ../frontend && npm install

# 4. Start development environment
cd .. && DOCKER_BUILDKIT=1 docker compose --file deployment/development/docker-compose.yml --project-directory . up
```

## Detailed Setup Instructions

### 1. Python Environment Setup

#### Install Python 3.12

- **macOS with Homebrew**: `brew install python@3.12`
- **Ubuntu/Debian**: `sudo apt install python3.12 python3.12-venv`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

#### Create Virtual Environment

From the project root directory:

```bash
# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Verify Python version
python --version  # Should output Python 3.12.x
```

#### Install Python Dependencies

```bash
cd backend
pip install -e ".[dev]"
```

This installs the project in editable mode with all development dependencies defined in `pyproject.toml`.

### 2. Node.js and npm Setup

#### Install nvm (Node Version Manager)

**macOS/Linux:**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# Restart terminal or run:
source ~/.bashrc
```

**Windows:**
Download nvm-windows from [github.com/coreybutler/nvm-windows](https://github.com/coreybutler/nvm-windows)

#### Install and Use Node.js

```bash
# Install latest LTS Node.js
nvm install --lts
nvm use --lts

# Verify installation
node --version
npm --version
```

#### Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Code Quality Tools Setup

#### Pre-commit Hooks

Pre-commit hooks ensure code quality by running:
- **pyright** - Type checking
- **black** - Code formatting
- **isort** - Import sorting
- **Basic file checks** - Trailing whitespace, large files, etc.

```bash
# Install pre-commit hooks (from project root with .venv activated)
pre-commit install --install-hooks
pre-commit install -t pre-push

# Test installation (optional)
pre-commit run --all-files
```

#### Configuration Files

The setup uses these configuration files:
- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `pyrightconfig.json` - Type checking configuration
- `backend/pyproject.toml` - Python project configuration (black, isort settings)

### 4. Development Environment

#### Start Backend Services

```bash
# From project root
DOCKER_BUILDKIT=1 docker compose --file deployment/development/docker-compose.yml --project-directory . up
```

This starts:
- Backend Flask application
- Frontend development server (Vite)
- PostgreSQL database
- Redis cache

**Note**: Initial startup takes several minutes for image building and frontend compilation.

#### Initialize Database

```bash
# Run database migrations
docker compose --file deployment/development/docker-compose.yml --project-directory . exec backend python -m flask db upgrade
```

#### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

### 5. Development Workflow

#### Type Checking

Run type checking manually:

```bash
# Check specific file
/Users/jacobroberts/git/foldy/.venv/bin/pre-commit run pyright backend/app/models.py

# Check all files
/Users/jacobroberts/git/foldy/.venv/bin/pre-commit run pyright --all-files
```

#### Database Migrations

When modifying database models:

```bash
# Create migration
docker compose --file deployment/development/docker-compose.yml --project-directory . exec backend /opt/conda/envs/worker/bin/python -m flask db migrate

# Apply migration
docker compose --file deployment/development/docker-compose.yml --project-directory . exec backend /opt/conda/envs/worker/bin/python -m flask db upgrade
```

#### Testing

**Backend Tests:**
```bash
# Run all tests
docker compose --file deployment/development/test-docker-compose.yml --project-directory . up

# Run specific test
docker compose --file deployment/development/docker-compose.yml --project-directory . exec backend pytest app/tests/test_file.py::TestClass::test_method -v
```

**Frontend Tests:**
```bash
cd frontend
npm test
```

### 6. Development Features

#### Test Data

The development environment includes pre-computed test cases for protein folding:

**Example Protein:**
- **Name**: Any name
- **Sequence**: `MEHLYLSLVLLFVSSISLSLFFLFYKHKSMFTGANLPPGKIGYPLIGESLEFLSTGWKGHPEKFIFDRMSKYSSQIFKTSILGEPTAVFPGAVCNKFLFSNENKLVNAWWPASVDKIFPSSLQTSSKEEAKKMRKLLPQFLKPEALHRYIGIMDSIAQRHFADSWENKNQVIVFPLAKRYTFWLACRLFISVEDPTHVSRFADPFQLLAAGIISIPIDLPGTPFRKAINASQFIRKELLAIIRQRKIDLGEGKASPTQDILSHMLLTCDENGQYMNELDIADKILGLLVGGHDTASAACTFVVKFLAELPHIYEQVYKEQMEIAKSKVPGELLNWEDIQKMKYSWNVACEVMRLAPPLQGAFREAITDFVFNGFSIPKGWKLYWSANSTHKSPDYFPEPDKFDPTRFEGNGPAPYTFVPFGGGPRMCPGKEYARLEILVFMHNLVKRFKWEKLVPDEKIVVDPMPIPAKGLPVRLYPHKA`
- **Dock Name**: nadhd2
- **Dock Tool**: diffdock
- **SMILES**: Any valid SMILES string

#### Hot Reloading

Both frontend and backend support hot reloading during development:
- **Frontend**: Vite automatically reloads on file changes
- **Backend**: Flask development server reloads on Python file changes

## Troubleshooting

### Python/Type Checking Issues

```bash
# Clear pre-commit cache
pre-commit clean
pre-commit gc

# Reinstall hooks
pre-commit install --install-hooks

# Verify Python version in virtual environment
python --version
```

### Docker Issues

```bash
# Rebuild containers
DOCKER_BUILDKIT=1 docker compose --file deployment/development/docker-compose.yml --project-directory . build --no-cache

# View logs
docker compose --file deployment/development/docker-compose.yml --project-directory . logs
```

### Node.js Issues

```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

## Code Style Guidelines

- **Python**: Type hints required, follow PEP 8, snake_case naming
- **TypeScript**: Strict typing, camelCase naming, React functional components
- **Testing**: Descriptive test names, arrange-act-assert pattern
- **Commits**: Pre-commit hooks enforce code quality

## Next Steps

After setup:
1. Explore the codebase in `backend/` and `frontend/src/`
2. Try creating a test protein fold at http://localhost:3000
3. Review the [architecture documentation](../../docs/architecture.md)
4. Check out the [interface documentation](../../docs/interface.md)

For deployment options beyond development, see the other deployment guides in the `deployment/` directory.
