# Foldy Backend

## Development on Apple Silicon

To install backend requirements for python code complete on apple silicon:

```bash
CONDA_SUBDIR=osx-64 conda create -n foldy-env
conda activate foldy-env
conda install python=3.12
pip install -r backend/requirements.txt
```
