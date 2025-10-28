# FolDE: Few-Shot Optimized Learning for Directed Evolution

This module provides infrastructure for simulating and evaluating machine learning models on protein engineering tasks, particularly for low-N protein engineering campaigns. FolDE combines protein language model embeddings with few-shot learning to guide directed evolution experiments.

## Quick Start

### System Requirements

- Python 3.8+
- ~200GB disk space for data
- Google Cloud SDK (for data download)

### Installation

1. **Install the backend dependencies:**
   ```bash
   # Recommended: make a python3.12 virtual environment:
   python3.12 -m venv .venv
   source .venv/bin/activate

   # Install dependencies:
   cd backend
   pip install -e ".[dev]"
   ```

2. **Download required data (~200GB):**

   The simulation data is hosted in a public Google Cloud Storage bucket. You'll need the Google Cloud SDK installed:

   ```bash
   gsutil -m rsync -r -d gs://foldedata/ backend/folde/data/
   ```

3. **Verify installation:**
   ```bash
   python -c "from folde.data import get_dms_metadata; print(f'Found {len(get_dms_metadata())} DMS datasets')"
   ```

## Running Benchmark Simulations

This repository includes the benchmark simulations used in the FolDE paper. These simulations evaluate different model configurations across protein engineering datasets.

### Example 1: Test Benchmark (Single-Mutant Datasets)

This benchmark evaluates FolDE on 17 single-mutant DMS datasets with 6 rounds of 16 variants each:

```bash
python backend/notebooks/jacob/251003_test_benchmark.py
```

**Key parameters:**
- 17 single mutant datasets from ProteinGym
- 10 simulations per dataset
- 6 rounds × 16 variants = 96 total measurements per simulation
- Compares: Random, RandomForest, Naturalness-only, and FolDE variants

**Expected runtime:** ~2-4 hours (depending on hardware and parallelization)

**Output:** Results saved to `backend/notebooks/jacob/model_evals/251003-test-benchmark_*.json`

### Customizing Benchmarks

You can create custom benchmarks by modifying the configuration:

```python
from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

# Define your model configuration
config = FolDEModelConfig(
    name="MyFolDE",
    naturalness_model_id="600m",  # ESMC 600M
    embedding_model_id="300m",    # ESMC 300M
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="TorchMLPFewShotModel",
    few_shot_model_params={
        "pretrain": True,
        "pretrain_epochs": 50,
        "ensemble_size": 5,
        "embedding_dim": 960,
        "hidden_dims": [100, 50],
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "train_epochs": 200,
        "train_patience": 40,
        "val_frequency": 10,
        "do_validation_with_pair_fraction": 0.2,
        "decision_mode": "constantliar",
        "lie_noise_stddev_multiplier_schedule": [6.0] * 2 + [100.0] * 8,
    },
)

# Run simulations
results = simulate_campaigns_with_config_checkpoints(
    eval_prefix="my-experiment",
    dms_ids=["BLAT_ECOLX_Stiffler_2015"],  # Choose your datasets
    config_list=[config],
    checkpoint_dir="results",
    round_size=16,           # Variants per round
    number_of_simulations=10,  # Number of simulation replicates
    max_rounds=6,            # Campaign rounds
    random_seed=42,
    num_workers=4,           # Parallel workers
)
```

## Understanding the Code

### Core Simulation Function

The main entry point for running simulations is `simulate_campaigns_with_config_checkpoints()` in [campaign.py](backend/folde/campaign.py:613). This function:

1. Loads protein datasets (DMS data, embeddings, naturalness scores)
2. Runs multiple simulated campaigns with different random seeds
3. Saves checkpoints after each DMS dataset (resumable)
4. Returns evaluation metrics for all configurations

### Model Configuration

`FolDEModelConfig` defines a complete model setup:

- **`naturalness_model_id`**: Which ESM-2 model to use for naturalness scores ("300m", "600m", "3b", "15b")
- **`embedding_model_id`**: Which ESM-2 model to use for sequence embeddings
- **`zero_shot_model_name`**: Model for first round (before any measurements)
  - `"NaturalnessZeroShotModel"`: Use naturalness scores only
  - `"RandomZeroShotModel"`: Random selection (baseline)
- **`few_shot_model_name`**: Model for subsequent rounds
  - `"TorchMLPFewShotModel"`: Neural network ensemble with ranking loss and warm-start (FolDE)
  - `"RandomForestFewShotModel"`: Random forest baseline
  - `"NaturalnessFewShotModel"`: Naturalness-only (no learning)
- **`few_shot_model_params`**: Hyperparameters for the few-shot model
  - `decision_mode`: "mean", "constantliar", or "ucb" for ensemble aggregation
  - `ensemble_size`: Number of models in ensemble
  - `pretrain`: Whether to pretrain on single-mutant naturalness
  - See [few_shot_models.py](backend/folde/few_shot_models.py) for all options

### Key Modules

- [campaign.py](backend/folde/campaign.py) - Campaign simulation logic
- [data.py](backend/folde/data.py) - Data loading utilities
- [few_shot_models.py](backend/folde/few_shot_models.py) - Few-shot learning models (MLP, RandomForest)
- [zero_shot_models.py](backend/folde/zero_shot_models.py) - Zero-shot models (naturalness-based)
- [types.py](backend/folde/types.py) - Type definitions and configuration classes
- [util.py](backend/folde/util.py) - Utility functions for metrics and data processing

## ProteinGym Data

This module uses data from **ProteinGym**, a comprehensive benchmark for assessing protein fitness prediction models. ProteinGym was developed by Cheng et al. and provides a standardized collection of Deep Mutational Scanning (DMS) datasets.

**Citation**:
Cheng, Y., Raghuram, J., Aghazadeh, A., Huang, P.-S., & Russ, W. P. (2023). ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *Nature Methods*.
DOI: [link](https://pubmed.ncbi.nlm.nih.gov/38106144/)

**ProteinGym Repository**:
[https://github.com/OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym)

## Data Structure

The `backend/folde/data` directory contains datasets and related files organized as follows:

### Directory Structure

```
backend/folde/data/
├── DMS_substitutions.csv           # Metadata file from ProteinGym (~200KB)
├── DMS_ProteinGym_substitutions/   # DMS datasets from ProteinGym (~1.0GB)
│   ├── BLAT_ECOLX_Stiffler_2015.csv
│   ├── PTEN_HUMAN_Mighell_2018.csv
│   └── ...
├── embeddings/                     # Protein embeddings from ESM-2 models (**~150GB**)
│   ├── BLAT_ECOLX_Stiffler_2015_embedding_300m.csv
│   ├── BLAT_ECOLX_Stiffler_2015_embedding_600m.csv
│   └── ...
└── naturalness/                    # Protein naturalness scores from ESM-2 (~500MB)
    ├── BLAT_ECOLX_Stiffler_2015_naturalness_300m.csv
    ├── BLAT_ECOLX_Stiffler_2015_naturalness_600m.csv
    └── ...
```

**Data Download:** All files are available from our public GCS bucket at `gs://foldedata/`. See the Quick Start section above for download instructions.

**Original Sources:**
- DMS datasets from [ProteinGym](https://proteingym.org/) (Cheng et al., 2023)
- Embeddings & naturalness scores pre-computed using ESM models (ESM2-600M, 3B, 15B, etc; ESMC-300M, 600M)

See the module documentation for more details on available functions and their usage.
