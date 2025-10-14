# Protein Engineering Prediction Module

This module provides infrastructure for evaluating machine learning models on protein engineering tasks, particularly for low-N protein engineering campaigns.

## ProteinGym Data

This module uses data from **ProteinGym**, a comprehensive benchmark for assessing protein fitness prediction models. ProteinGym was developed by Cheng et al. and provides a standardized collection of Deep Mutational Scanning (DMS) datasets.

**Citation**:
Cheng, Y., Raghuram, J., Aghazadeh, A., Huang, P.-S., & Russ, W. P. (2023). ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *Nature Methods*.
DOI: [link](https://pubmed.ncbi.nlm.nih.gov/38106144/)

**ProteinGym Repository**:
[https://github.com/OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym)

## Data Structure

The `prediction/data` directory contains datasets and related files organized as follows:

### Directory Structure

```
prediction/data/
├── DMS_substitutions.csv           # Metadata file from ProteinGym listing all available DMS datasets. Download it from the [ProteinGym github](https://github.com/OATML-Markslab/ProteinGym/blob/main/reference_files/DMS_substitutions.csv).
├── DMS_ProteinGym_substitutions/   # Directory containing DMS datasets from ProteinGym. Get a zip file with this content from the [ProteinGym website](https://proteingym.org/download): Go to ProteinGym.com/download, then select DMS Assays > Substitutions
│   ├── PROTEIN_ID1.csv             # Individual DMS dataset files
│   ├── PROTEIN_ID2.csv
│   └── ...
├── embeddings/                     # Directory containing protein embeddings. Download these from a Foldy instance where embeddings have been run. Recommend downloading from the "Embed" tab directly, rather than the files tab, so the file names are correctly formatted.
│   ├── PROTEIN_ID1_embedding_MODEL_ID.csv  # Embedding files with MODEL_ID identifier
│   ├── PROTEIN_ID2_embedding_MODEL_ID.csv
│   └── ...
└── naturalness/                    # Directory containing protein naturalness scores.  Download these from a Foldy instance where embeddings have been run. Recommend downloading from the "Naturalness" tab directly, rather than the files tab, so the file names are correctly formatted.
    ├── PROTEIN_ID1_naturalness_MODEL_ID.csv  # Naturalness files with MODEL_ID identifier
    ├── PROTEIN_ID2_naturalness_MODEL_ID.csv
    └── ...
```

### File Formats

#### DMS_substitutions.csv

This CSV file is directly from ProteinGym and contains metadata about Deep Mutational Scanning (DMS) datasets, with columns including:

- `DMS_id`: Unique identifier for each DMS dataset (e.g., "BLAT_ECOLX_Stiffler_2015")
- `DMS_filename`: Filename of the DMS data in the DMS_ProteinGym_substitutions directory
- `UniProt_ID`: UniProt identifier for the protein
- Various additional metadata columns about the dataset, protein, and experimental conditions

#### DMS Dataset Files (inside DMS_ProteinGym_substitutions/)

These files are sourced directly from ProteinGym. They are CSV files containing mutation data with columns:

- `mutant`: Mutation identifier (e.g., "H24C")
- `mutated_sequence`: Full protein sequence with mutation
- `DMS_score`: Experimental measurement of protein function/fitness
- Additional dataset-specific columns

The `mutant` column is mapped to `seq_id` in code by replacing any colons with underscores.

#### Embedding Files (inside embeddings/)

Embedding files contain protein embeddings with columns:

- `seq_id`: Sequence identifier matching the DMS dataset
- `seq`: Protein sequence
- `embedding`: Vector representation of the protein (stored as a string representation of a list)

File naming pattern: `{DMS_id}_embedding_{model_id}.csv`

#### Naturalness Files (inside naturalness/)

Naturalness files contain protein naturalness scores with columns:

- `seq_id`: Sequence identifier matching the DMS dataset
- `wt_marginal`: Naturalness score (renamed to `naturalness` in code)
- Additional columns may be present depending on the naturalness model

File naming pattern: `{DMS_id}_naturalness_{model_id}.csv`

## Usage

To use these datasets, you can load them with the provided module functions:

```python
from prediction import get_available_proteingym_datasets, get_proteingym_dataset

# Get available datasets for specific models
datasets = get_available_proteingym_datasets("300m", "esm2")

# Load a specific dataset
naturalness_df, embedding_df, activity_df = get_proteingym_dataset(
    "BLAT_ECOLX_Stiffler_2015", "300m", "esm2"
)
```

See the module documentation for more details on available functions and their usage.
