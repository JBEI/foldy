<div align="center">

# Foldy

**Run Boltz, DiffDock, ESM, and FolDE locally through a simple web interface.**

<img src="frontend/public/pksito.gif" width="200" height="200" />

</div>

Protein design tools like Boltz, DiffDock, and ESM are powerful but difficult to install and run. Foldy wraps them in a Docker-based interface so you can focus on your science instead of debugging environments.

**What you can do:**
- Predict protein structures and binding affinities with Boltz
- Dock small molecules with DiffDock
- Get protein language model predictions from ESM
- Run protein engineering campaigns via FolDE

**[Click here for quick setup →](deployment/local/README.md)**

## Deployment Options

Foldy is a composable set of services which can be deployed lots of ways. We currently document four types of deployment: Local, Development, and Helm. The Local deployment runs Foldy with a single command using pre-built Docker images - no git clone required. The Development deployment is not fully featured - it cannot run jobs - but supports the frontend features and can easily be run on a laptop for development purposes. Foldy-in-a-box is a quick deployment option for creating a full featured Foldy instance on a Google Cloud machine. Finally, the Helm deployment is the horizontally scalable, cloud deployment built on Kubernetes. The Helm deployment is involved, but it is secure and can be scaled to hundreds of users and tens of thousands of folds.

You can find more information about employing the different deployment options in their respective `deployment` directories.

|Deployment Type|Description|Ease of setup|Setup|
|---|---|---|---|
|Local|Full featured, single command|Very easy|[Instructions](deployment/local/README.md)|
|Development|Run locally when making code changes|Extremely easy|[Instructions](deployment/development/README.md)|
|Helm|Scalable to hundreds of users|Hard|[Instructions](deployment/helm/README.md)|

## The Interface

See [docs/interface.md](docs/interface.md).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Comparison to Other Tools

There is a rich ecosystem for running structural biology tools, and Foldy is not the right structural biology wrapper for everyone! Please review the Foldy paper for a comparison to other useful structural biology tool wrappers.

## Acknowledgements

Foldy is built on the work of many open-source projects and databases. We are grateful to the developers and maintainers of:

### Structure Prediction & Modeling
- [AlphaFold](https://github.com/deepmind/alphafold) - Deep learning system for protein structure prediction
- [Boltz](https://github.com/jwohlwend/boltz) - Generative protein–ligand interaction modeling
- [ESM2](https://github.com/facebookresearch/esm) - Evolutionary Scale Modeling protein language models (Meta FAIR)
- [ESM C](https://github.com/evolutionaryscale/esm) - Protein representation models (EvolutionaryScale)

### Molecular Docking
- [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) - Molecular docking and virtual screening (CCSB/Scripps Research)
- [DiffDock](https://github.com/gcorso/DiffDock) - Diffusion-based molecular docking

### Protein Domain Analysis
- [Pfam Database](https://www.ebi.ac.uk/interpro/entry/pfam/) - Protein families database (EMBL-EBI InterPro)
- [PfamScan](https://github.com/ebi-pf-team/PfamScan) - Tool for scanning sequences against Pfam HMMs
- [HMMER Suite](http://eddylab.org/software/hmmer) - Biosequence analysis using profile hidden Markov models

### Visualization
- [Mol*](https://github.com/molstar/molstar) - Macromolecular 3D visualization library (PDBe/EMBL-EBI & RCSB PDB)

### Cheminformatics
- [RDKit](https://github.com/rdkit/rdkit) - Cheminformatics and machine learning toolkit

### Web Framework & Libraries
- [Flask](https://flask.palletsprojects.com/) - Python web framework
- [Plotly](https://github.com/plotly/plotly.js) - Interactive graphing library
- [PyTorch](https://github.com/pytorch/pytorch) - Machine learning framework

We thank all contributors and maintainers of these projects!

Use of the third-party software, libraries or code Foldy may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## License

Foldy is distributed under a modified BSD license (see LICENSE).

## Copyright Notice

Foldy Copyright (c) 2023 to 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and University
of California, Berkeley. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
