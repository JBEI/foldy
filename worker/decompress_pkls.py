
#!/usr/bin/env python
"""CLI tool to decompress the PKLs output by AlphaFold.

Specifically, the results pkls will be copied into files of the form
ranked_N.pkl, and some of the numpy arrays within those pickles will
be copied into *.np files within a new ranked_N subdirectory."""

import argparse
from pathlib import Path
import json
import pickle
import shutil
import numpy as np
import scipy.special


NUMPY_KEYS = [
    'plddt',
    'max_predicted_aligned_error',
    'iptm',
    'predicted_aligned_error',
    # 'aligned_confidence_probs',  # This one is too big!
]


CONTACT_DISTANCE_THRESHOLDS = [8, 12]


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("A CLI for parsing out the PKLs from an alphafold run."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Raw output from an Alphafold run.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Where to place new pkls and numpy files. Can be the same as run_dir.",
    )
    return parser


def make_ranked_pkls(ranking_debug: dict, run_dir: Path, out_dir: Path):
    for rank, model_name in enumerate(ranking_debug['order']):
        shutil.copy(run_dir / f'result_{model_name}.pkl', out_dir / f'ranked_{rank}.pkl')


def decompress_pkls(run_dir, out_dir):
    with open(run_dir / 'ranking_debug.json', 'rb') as f:
        ranking_debug = json.load(f)

    for rank, model_name in enumerate(ranking_debug['order']):
        rank_name = f'ranked_{rank}'

        original_pkl_fpath = run_dir / f'result_{model_name}.pkl'
        print(f'Copying {original_pkl_fpath} to {rank_name}')
        shutil.copy(original_pkl_fpath, out_dir / f'{rank_name}.pkl')

        # Write out the numpy arrays from the pkl, individually.
        with open(out_dir / f'{rank_name}.pkl', 'rb') as f:
            model_pkl = pickle.load(f)

        rank_dir = out_dir / rank_name
        rank_dir.mkdir(exist_ok=True)
        for np_key in NUMPY_KEYS:
            if np_key in model_pkl:
                np.save(rank_dir / f'{np_key}.npy', model_pkl[np_key])
        
        # Compute contact distances.
        #
        # Supplemental of https://www.science.org/doi/full/10.1126/science.abm4805
        if 'distogram' in model_pkl and 'aligned_confidence_probs' in model_pkl:
            for dist_threshold in CONTACT_DISTANCE_THRESHOLDS:
                last_bin = np.where(model_pkl['distogram']['bin_edges'] < dist_threshold)[0].max() + 1
                pdist = model_pkl['distogram']['logits']
                pdist = scipy.special.softmax(pdist, axis=-1)
                prob12 = np.sum(pdist[:,:,:last_bin], axis=-1)
                np.save(rank_dir / f'contact_prob_{dist_threshold}A.npy', prob12.astype(np.float16))



if __name__ == '__main__':
    args: dict = vars(cli().parse_args())
    model_script = decompress_pkls(Path(args['run_dir']), Path(args['out_dir']))
