"""Reviewer 2 comment 2: does FolDE predict better on multi-mutants whose constituent
substitutions it has already measured?

This cannot be answered from cached results. The result JSONs retain only the ~160 variants
each simulation selected, and within that set every variant is high-percentile, so
predicted-vs-actual correlation is range-restricted to noise. So we log predictions over
UNMEASURED pool variants instead, stratified by familiarity.

At each round >= 2, after the few-shot model is fit and BEFORE the campaign selects, we score
a stratified sample of variants the simulation has not measured, and record predicted vs
realized activity.

*** RNG ISOLATION IS THE WHOLE BALL GAME ***
run_single_sim_parallel() calls np.random.seed(sim_random_seed), and the campaign's selection
logic draws from that global stream. If this observer drew from it too, every subsequent
selection would shift and the run would no longer reproduce the published campaigns. So:
  * sampling uses a private np.random.RandomState, never np.random.*
  * predict() consumes no randomness (eval mode: dropout is identity, batchnorm uses running
    stats), so scoring extra variants cannot perturb the stream either
Step 5 of the task verifies this empirically against the cached benchmark.

Usage:  python folde/scripts/261001_seen_unseen.py <out_dir> [dms_id ...]
"""

import hashlib
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

from app.helpers.sequence_util import get_allele_set
from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

NAME = "261001-seen-unseen"

MULTI_DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
]

# Exactly FolDE-base as used in 251003-multimutant-benchmark.
base_config = FolDEModelConfig(
    name="FolDE-base",
    naturalness_model_id="600m",
    embedding_model_id="300m",
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
        "lie_noise_stddev_multiplier_schedule": [6, 6, 100, 100, 100, 100, 100, 100, 100, 100],
    },
)

ROWS = [
    "dms_id",
    "sim",
    "round",
    "seq_id",
    "n_alleles_seen",
    "n_positions_seen",
    "predicted_activity",
    "predicted_activity_stddev",
    "activity",
    "percentile",
]


def _locus_of(allele: str) -> str:
    """'Q228T' -> '228'. Position identity only; the residue identities are stripped."""
    return allele[1:-1]


class SeenUnseenObserver:
    """Scores a stratified sample of unmeasured pool variants after each fit.

    Must be picklable: it crosses the ProcessPoolExecutor boundary, so it holds only plain
    data and writes its own per-(dataset, sim) CSV rather than mutating shared state.
    """

    def __init__(self, out_dir: str, per_stratum: int = 2000, seed: int = 20261001):
        self.out_dir = out_dir
        self.per_stratum = per_stratum
        self.seed = seed

    def __call__(
        self,
        round_num: int,
        sim_idx: Any,
        dms_id: str,
        few_shot_model: Any,
        world_state: Any,
        whole_world_activity_series: pd.Series,
    ) -> None:
        if round_num < 2:
            return

        measured_ids = list(world_state.get_measured_activity_series().index)
        measured_alleles = set()
        for seq_id in measured_ids:
            measured_alleles |= get_allele_set(seq_id)
        measured_positions = {_locus_of(a) for a in measured_alleles}

        candidate_naturalness = world_state.get_unmeasured_naturalness_df()
        candidate_ids = list(candidate_naturalness.index)

        # -- stratify ---------------------------------------------------------------------
        n_alleles_seen: Dict[str, int] = {}
        n_positions_seen: Dict[str, int] = {}
        strata: Dict[int, List[str]] = {}
        for seq_id in candidate_ids:
            alleles = get_allele_set(seq_id)
            n_a = len(alleles & measured_alleles)
            n_p = len({_locus_of(a) for a in alleles} & measured_positions)
            n_alleles_seen[seq_id] = n_a
            n_positions_seen[seq_id] = n_p
            strata.setdefault(n_a, []).append(seq_id)

        # -- sample, with a PRIVATE rng (see module docstring) ----------------------------
        # Seeded by (dataset, sim, round) so the sample is reproducible but independent of the
        # campaign's stream. np.random.* is deliberately never called here.
        # hashlib, not hash(): Python salts str hashing per process, so hash() would make the
        # "reproducible" sample differ between runs.
        digest = hashlib.sha256(
            f"{self.seed}|{dms_id}|{int(sim_idx or 0)}|{round_num}".encode()
        ).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
        chosen: List[str] = []
        for stratum in sorted(strata):
            ids = strata[stratum]
            if len(ids) <= self.per_stratum:
                chosen.extend(ids)
            else:
                picks = rng.choice(len(ids), size=self.per_stratum, replace=False)
                chosen.extend([ids[i] for i in picks])
        if not chosen:
            return

        # -- predict ---------------------------------------------------------------------
        # DELIBERATE DEVIATION from "chunk at 2048 rows": TorchMLPFewShotModel.predict centers
        # scores by the mean of whatever batch it is handed. Chunking would centre each chunk
        # by a different offset, and a Spearman pooled across chunks would partly measure that
        # artifact rather than the model. The sample is bounded at 3 strata x per_stratum
        # (<= 6000 rows, ~46 MB at 960 float64), so one call is cheap -- the chunking advice
        # exists to avoid materializing the full 268,481-variant pool, which this never does.
        chosen_index = pd.Index(chosen)
        ensemble = few_shot_model.predict(
            world_state.naturalness_df.loc[chosen_index],
            world_state.embedding_series.loc[chosen_index],
        )
        stacked = np.stack([s.to_numpy(dtype=np.float64) for s in ensemble], axis=0)
        predicted = stacked.mean(axis=0)
        predicted_std = stacked.std(axis=0)

        percentiles = whole_world_activity_series.rank(pct=True)
        frame = pd.DataFrame(
            {
                "dms_id": dms_id,
                "sim": int(sim_idx or 0),
                "round": round_num,
                "seq_id": chosen,
                "n_alleles_seen": [n_alleles_seen[s] for s in chosen],
                "n_positions_seen": [n_positions_seen[s] for s in chosen],
                "predicted_activity": predicted,
                "predicted_activity_stddev": predicted_std,
                "activity": whole_world_activity_series.loc[pd.Index(chosen)].to_numpy(),
                "percentile": percentiles.loc[pd.Index(chosen)].to_numpy(),
            }
        )[ROWS]

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{dms_id}__sim{int(sim_idx or 0)}.csv")
        # One file per (dataset, sim); each worker owns its own file, so no write races.
        frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main() -> None:
    out_dir = sys.argv[1]
    dms_ids = sys.argv[2:] or MULTI_DMS_IDS
    for dms_id in dms_ids:
        if dms_id not in MULTI_DMS_IDS:
            sys.exit(f"unknown dms_id {dms_id}")

    print(f"seen/unseen run over {dms_ids}, writing per-sim CSVs to {out_dir}")
    if True:
        simulate_campaigns_with_config_checkpoints(
            eval_prefix=NAME,
            dms_ids=dms_ids,
            config_list=[base_config],
            checkpoint_dir="folde/model_evals",
            round_size=16,
            number_of_simulations=10,
            activity_column="DMS_score",
            max_rounds=6,
            random_seed=42,
            num_workers=int(os.environ.get("SEEN_UNSEEN_WORKERS", "2")),
            round_observer=SeenUnseenObserver(out_dir=out_dir),
        )
    print("SEEN UNSEEN RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
