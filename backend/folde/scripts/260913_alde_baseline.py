"""ALDE baseline benchmark (Reviewer 1, comment 5).

Runs the reimplemented ALDE surrogate of Yang et al. (2025) on the FolDE test benchmark, so
the paper can report a comparison against a current uncertainty-aware ALDE method rather than
only against random selection, zero-shot naturalness, and an EVOLVEpro-style random forest.

The ALDE configurations here use ONE-HOT encodings and therefore need no protein language
model artifacts -- no embeddings and no naturalness files. That is not a weakened baseline:
one-hot is the default in the ALDE repository, it is the encoding used for their wet-lab
campaign, and their own conclusion is that ESM2 embeddings gave no meaningful benefit over it.

Two comparison regimes are included, and both belong in the response letter:

  * ``ALDE-*`` starts from a RANDOM first round, exactly as the base ALDE repository does.
    This is ALDE as published.
  * ``ALDEzs-*`` starts from FolDE's naturalness-based zero-shot first round. This isolates
    the surrogate and acquisition function from the round-1 advantage, and mirrors the
    "focused training" option in their own alde4ssmula fork. Note this variant DOES require
    naturalness files.

Run from the backend directory:
    python folde/scripts/260913_alde_baseline.py
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelDiff
from folde.util import apply_diff_list_to_config

folde_test_dms_ids = [
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "SC6A4_HUMAN_Young_2021",
    "PTEN_HUMAN_Mighell_2018",
    "S22A1_HUMAN_Yee_2023_activity",
    "KKA2_KLEPN_Melnikov_2014",
    "PPARG_HUMAN_Majithia_2016",
    "MET_HUMAN_Estevam_2023",
    "MTHR_HUMAN_Weile_2021",
    "LGK_LIPST_Klesmith_2015",
    "AMIE_PSEAE_Wrenbeck_2017",
    "PAI1_HUMAN_Huttinger_2021",
    "A4GRB6_PSEAI_Chen_2020",
    "MSH2_HUMAN_Jia_2020",
    "MLAC_ECOLI_MacRae_2023",
    "RNC_ECOLI_Weeks_2023",
    "HMDH_HUMAN_Jiang_2019",
    "CAS9_STRP1_Spencer_2017_positive",
]

NAME = "260913-alde-baseline"

# ALDE as published: random round 1, one-hot encoding, no PLM anywhere in the pipeline.
alde_config = FolDEModelConfig(
    name="ALDE",
    naturalness_model_id="onehot",  # placeholder; never consulted in one-hot mode
    embedding_model_id="onehot",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="ALDEFewShotModel",
    few_shot_model_params={
        "ensemble_size": 5,
        "hidden_dims": [30, 30],
        "bootstrap_size": 0.9,
        "train_iter": 300,
        "learning_rate": 1e-3,
        "early_stop_window": 30,
        "acquisition": "greedy",
    },
)

# Same surrogate, same one-hot encoding, but handed FolDE's naturalness-based zero-shot first
# round instead of a random one. Exactly ONE variable differs from ``alde_config`` above -- the
# round-1 strategy -- so the pair separates "FolDE's advantage comes from starting with
# zero-shot" from "FolDE's advantage comes from the surrogate and acquisition function". That
# is the comparison a reviewer will ask for, and it is cleaner than swapping ALDE onto 300m
# embeddings, which would move the encoding at the same time.
#
# Needs `naturalness/<DMS_ID>_naturalness_600m.csv`; no embedding files required, because the
# one-hot data path now loads real naturalness whenever it is present.
alde_zs_config = FolDEModelConfig(
    name="ALDEzs",
    naturalness_model_id="600m",
    embedding_model_id="onehot",
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="ALDEFewShotModel",
    few_shot_model_params=dict(alde_config.few_shot_model_params),
)

# A third variant worth running once 300m embeddings are local: ALDE's surrogate on the same
# embeddings FolDE uses, which controls the encoding rather than the round-1 strategy.
alde_emb_config = FolDEModelConfig(
    name="ALDEemb",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="ALDEFewShotModel",
    few_shot_model_params=dict(alde_config.few_shot_model_params),
)

# Random baseline recomputed in one-hot mode, as an internal control: it should reproduce the
# cached ``251003-test-benchmark_Random`` numbers, confirming the one-hot data path selects
# the same candidate pool as the embedding path.
random_control_config = FolDEModelConfig(
    name="RandomControl",
    naturalness_model_id="onehot",
    embedding_model_id="onehot",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="RandomFewShotModel",
    few_shot_model_params={},
)

config_list = [random_control_config] + apply_diff_list_to_config(
    alde_config,
    [
        # ALDE-base is the published default: greedy acquisition.
        ModelDiff(name="ucb", diffs={"few_shot_model_params.acquisition": "ucb"}),
        ModelDiff(name="ts", diffs={"few_shot_model_params.acquisition": "ts"}),
    ],
)

if __name__ == "__main__":
    print(f"Running {len(config_list)} configs over {len(folde_test_dms_ids)} datasets")
    for config in config_list:
        print(f"  - {config.name}")

    results = simulate_campaigns_with_config_checkpoints(
        eval_prefix=NAME,
        dms_ids=folde_test_dms_ids,
        config_list=config_list,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        num_workers=2,
    )
