"""Screen alternatives to warm-starting on naturalness -- TRAINING BENCHMARK ONLY.

Reviewer 1 comment 6 asks whether warm-starting the MLP on naturalness is the best way to
exploit the prior, versus (a) naturalness as an explicit input feature or (c) a residual
correction on top of the prior. This script screens both alternatives against warm-starting.

*** THE 9 DATASETS BELOW ARE THE TRAINING BENCHMARK. DO NOT ADD TEST-BENCHMARK IDS. ***
FolDE was designed using only these 9; the 17-dataset test benchmark was held out. The point
of this experiment is that these variants were eliminated at the training stage and never
touched held-out data, so running them on test would destroy that claim.

Four configs, all sharing one harness so the comparison is apples-to-apples:
  FolDE-FeatureConcat       naturalness appended as an input feature (pretrain=False)
  FolDE-ResidualCorrection  score = naturalness_z + MLP correction (pretrain=False)
  FolDE-hotstart-50epochs   warm-starting, i.e. FolDE as published
  FolDE-hotstart-0epochs    no naturalness in the few-shot model at all

Plus one plumbing check, reported as validation rather than as a method:
  FolDE-FeatureConcatConstNat  FeatureConcat with the naturalness feature forced to a
      constant. It should closely reproduce hotstart-0epochs. It will not reproduce it
      exactly: the feature-concat net is one column wider, so torch's seeded initialization
      draws different weights for the embedding columns.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelDiff
from folde.util import apply_diff_list_to_config

# The 9-dataset TRAINING benchmark. See the module docstring before touching this list.
folde_train_dms_ids = [
    "ANCSZ_Hobbs_2022",
    "BLAT_ECOLX_Firnberg_2014",
    "CBS_HUMAN_Sun_2020",
    "HEM3_HUMAN_Loggerenberg_2023",
    "HSP82_YEAST_Flynn_2019",
    "HXK4_HUMAN_Gersing_2022_activity",
    "OXDA_RHOTO_Vanella_2023_activity",
    "PPM1D_HUMAN_Miller_2022",
    "SHOC2_HUMAN_Kwon_2022",
]

NAME = "260926-naturalness-alternatives"

# Hyperparameters identical to 250718_naturalness_training_tuning.py's base_config, so the
# only thing that varies across configs is how naturalness enters the few-shot model.
base_config = FolDEModelConfig(
    name="FolDE",
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
        "lie_noise_stddev_multiplier": 2.0,
    },
)

# Both alternatives run with pretrain=False: they replace warm-starting rather than add to it.
config_list = apply_diff_list_to_config(
    base_config,
    [
        ModelDiff(
            name="FeatureConcat",
            diffs={
                "few_shot_model_name": "NaturalnessFeatureMLPFewShotModel",
                "few_shot_model_params.pretrain": False,
            },
        ),
        ModelDiff(
            name="ResidualCorrection",
            diffs={
                "few_shot_model_name": "NaturalnessResidualMLPFewShotModel",
                "few_shot_model_params.pretrain": False,
            },
        ),
        ModelDiff(name="hotstart-50epochs", diffs={"few_shot_model_params.pretrain_epochs": 50}),
        ModelDiff(name="hotstart-0epochs", diffs={"few_shot_model_params.pretrain": False}),
        ModelDiff(
            name="FeatureConcatConstNat",
            diffs={
                "few_shot_model_name": "NaturalnessFeatureMLPFewShotModel",
                "few_shot_model_params.pretrain": False,
                "few_shot_model_params.constant_naturalness_for_validation": True,
            },
        ),
    ],
    exclude_base_config=True,
)

if __name__ == "__main__":
    print(f"Running {len(config_list)} configs over {len(folde_train_dms_ids)} TRAIN datasets")
    for config in config_list:
        print(f"  - {config.name}  ({config.few_shot_model_name})")

    # Guard against the one mistake that would invalidate the whole experiment.
    forbidden = set(folde_train_dms_ids) - {
        "ANCSZ_Hobbs_2022",
        "BLAT_ECOLX_Firnberg_2014",
        "CBS_HUMAN_Sun_2020",
        "HEM3_HUMAN_Loggerenberg_2023",
        "HSP82_YEAST_Flynn_2019",
        "HXK4_HUMAN_Gersing_2022_activity",
        "OXDA_RHOTO_Vanella_2023_activity",
        "PPM1D_HUMAN_Miller_2022",
        "SHOC2_HUMAN_Kwon_2022",
    }
    if forbidden:
        sys.exit(f"REFUSING TO RUN: non-training datasets present: {sorted(forbidden)}")

    results = simulate_campaigns_with_config_checkpoints(
        eval_prefix=NAME,
        dms_ids=folde_train_dms_ids,
        config_list=config_list,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        num_workers=4,
    )
    print("NATURALNESS ALTERNATIVES RUN COMPLETE", flush=True)
