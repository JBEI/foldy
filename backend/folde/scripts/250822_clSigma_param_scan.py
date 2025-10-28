# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelDiff
from folde.util import apply_diff_list_to_config

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


# Example configuration
NAME = "250922-clSigma-param-scan"

base_config = FolDEModelConfig(
    name="FolDE",
    # Required parameters
    naturalness_model_id="600m",  # ESM-2 650M model
    embedding_model_id="300m",  # Same model for embeddings
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    # Few-shot model configuration (used after first round)
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
        # "lie_noise_stddev_multiplier": 2.0,
    },
)

config_list = apply_diff_list_to_config(
    base_config,
    [
        # ModelDiff(name="clS1000", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 1000}),
        ModelDiff(name="clS100", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 100}),
        ModelDiff(name="clS10", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 10}),
        ModelDiff(name="clS6", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 6}),
        ModelDiff(name="clS4", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 4}),
        ModelDiff(name="clS3", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 3}),
        ModelDiff(name="clS2", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 2}),
        ModelDiff(name="clS1", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 1}),
        # ModelDiff(name="clS05", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 0.5}),
        ModelDiff(name="clS025", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 0.25}),
        # ModelDiff(name="clS01", diffs={"few_shot_model_params.lie_noise_stddev_multiplier": 0.1}),
        ModelDiff(
            name="clS12-schedule",
            diffs={
                "few_shot_model_params.lie_noise_stddev_multiplier": None,
                "few_shot_model_params.lie_noise_stddev_multiplier_schedule": [12.0] * 2
                + [100.0] * 8,
            },
        ),
        ModelDiff(
            name="clS6-schedule",
            diffs={
                "few_shot_model_params.lie_noise_stddev_multiplier": None,
                "few_shot_model_params.lie_noise_stddev_multiplier_schedule": [6.0] * 2
                + [100.0] * 8,
            },
        ),
        ModelDiff(
            name="clS3-schedule",
            diffs={
                "few_shot_model_params.lie_noise_stddev_multiplier": None,
                "few_shot_model_params.lie_noise_stddev_multiplier_schedule": [3.0] * 2
                + [100.0] * 8,
            },
        ),
        ModelDiff(
            name="clS1-schedule",
            diffs={
                "few_shot_model_params.lie_noise_stddev_multiplier": None,
                "few_shot_model_params.lie_noise_stddev_multiplier_schedule": [1.0] * 2
                + [100.0] * 8,
            },
        ),
    ],
    exclude_base_config=True,
)

print(f"Config 1/{len(config_list)}:")
print(config_list[0].model_dump_json(indent=2))

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
    num_workers=5,
)
