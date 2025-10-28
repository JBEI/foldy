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
NAME = "250718-naturalness-training-tuning"

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
        "lie_noise_stddev_multiplier": 2.0,
    },
)

config_list = apply_diff_list_to_config(
    base_config,
    [
        ModelDiff(name="hotstart-0epochs", diffs={"few_shot_model_params.pretrain": False}),
        ModelDiff(name="hotstart-1epochs", diffs={"few_shot_model_params.pretrain_epochs": 1}),
        ModelDiff(name="hotstart-5epochs", diffs={"few_shot_model_params.pretrain_epochs": 5}),
        ModelDiff(name="hotstart-10epochs", diffs={"few_shot_model_params.pretrain_epochs": 10}),
        ModelDiff(name="hotstart-20epochs", diffs={"few_shot_model_params.pretrain_epochs": 20}),
        ModelDiff(name="hotstart-50epochs", diffs={"few_shot_model_params.pretrain_epochs": 50}),
        ModelDiff(name="hotstart-100epochs", diffs={"few_shot_model_params.pretrain_epochs": 100}),
        ModelDiff(name="hotstart-200epochs", diffs={"few_shot_model_params.pretrain_epochs": 200}),
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
    max_rounds=10,
    random_seed=42,
    num_workers=10,
)
