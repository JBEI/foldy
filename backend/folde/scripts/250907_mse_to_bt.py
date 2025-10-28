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
NAME = "250907-mse-to-bt"

random_config = FolDEModelConfig(
    name="Random",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="RandomFewShotModel",
    few_shot_model_params={},
)

random_forest_config = FolDEModelConfig(
    name="RandomForest",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="RandomForestFewShotModel",
    few_shot_model_params={
        "n_estimators": 100,
        "criterion": "friedman_mse",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "verbose": 0,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "max_samples": None,
    },
)

mlp_config_list = apply_diff_list_to_config(
    FolDEModelConfig(
        name="MLP",
        # Required parameters
        naturalness_model_id="600m",  # ESM-2 650M model
        embedding_model_id="300m",  # Same model for embeddings
        zero_shot_model_name="RandomZeroShotModel",
        zero_shot_model_params={},
        # Few-shot model configuration (used after first round)
        few_shot_model_name="TorchMLPFewShotModel",
        few_shot_model_params={
            "pretrain": False,
            "pretrain_epochs": 50,
            "ensemble_size": 1,
            "embedding_dim": 960,
            "hidden_dims": [100, 50],
            "dropout": 0.2,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "train_epochs": 200,
            "train_patience": 40,
            "val_frequency": 10,
            "do_validation_with_pair_fraction": 0.2,
        },
    ),
    [
        ModelDiff(name="MSE", diffs={"few_shot_model_params.use_mse_loss": True}),
        ModelDiff(name="BT", diffs={"few_shot_model_params.use_mse_loss": False}),
    ],
    exclude_base_config=True,
)

config_list = [random_config] + [random_forest_config] + mlp_config_list

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
    num_workers=10,
)
