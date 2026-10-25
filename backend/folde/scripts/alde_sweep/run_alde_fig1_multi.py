"""Re-run ALDE (and ALDEzs) on the multi-mutation benchmark, saving FULL per-round results.

The earlier sweeps only persisted round-3 summaries; Figure 1's panels are cumulative curves,
so we need the complete SimulationResult objects. Uses the repo's checkpointing helper, which
writes one model_evals JSON per config.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
]

P = {
    "ensemble_size": 5,
    "hidden_dims": [30, 30],
    "bootstrap_size": 0.9,
    "train_iter": 300,
    "learning_rate": 1e-3,
    "early_stop_window": 30,
    "acquisition": "greedy",
}

configs = [
    FolDEModelConfig(
        name="ALDE",
        naturalness_model_id="onehot",
        embedding_model_id="onehot",
        zero_shot_model_name="RandomZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="ALDEFewShotModel",
        few_shot_model_params=dict(P),
    ),
    FolDEModelConfig(
        name="ALDEzs",
        naturalness_model_id="600m",
        embedding_model_id="onehot",
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="ALDEFewShotModel",
        few_shot_model_params=dict(P),
    ),
]

if __name__ == "__main__":
    simulate_campaigns_with_config_checkpoints(
        eval_prefix="260920-alde-fig1-multi",
        dms_ids=DMS_IDS,
        config_list=configs,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        num_workers=4,
    )
    print("FIG1 RUN COMPLETE", flush=True)
