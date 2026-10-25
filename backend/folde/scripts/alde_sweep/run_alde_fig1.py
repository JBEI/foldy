"""Re-run ALDE (and ALDEzs) on the single-mutation benchmark, saving FULL per-round results.

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
    "A4GRB6_PSEAI_Chen_2020",
    "KKA2_KLEPN_Melnikov_2014",
    "MET_HUMAN_Estevam_2023",
    "MLAC_ECOLI_MacRae_2023",
    "RNC_ECOLI_Weeks_2023",
    "AMIE_PSEAE_Wrenbeck_2017",
    "PAI1_HUMAN_Huttinger_2021",
    "PTEN_HUMAN_Mighell_2018",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "LGK_LIPST_Klesmith_2015",
    "PPARG_HUMAN_Majithia_2016",
    "S22A1_HUMAN_Yee_2023_activity",
    "SC6A4_HUMAN_Young_2021",
    "MTHR_HUMAN_Weile_2021",
    "CAS9_STRP1_Spencer_2017_positive",
    "HMDH_HUMAN_Jiang_2019",
    "MSH2_HUMAN_Jia_2020",  # now fits: OneHotVector keeps the pool sparse
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
        eval_prefix="260920-alde-fig1",
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
