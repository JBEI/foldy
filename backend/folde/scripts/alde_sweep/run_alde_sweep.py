"""Run the ALDE baseline across the FolDE test benchmark, one dataset at a time.

Appends a JSON line per dataset to alde_results.jsonl as soon as it finishes, so partial
results are readable while the sweep is still running.
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from folde.campaign import simulate_campaign
from folde.types import FolDEModelConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alde_results.jsonl")

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
    "MSH2_HUMAN_Jia_2020",
]

alde_config = FolDEModelConfig(
    name="ALDE",
    naturalness_model_id="onehot",
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


def summarize(config_result, max_round=3):
    n10, hit1, best = [], [], []
    for sim in config_result.simulation_results:
        pcts = [m.percentile for m in sim.mutant_metrics if m.round_found <= max_round]
        n10.append(sum(1 for p in pcts if p >= 0.90))
        hit1.append(1.0 if any(p >= 0.99 for p in pcts) else 0.0)
        best.append(max(pcts))
    n = len(n10)
    return {
        "top10_count": sum(n10) / n,
        "p_top1": sum(hit1) / n,
        "best_percentile": sum(best) / n,
        "n_sims": n,
    }


already_done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            try:
                already_done.add(json.loads(line)["dms_id"])
            except Exception:
                pass

for dms_id in DMS_IDS:
    if dms_id in already_done:
        print(f"skip {dms_id} (done)", flush=True)
        continue
    t0 = time.time()
    try:
        result = simulate_campaign(
            dms_id=dms_id,
            round_size=16,
            number_of_simulations=10,
            config_list=[alde_config],
            activity_column="DMS_score",
            max_rounds=6,
            random_seed=42,
            num_workers=int(os.environ.get("ALDE_WORKERS", "2")),
        )
        record = {
            "dms_id": dms_id,
            "pool_size": result.config_results[0].simulation_results[0].variant_pool_size,
            "elapsed_sec": round(time.time() - t0, 1),
            **summarize(result.config_results[0]),
        }
    except Exception as e:
        record = {"dms_id": dms_id, "error": repr(e), "elapsed_sec": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"done {dms_id} in {record['elapsed_sec']}s", flush=True)

print("SWEEP COMPLETE", flush=True)
