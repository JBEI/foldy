"""Reviewer 2 comment 1: augmented ridge baselines built on top of the zero-shot predictions.

(i) a linear model instead of a NN, and (ii) a model combining zero-shot predictions with
measurements but WITHOUT ESM representations. These are the "augmented" regression baselines of

  Hsu, Nisonoff, Fannjiang & Listgarten, "Learning protein fitness models from evolutionary
  and assay-limited data", Nature Biotechnology 40, 1114-1122 (2022)

i.e. ridge on sequence features with the density-model score as one extra feature. Their
48-240 labeled-example regime brackets our 3-round budget of 48.

Three arms, all keeping FolDE's zero-shot round 1, because the reviewer asked for baselines
built ON TOP of zero-shot:
  Ridge-ESM      300m embeddings, augment=False  -> answers (i), linear instead of NN
  Ridge-OHE-aug  one-hot,         augment=True   -> answers (ii), Hsu's OHE+density model
  Ridge-ESM-aug  300m embeddings, augment=True   -> Hsu's embedding-augmented model

Ridge is deterministic, so there is no ensemble and no constant liar: these arms use plain
top-N selection (decision_mode "mean") and therefore differ from FolDE in the ACQUISITION as
well as the surrogate. That is noted in the arm names and must be stated in the response.

Usage:
  python folde/scripts/261001_ridge_baselines.py ohe-single    # Ridge-OHE-aug, 17 singles
  python folde/scripts/261001_ridge_baselines.py ohe-multi     # Ridge-OHE-aug, 3 multis
  python folde/scripts/261001_ridge_baselines.py esm-single    # both ESM arms, 17 singles
  python folde/scripts/261001_ridge_baselines.py esm-multi     # both ESM arms, 3 multis
"""

import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

NAME = "261001-ridge"

# The 17-dataset single-mutation TEST benchmark.
SINGLE_DMS_IDS = [
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

MULTI_DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
]

RIDGE_PARAMS = {
    # Ridge is deterministic: plain top-N, no ensemble, no constant liar.
    "decision_mode": "mean",
    # Recorded in the config (rather than left to the class default) so the checkpoint
    # validation catches any change to it. The grid spans 1e-6..1e6 because a first probe
    # pinned LOO at the floor of a 1e-3..1e6 grid, i.e. the grid, not the data, was choosing
    # the penalty. With p >> n (8000 one-hot columns, 16 labels) the optimum can sit very low,
    # so the grid has to extend far enough for LOO to find an interior minimum.
    "alphas": [10.0**e for e in range(-6, 7)],
}


def ridge_config(name: str, embedding_model_id: str, augment: bool) -> FolDEModelConfig:
    return FolDEModelConfig(
        name=name,
        naturalness_model_id="600m",
        embedding_model_id=embedding_model_id,
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="RidgeFewShotModel",
        few_shot_model_params={**RIDGE_PARAMS, "augment": augment},
    )


RIDGE_ESM = ridge_config("Ridge-ESM", "300m", augment=False)
RIDGE_OHE_AUG = ridge_config("Ridge-OHE-aug", "onehot", augment=True)
RIDGE_ESM_AUG = ridge_config("Ridge-ESM-aug", "300m", augment=True)

# simulate_campaigns_with_config_checkpoints requires every dataset already stored in a
# config's checkpoint to appear in the requested list, so the multi groups request the union
# and rely on resume to skip the singles. Requesting the multis alone aborts the run.
GROUPS = {
    "ohe-single": ([RIDGE_OHE_AUG], SINGLE_DMS_IDS),
    "ohe-multi": ([RIDGE_OHE_AUG], SINGLE_DMS_IDS + MULTI_DMS_IDS),
    "esm-single": ([RIDGE_ESM, RIDGE_ESM_AUG], SINGLE_DMS_IDS),
    "esm-multi": ([RIDGE_ESM, RIDGE_ESM_AUG], SINGLE_DMS_IDS + MULTI_DMS_IDS),
    # SPG1's dense 300m pool does not fit in RAM yet; this group is the two that do. Passing
    # the datasets on the command line instead would shrink the requested list and trip the
    # same checkpoint validation, so it has to be its own group.
    "esm-multi-nospg1": (
        [RIDGE_ESM, RIDGE_ESM_AUG],
        SINGLE_DMS_IDS + ["GRB2_HUMAN_Faure_2021", "PABP_YEAST_Melamed_2013"],
    ),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(GROUPS)}> [dms_id ...]")
    group = sys.argv[1]
    configs, dms_ids = GROUPS[group]
    if len(sys.argv) > 2:
        requested = sys.argv[2:]
        unknown = [d for d in requested if d not in dms_ids]
        if unknown:
            sys.exit(f"{unknown} not in group {group}")
        dms_ids = requested

    print(f"[{group}] {len(configs)} config(s) over {len(dms_ids)} dataset(s)")
    for c in configs:
        print(
            f"  - {c.name}: embeddings={c.embedding_model_id} "
            f"augment={c.few_shot_model_params['augment']} "
            f"decision_mode={c.few_shot_model_params['decision_mode']}"
        )

    simulate_campaigns_with_config_checkpoints(
        eval_prefix=NAME,
        dms_ids=dms_ids,
        config_list=configs,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        num_workers=int(os.environ.get("RIDGE_WORKERS", "4")),
    )
    print(f"RIDGE {group} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
