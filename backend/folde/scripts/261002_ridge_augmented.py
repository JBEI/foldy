"""Corrected augmented-ridge baselines (Hsu et al. 2022) -- rerun of the two augmented arms.

The 261001-ridge augmented arms were invalid. Both failed, in opposite directions:

  Ridge-ESM-aug: 961 features, median alpha 1e3, |naturalness coef| 0.00187 against mean
    |embedding coef| 0.00086 -- a ratio of 2.2. The single most informative feature was
    penalized down to roughly the level of an average embedding dimension, so the fit did not
    test augmentation at all.
  Ridge-OHE-aug: 5321 features, alpha pinned at the bottom of the grid, i.e. unregularized
    least squares on a badly underdetermined system rather than ridge.

Lowering alpha globally would trade the first failure for the second. The fix is to take the
naturalness column out of the L2 penalty, using the two-step form: OLS of activity on
naturalness alone, then ridge the residuals on the embedding / one-hot block. That makes the
naturalness coefficient unpenalized by construction.

Ridge-ESM (no augmentation) was unaffected and is NOT rerun; the 261001-ridge results stand.

Alpha grid spans 1e-9..1e9 (37 points). A diagnostic over an even wider 1e-12..1e12 grid found
a genuine interior optimum for the ESM block (LOO curve max/min ~1.55-1.77, optimum at 1e3 to
3.2e3) but an essentially FLAT curve for the one-hot block (max/min 1.03-1.06 across 24 orders
of magnitude). Every fit logs alpha plus whether it sat on a grid boundary, so that is
verifiable from the saved results rather than taken on trust.

Usage:
  python folde/scripts/261002_ridge_augmented.py single
  python folde/scripts/261002_ridge_augmented.py multi
"""

import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

NAME = "261002-ridge-aug"

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

AUGMENTED_PARAMS = {
    "decision_mode": "mean",  # ridge is deterministic: plain top-N, no ensemble, no liar
    "augment": True,
    # The fix: naturalness enters through an unpenalized OLS step, not through the ridge.
    "two_step": True,
    "alphas": [float(a) for a in np.logspace(-9.0, 9.0, 37)],
}


def augmented_config(name: str, embedding_model_id: str) -> FolDEModelConfig:
    return FolDEModelConfig(
        name=name,
        naturalness_model_id="600m",
        embedding_model_id=embedding_model_id,
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="RidgeFewShotModel",
        few_shot_model_params=dict(AUGMENTED_PARAMS),
    )


RIDGE_ESM_AUG = augmented_config("Ridge-ESM-aug", "300m")
RIDGE_OHE_AUG = augmented_config("Ridge-OHE-aug", "onehot")
CONFIGS = [RIDGE_ESM_AUG, RIDGE_OHE_AUG]

# The checkpoint helper requires stored dataset IDs to be a subset of the requested ones, so
# the multi group requests the union and relies on resume to skip the singles.
GROUPS = {
    "single": SINGLE_DMS_IDS,
    "multi": SINGLE_DMS_IDS + MULTI_DMS_IDS,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(GROUPS)}>")
    group = sys.argv[1]
    dms_ids = GROUPS[group]

    print(f"[{group}] {len(CONFIGS)} augmented arms over {len(dms_ids)} dataset(s)")
    for c in CONFIGS:
        print(
            f"  - {c.name}: embeddings={c.embedding_model_id} "
            f"two_step={c.few_shot_model_params['two_step']} "
            f"alphas={len(c.few_shot_model_params['alphas'])} points "
            f"[{min(c.few_shot_model_params['alphas']):.0e}, "
            f"{max(c.few_shot_model_params['alphas']):.0e}]"
        )

    simulate_campaigns_with_config_checkpoints(
        eval_prefix=NAME,
        dms_ids=dms_ids,
        config_list=CONFIGS,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        # SPG1's dense 300m pool only fits via the in-process path, so the multi group runs
        # single-worker. See the chunked loader in data.py.
        num_workers=int(os.environ.get("RIDGE_WORKERS", "4")),
    )
    print(f"RIDGE-AUG {group} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
