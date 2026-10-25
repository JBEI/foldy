"""Naturalness-as-input alternatives, evaluated on the HELD-OUT test benchmark.

Reviewer 1 comment 6 asks whether warm-starting on naturalness is the best way to exploit the
prior. The alternatives -- naturalness as an explicit input feature, and naturalness as an
additive prior the MLP corrects -- were screened on the 9-dataset development (training)
benchmark by 260926_naturalness_alternatives.py, which deliberately refused to touch test.

They are additionally evaluated here on the held-out 20-target benchmark at the reviewer's
request, so the response letter can report them on the same footing as Table 1. This is a
deliberate crossing of the train/test line, recorded so the provenance is unambiguous:

  * The alternatives were SCREENED on the development benchmark first. Nothing about their
    design, hyperparameters, or inclusion was chosen using held-out data.
  * FolDE itself is unchanged and is NOT being re-selected. Its published test-benchmark
    numbers (251003-test-benchmark_FolDE-base) stand as they are.
  * This run adds held-out numbers for two already-eliminated-or-screened variants. It does
    not alter what FolDE is.

base_config is copied verbatim from model_evals/251003-test-benchmark_FolDE-base.json, so the
ONLY thing varying across configs is how naturalness enters the few-shot model. In particular
it uses lie_noise_stddev_multiplier_schedule, NOT the flat lie_noise_stddev_multiplier=2.0 that
the 260926 development harness used -- those two are not comparable, and the assertions below
fail the run rather than let a mismatched config through.

Usage:
  python folde/scripts/261004_naturalness_alternatives_test.py all     # all 20 at once
  python folde/scripts/261004_naturalness_alternatives_test.py single  # the 17 singles
  python folde/scripts/261004_naturalness_alternatives_test.py multi   # union, resume skips singles
"""

import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

NAME = "261004-naturalness-alternatives-test"

SINGLE_DMS_IDS = [
    "A4GRB6_PSEAI_Chen_2020",
    "AMIE_PSEAE_Wrenbeck_2017",
    "CAS9_STRP1_Spencer_2017_positive",
    "HMDH_HUMAN_Jiang_2019",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "KKA2_KLEPN_Melnikov_2014",
    "LGK_LIPST_Klesmith_2015",
    "MET_HUMAN_Estevam_2023",
    "MLAC_ECOLI_MacRae_2023",
    "MSH2_HUMAN_Jia_2020",
    "MTHR_HUMAN_Weile_2021",
    "PAI1_HUMAN_Huttinger_2021",
    "PPARG_HUMAN_Majithia_2016",
    "PTEN_HUMAN_Mighell_2018",
    "RNC_ECOLI_Weeks_2023",
    "S22A1_HUMAN_Yee_2023_activity",
    "SC6A4_HUMAN_Young_2021",
]

MULTI_DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
]

ALL_TEST_DMS_IDS = SINGLE_DMS_IDS + MULTI_DMS_IDS

# Verbatim from 251003-test-benchmark_FolDE-base.json.
TEST_BENCHMARK_FEW_SHOT_PARAMS = {
    "pretrain": True,
    "pretrain_epochs": 50,
    "ensemble_size": 5,
    "embedding_dim": 960,
    "hidden_dims": [100, 50],
    "dropout": 0.2,
    "learning_rate": 0.0003,
    "weight_decay": 1e-05,
    "train_epochs": 200,
    "train_patience": 40,
    "val_frequency": 10,
    "do_validation_with_pair_fraction": 0.2,
    "decision_mode": "constantliar",
    "lie_noise_stddev_multiplier_schedule": [
        6.0,
        6.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    ],
}

base_config = FolDEModelConfig(
    name="FolDE",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="TorchMLPFewShotModel",
    few_shot_model_params=dict(TEST_BENCHMARK_FEW_SHOT_PARAMS),
)


def alternative_config(name: str, few_shot_model_name: str) -> FolDEModelConfig:
    """base_config with naturalness entering the few-shot model differently.

    pretrain=False: these REPLACE warm-starting rather than add to it.
    """
    params = dict(TEST_BENCHMARK_FEW_SHOT_PARAMS)
    params["pretrain"] = False
    return FolDEModelConfig(
        name=name,
        naturalness_model_id=base_config.naturalness_model_id,
        embedding_model_id=base_config.embedding_model_id,
        zero_shot_model_name=base_config.zero_shot_model_name,
        zero_shot_model_params={},
        few_shot_model_name=few_shot_model_name,
        few_shot_model_params=params,
    )


config_list = [
    alternative_config("FeatureConcat", "NaturalnessFeatureMLPFewShotModel"),
    alternative_config("ResidualCorrection", "NaturalnessResidualMLPFewShotModel"),
]

# The checkpoint helper requires a config's stored dataset IDs to be a subset of the requested
# ones, so the multi group requests the union and lets resume skip the singles.
GROUPS = {
    "all": ALL_TEST_DMS_IDS,
    "single": SINGLE_DMS_IDS,
    "multi": SINGLE_DMS_IDS + MULTI_DMS_IDS,
}


def assert_config_matches_test_benchmark() -> None:
    """Fail before running if the acquisition settings differ from the test benchmark.

    The 260926 development harness used a flat lie_noise_stddev_multiplier of 2.0. Running
    these alternatives with that setting and reporting them beside Table 1 would be comparing
    across two different acquisition schedules.
    """
    for config in config_list + [base_config]:
        params = config.few_shot_model_params
        assert "lie_noise_stddev_multiplier_schedule" in params, (
            f"{config.name}: missing lie_noise_stddev_multiplier_schedule -- the test "
            "benchmark uses a schedule, not a flat multiplier"
        )
        assert params["lie_noise_stddev_multiplier_schedule"] == [
            6.0,
            6.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ], f"{config.name}: schedule does not match the test benchmark"
        assert "lie_noise_stddev_multiplier" not in params, (
            f"{config.name}: lie_noise_stddev_multiplier must be ABSENT (the 260926 "
            "development harness set it to 2.0; that config is not comparable to Table 1)"
        )
        assert params["decision_mode"] == "constantliar", f"{config.name}: decision_mode"
        assert config.naturalness_model_id == "600m", f"{config.name}: naturalness_model_id"
        assert config.embedding_model_id == "300m", f"{config.name}: embedding_model_id"
        assert (
            config.zero_shot_model_name == "NaturalnessZeroShotModel"
        ), f"{config.name}: zero_shot_model_name"

    # Everything except how naturalness enters must be identical to base_config.
    for config in config_list:
        assert not config.few_shot_model_params[
            "pretrain"
        ], f"{config.name}: must run with pretrain=False -- these replace warm-starting"
        differing = {
            key
            for key in set(base_config.few_shot_model_params) | set(config.few_shot_model_params)
            if base_config.few_shot_model_params.get(key) != config.few_shot_model_params.get(key)
        }
        assert differing == {"pretrain"}, (
            f"{config.name}: differs from the test-benchmark config in {sorted(differing)}; "
            "only 'pretrain' may differ"
        )


def main() -> None:
    group = sys.argv[1] if len(sys.argv) > 1 else "all"
    if group not in GROUPS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(GROUPS)}>")
    dms_ids = GROUPS[group]

    assert_config_matches_test_benchmark()

    # The requested set must be exactly the 20 test targets (the single/multi groups are
    # execution splits of that same set, not different benchmarks).
    assert set(ALL_TEST_DMS_IDS) == set(SINGLE_DMS_IDS) | set(MULTI_DMS_IDS)
    assert len(ALL_TEST_DMS_IDS) == 20, f"expected 20 test targets, got {len(ALL_TEST_DMS_IDS)}"
    assert len(set(ALL_TEST_DMS_IDS)) == 20, "duplicate dataset id in the test list"
    assert set(dms_ids) <= set(ALL_TEST_DMS_IDS), (
        f"requested datasets outside the test benchmark: "
        f"{sorted(set(dms_ids) - set(ALL_TEST_DMS_IDS))}"
    )
    if group in ("all", "multi"):
        assert set(dms_ids) == set(
            ALL_TEST_DMS_IDS
        ), f"group '{group}' must cover all 20 test targets, got {len(set(dms_ids))}"

    print(f"[{group}] HELD-OUT test benchmark, {len(dms_ids)} of {len(ALL_TEST_DMS_IDS)} targets")
    print("  config matches 251003-test-benchmark_FolDE-base (schedule asserted)")
    for config in config_list:
        print(f"  - {config.name}: {config.few_shot_model_name} (pretrain=False)")

    simulate_campaigns_with_config_checkpoints(
        eval_prefix=NAME,
        dms_ids=dms_ids,
        config_list=config_list,
        checkpoint_dir="folde/model_evals",
        round_size=16,
        number_of_simulations=10,
        activity_column="DMS_score",
        max_rounds=6,
        random_seed=42,
        # SPG1's dense 300m pool only fits via the in-process path, so the multi split runs
        # single-worker. See the chunked loader in data.py.
        num_workers=int(os.environ.get("NAT_ALT_WORKERS", "4")),
    )
    print(f"NATURALNESS ALTERNATIVES TEST {group} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
