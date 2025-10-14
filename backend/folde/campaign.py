"""
Campaign simulation functions for protein engineering prediction tasks.

This module provides functions for simulating protein engineering campaigns
and evaluating different model configurations.
"""

import json
import logging
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from app.helpers.sequence_util import get_allele_set, get_loci_set, is_homolog_seq_id
from folde.data import get_proteingym_dataset
from folde.few_shot_models import get_consensus_scores, get_few_shot_model
from folde.types import (
    CampaignResult,
    FolDEModelConfig,
    ModelEvaluation,
    MutantMetrics,
    RoundMetrics,
    SimulationResult,
    SingleConfigCampaignResult,
)
from folde.util import (
    get_consensus_scores,
    get_top_percentile_recall_score,
    get_top_percentile_recall_score_slate,
    top_k_mask,
)
from folde.zero_shot_models import get_zero_shot_model
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score

logger = logging.getLogger(__name__)


class CampaignWorldState:
    def __init__(
        self,
        golden_activity_series: pd.Series,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        one_mutation_at_a_time: bool,
    ):
        assert golden_activity_series.index.equals(naturalness_df.index)
        assert golden_activity_series.index.equals(embedding_series.index)
        # Removing these copies shaves ~25% time off the faster runs, and more when
        # my laptop is memory constrained...
        self.golden_activity_series = golden_activity_series  # .copy()
        self.naturalness_df = naturalness_df  # .copy()
        self.embedding_series = embedding_series  # .copy()
        self.measured_seq_ids: List[str] = []
        self.one_mutation_at_a_time = one_mutation_at_a_time

    def measure_variant_activities(self, seq_ids: List[str]):
        """Adds seq ids to the collection of measured samples."""
        assert len(set(seq_ids)) == len(seq_ids), f"seq_ids must be unique, got {seq_ids}"
        for seq_id in seq_ids:
            assert type(seq_id) == str, f"seq_id must be a string, got {type(seq_id)} ({seq_id})"
            assert seq_id not in self.measured_seq_ids, f"seq_id {seq_id} already measured"
        self.measured_seq_ids.extend(seq_ids)

    def get_mutant_pool(self) -> List[str]:
        if self.one_mutation_at_a_time:
            measured_allele_sets = [get_allele_set(s) for s in self.measured_seq_ids]

            mutant_pool = []
            for s in self.golden_activity_series.index.tolist():
                if s in self.measured_seq_ids:
                    continue

                if is_homolog_seq_id(s):
                    mutant_pool.append(s)
                    continue

                allele_set = get_allele_set(s)
                if len(allele_set) == 1:
                    mutant_pool.append(s)
                    continue

                # See if this sequence is 1 away from any measured sequences.
                for measured_allele_set in measured_allele_sets:
                    if len(allele_set ^ measured_allele_set) == 1:
                        mutant_pool.append(s)
                        break

            return mutant_pool
        else:
            return [
                s
                for s in self.golden_activity_series.index.tolist()
                if s not in self.measured_seq_ids
            ]

    def get_unmeasured_activity_series(self) -> pd.Series:
        return self.golden_activity_series.loc[self.get_mutant_pool()]

    def get_unmeasured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df.loc[pd.Index(self.get_mutant_pool())]

    def get_unmeasured_embeddings_series(self) -> pd.Series:
        return self.embedding_series.loc[self.get_mutant_pool()]

    def get_measured_activity_series(self) -> pd.Series:
        return self.golden_activity_series.loc[self.measured_seq_ids]

    def get_measured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df.loc[pd.Index(self.measured_seq_ids)]

    def get_measured_embeddings_series(self) -> pd.Series:
        return self.embedding_series.loc[self.measured_seq_ids]


def _run_single_simulation(
    available_seq_ids: List[str],
    entire_activity_series: pd.Series,
    entire_naturalness_df: pd.DataFrame,
    entire_embedding_series: pd.Series,
    round_size: int,
    config: FolDEModelConfig,
    random_seed: int,
    wt_aa_seq: str,
    max_rounds: int = 10,
) -> SimulationResult:
    """Run a single campaign simulation.

    Args:
        golden_activity_series: Series with ground truth activity data (some of which may be NAN)
        naturalness_df: DataFrame with naturalness scores (some of which may be NAN)
        embedding_series: Series with embeddings
        round_size: Number of variants to test in each round
        config: Model configuration
        max_rounds: Maximum number of rounds to simulate

    Returns:
        Dictionary with simulation results
    """
    # Initialize results tracking
    results = SimulationResult(
        rounds=0,
        variant_pool_size=len(available_seq_ids),
        mutant_metrics=[],
        round_metrics=[],
    )

    assert entire_naturalness_df.index.equals(entire_activity_series.index)
    assert entire_naturalness_df.index.equals(entire_embedding_series.index)

    whole_world_activity_series = entire_activity_series.loc[available_seq_ids]
    whole_world_naturalness_df = entire_naturalness_df.loc[pd.Index(available_seq_ids)]
    whole_world_embedding_series = entire_embedding_series.loc[available_seq_ids]
    assert (
        not whole_world_activity_series.isna().any()
    ), f'{whole_world_activity_series.isna().sum()} activity values in the "whole world" set are NAN'
    world_state = CampaignWorldState(
        whole_world_activity_series,
        whole_world_naturalness_df,
        whole_world_embedding_series,
        config.one_mutation_at_a_time,
    )

    def is_single_mutant_id(seq_id: str) -> bool:
        if seq_id == "WT" or is_homolog_seq_id(seq_id):
            return False
        return len(get_loci_set(seq_id)) == 1

    single_mutant_seq_ids = [
        seq_id for seq_id in entire_naturalness_df.index if is_single_mutant_id(seq_id)
    ]

    pretraining_naturalness_df = entire_naturalness_df.loc[pd.Index(single_mutant_seq_ids)]
    pretraining_embedding_series = entire_embedding_series.loc[single_mutant_seq_ids]
    assert (
        not pretraining_naturalness_df.isna()
        .any()
        .any()  # Make sure SOME naturalness values are not NAN,
    ), f'{pretraining_naturalness_df.isna().sum()}/{len(pretraining_naturalness_df)} naturalness values in the "pretraining" set are NAN'

    held_out_series = (
        ~entire_activity_series.index.isin(available_seq_ids) & ~entire_activity_series.isna()
    )
    held_out_activity_series = entire_activity_series.loc[held_out_series]
    held_out_naturalness_df = entire_naturalness_df.loc[pd.Index(held_out_series)]
    held_out_embedding_series = entire_embedding_series.loc[held_out_series]

    # Get the zero-shot model
    zero_shot_model = get_zero_shot_model(
        config.zero_shot_model_name, **config.zero_shot_model_params
    )

    # Get few-shot model
    few_shot_model = get_few_shot_model(
        config.few_shot_model_name,
        random_state=random_seed,
        wt_aa_seq=wt_aa_seq,
        **config.few_shot_model_params,
    )

    zero_shot_model.pretrain(
        pretraining_naturalness_df,
        pretraining_embedding_series,
    )
    few_shot_model.pretrain(
        pretraining_naturalness_df,
        pretraining_embedding_series,
    )

    # Run the simulation for the specified number of rounds
    for round_num in range(1, max_rounds + 1):
        logger.debug(f"Running round {round_num}")

        if len(world_state.get_unmeasured_activity_series()) == 0:
            logger.info("No more unmeasured variants, ending simulation")
            break

        # We're getting a topN to synthesize, and a predicted activity for every variant.
        top_seq_ids = None
        predicted_activity_ensemble: List[pd.Series] = []
        held_out_prediction_ensemble: List[pd.Series] = []

        # First round: always use zero-shot model
        if round_num == 1:
            # Get top variants using zero-shot model's get_top_n method
            top_seq_ids, predicted_activity_ensemble = zero_shot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_series(),
            )

            held_out_batch_seq_ids, held_out_prediction_ensemble = zero_shot_model.get_top_n(
                round_size,
                held_out_naturalness_df,
                held_out_embedding_series,
            )

        # Subsequent rounds: use few-shot model if specified
        else:
            # Convert list of embeddings to numpy array
            train_activity_series = world_state.get_measured_activity_series()

            few_shot_model.fit(
                whole_world_naturalness_df,
                whole_world_embedding_series,
                train_activity_series,
                held_out_naturalness_df,
                held_out_embedding_series,
                held_out_activity_series,
            )

            # Use the get_top_n method from FewShotModel
            top_seq_ids, predicted_activity_ensemble = few_shot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_series(),
                round_number=round_num - 1,
            )

            held_out_batch_seq_ids, held_out_prediction_ensemble = few_shot_model.get_top_n(
                round_size,
                held_out_naturalness_df,
                held_out_embedding_series,
                round_number=round_num - 1,
            )

        # Update world state.
        assert type(top_seq_ids) == list, f"top_seq_ids must be a list, got {type(top_seq_ids)}"
        assert (
            len(top_seq_ids) == round_size
        ), f"Must choose {round_size} variants per rounds, only chose {len(top_seq_ids)}"
        logging.debug(
            f"In Round {round_num}: elected {len(top_seq_ids)} variants: {','.join(top_seq_ids)}"
        )
        world_state.measure_variant_activities(top_seq_ids)

        # Metric calculations.
        consensus_predicted_activity = get_consensus_scores(
            predicted_activity_ensemble, decision_mode="mean"
        )
        all_percentiles = whole_world_activity_series.rank(pct=True)

        mutant_metrics_list = []
        for top_seq_id in top_seq_ids:
            # Get the activity from the dataframe and convert to float if needed
            golden_activity = world_state.get_measured_activity_series().loc[top_seq_id]
            assert (
                type(golden_activity) == float or type(golden_activity) == np.float64
            ), f"golden_activity must be a float, got {type(golden_activity)}"
            percentile = all_percentiles.loc[top_seq_id]
            predicted_activity_stddev = float(
                np.std([pa.loc[top_seq_id] for pa in predicted_activity_ensemble])
            )
            mutant_metrics_list.append(
                MutantMetrics(
                    seq_id=top_seq_id,
                    round_found=round_num,
                    activity=float(golden_activity),
                    predicted_activity=consensus_predicted_activity.loc[top_seq_id],
                    predicted_activity_stddev=predicted_activity_stddev,
                    percentile=percentile,
                    relevant_mutants=[],  # TODO(jacob): Compute relevant mutants
                )
            )

        # Compute metrics for this round's predictions
        whole_dataset_spearman = spearmanr(
            entire_activity_series.loc[consensus_predicted_activity.index].values,
            consensus_predicted_activity.values,
        )[0]

        # Compute metrics over held-out mutants.
        consensus_held_out_predictions = get_consensus_scores(
            held_out_prediction_ensemble, decision_mode="mean"
        )
        assert held_out_activity_series.index.equals(consensus_held_out_predictions.index)
        assert (
            not held_out_activity_series.isna().any()
        ), f"{held_out_activity_series.isna().sum()}/{len(held_out_activity_series)} held out activity values are NAN"
        assert (
            not consensus_held_out_predictions.isna().any()
        ), f"{consensus_held_out_predictions.isna().sum()}/{len(consensus_held_out_predictions)} consensus held out predictions are NAN"

        held_out_activity_spearman = spearmanr(
            held_out_activity_series.values,
            consensus_held_out_predictions.values,
        )[0]

        batch_percentile_series = held_out_activity_series.rank(pct=True)[held_out_batch_seq_ids]
        held_out_batch_1pct = (batch_percentile_series >= 0.99).mean()
        held_out_batch_10pct = (batch_percentile_series >= 0.90).mean()

        def get_held_out_stats_for_percentile(percentile, slate_size):
            """Returns some stats on the held out predictions for a percentile, zero to 100 (eg 1.0 for top 1 percent)."""
            assert held_out_activity_series.index.equals(consensus_held_out_predictions.index)

            nonnull_activity_mask = held_out_activity_series.notna()

            assert (
                not held_out_activity_series[nonnull_activity_mask].isna().any()
            ), f"{held_out_activity_series[nonnull_activity_mask].isna().sum()}/{len(held_out_activity_series[nonnull_activity_mask])} held out activity values are NAN"
            assert (
                not consensus_held_out_predictions[nonnull_activity_mask].isna().any()
            ), f"{consensus_held_out_predictions[nonnull_activity_mask].isna().sum()}/{len(consensus_held_out_predictions[nonnull_activity_mask])} consensus held out predictions are NAN"

            num_seq_ids = len(held_out_activity_series[nonnull_activity_mask].index)
            num_nonduplicated_seq_ids = len(
                set(held_out_activity_series[nonnull_activity_mask].index)
            )
            assert (
                num_seq_ids == num_nonduplicated_seq_ids
            ), f"held_out_activity_series has duplicate indices ({num_seq_ids} vs {num_nonduplicated_seq_ids})!!!"

            held_out_stat_recall = get_top_percentile_recall_score(
                held_out_activity_series[nonnull_activity_mask].to_numpy(),
                consensus_held_out_predictions[nonnull_activity_mask].to_numpy(),
                percentile,
            )

            held_out_stat_recall_slate = get_top_percentile_recall_score_slate(
                held_out_activity_series[nonnull_activity_mask].to_numpy(),
                consensus_held_out_predictions[nonnull_activity_mask].to_numpy(),
                percentile,
                slate_size,
            )

            held_out_stat_auc = roc_auc_score(
                top_k_mask(held_out_activity_series[nonnull_activity_mask], percentile),
                consensus_held_out_predictions[nonnull_activity_mask],
            )

            logger.info(
                f"Computing the top {percentile} percentile recall score for {num_seq_ids} non-null seq_ids in the held out set of size {len(held_out_activity_series)} and a world size of {len(entire_activity_series)}: {held_out_stat_recall}"
            )

            return held_out_stat_recall, held_out_stat_recall_slate, held_out_stat_auc

        held_out_1pct_recall, held_out_1pct_recall_slate, held_out_1pct_auc = (
            get_held_out_stats_for_percentile(1, round_size)
        )
        held_out_10pct_recall, held_out_10pct_recall_slate, held_out_10pct_auc = (
            get_held_out_stats_for_percentile(10, round_size)
        )

        round_metrics = RoundMetrics(
            round_num=round_num,
            model_spearman=float(whole_dataset_spearman),  # type: ignore
            misc={
                "held_out_activity_spearman": float(held_out_activity_spearman),  # type: ignore
                "held_out_batch_1pct": float(held_out_batch_1pct),  # type: ignore
                "held_out_1pct_recall": float(held_out_1pct_recall),  # type: ignore
                "held_out_1pct_recall_slate": float(held_out_1pct_recall_slate),  # type: ignore
                "held_out_1pct_auc": float(held_out_1pct_auc),  # type: ignore
                "held_out_batch_10pct": float(held_out_batch_10pct),  # type: ignore
                "held_out_10pct_recall": float(held_out_10pct_recall),  # type: ignore
                "held_out_10pct_recall_slate": float(held_out_10pct_recall_slate),  # type: ignore
                "held_out_10pct_auc": float(held_out_10pct_auc),  # type: ignore
            },
        )

        if round_num == 1:
            round_metrics.misc["zero_shot_debug_info"] = zero_shot_model.get_debug_info()
        else:
            round_metrics.misc["few_shot_debug_info"] = few_shot_model.get_debug_info()

        results.rounds = round_num
        results.round_metrics.append(round_metrics)
        results.mutant_metrics.extend(mutant_metrics_list)

    return results


# Run multiple simulations
def run_single_sim_parallel(
    sim_idx,
    available_seq_ids: List[str],
    activity_series: pd.Series,
    naturalness_df: pd.DataFrame,
    embedding_series: pd.Series,
    sim_random_seed,
    **kwargs,
):
    logger.info(
        f"Running simulation {sim_idx+1} ({len(available_seq_ids)} / {len(activity_series)} mutants in sim)"
    )
    random.seed(sim_random_seed)
    np.random.seed(sim_random_seed)

    sim_result = _run_single_simulation(
        available_seq_ids,
        activity_series,
        naturalness_df,
        embedding_series,
        random_seed=sim_random_seed,
        **kwargs,
    )
    return sim_result


def simulate_campaign(
    dms_id: str,
    round_size: int,
    number_of_simulations: int,
    config_list: List[FolDEModelConfig],
    activity_column: str = "DMS_score",
    max_rounds: int = 10,
    random_seed: int = 42,
    num_workers: int = 10,
) -> CampaignResult:
    """Simulate protein engineering campaigns with different model configurations.

    Args:
        dms_id: Identifier for the DMS dataset to use
        round_size: Number of variants to evaluate in each round
        number_of_simulations: Number of times to run the simulation
        config_list: List of model configurations to evaluate
        activity_column: Column in the dataset containing activity values
        max_rounds: Maximum number of rounds to simulate
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing simulation results for each configuration
    """

    assert len(set([config.name for config in config_list])) == len(
        config_list
    ), f"Some configs have duplicate names."

    # Initialize results
    campaign_result = CampaignResult(
        dms_id=dms_id,
        round_size=round_size,
        number_of_simulations=number_of_simulations,
        activity_column=activity_column,
        max_rounds=max_rounds,
        random_seed=random_seed,
        min_activity=0.0,
        median_activity=0.0,
        max_activity=0.0,
        config_results=[],
    )

    df_cache = {}

    # Run simulations for each configuration
    for config_idx, model_config in enumerate(config_list):
        # Set random seed for reproducibility
        logger.info(f"Running simulations for configuration {config_idx+1}/{len(config_list)}")
        logger.info(f"Config: {model_config}")

        # Load dataset for this configuration
        cache_key = (model_config.embedding_model_id, model_config.naturalness_model_id)
        if cache_key not in df_cache:
            try:
                df_cache[cache_key] = get_proteingym_dataset(
                    dms_id,
                    model_config.embedding_model_id,
                    model_config.naturalness_model_id,
                )
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                import traceback

                print(traceback.format_exc(), flush=True)
                raise e
        wt_aa_seq, entire_naturalness_df, embedding_df, activity_df, category_df = df_cache[
            cache_key
        ]
        naturalness_ensemble_df = entire_naturalness_df[
            (
                ["log_wt_marginal"]
                if model_config.naturalness_columns is None
                else model_config.naturalness_columns
            )
        ]
        embedding_series = embedding_df[
            "embedding" if model_config.embedding_column is None else model_config.embedding_column
        ]
        activity_series = activity_df[activity_column]

        # Check that the activity column exists
        if activity_column not in activity_df.columns:
            raise ValueError(
                f"Activity column {activity_column} not found in dataset: {activity_df.columns}"
            )

        # Store some activity stats.
        campaign_result.min_activity = activity_df[activity_column].min(skipna=True)
        campaign_result.median_activity = activity_df[activity_column].median(skipna=True)
        campaign_result.max_activity = activity_df[activity_column].max(skipna=True)
        assert not np.isnan(campaign_result.min_activity)
        assert not np.isnan(campaign_result.median_activity)
        assert not np.isnan(campaign_result.max_activity)

        single_model_campaign_results = None
        print(
            f"NUMBER OF WORKERS: {num_workers} NUMBER OF WORKERS: {num_workers} NUMBER OF WORKERS: {num_workers} NUMBER OF WORKERS: {num_workers}"
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # with ThreadPoolExecutor() as executor:
            futures = []
            for sim_idx in range(number_of_simulations):
                if model_config.data_split_mode:
                    if not model_config.data_split_mode in category_df.columns:
                        raise ValueError(
                            f"Data split mode {model_config.data_split_mode} not found in category_df.columns: {category_df.columns}"
                        )
                    full_seq_id_list = list(
                        category_df[category_df[model_config.data_split_mode]].index.values
                    )
                else:
                    full_seq_id_list = list(
                        activity_df[activity_df[activity_column].notna()].index.values
                    )

                world_size = int(len(full_seq_id_list) * 0.5)

                if world_size < max_rounds * round_size:
                    raise ValueError(
                        f"World size {world_size} is less than max_rounds * round_size {max_rounds * round_size}"
                    )

                rng = np.random.RandomState(random_seed + 1000 * sim_idx)
                bootstrapped_seq_ids = rng.choice(
                    full_seq_id_list,
                    size=world_size,
                    replace=False,
                )

                futures.append(
                    executor.submit(
                        run_single_sim_parallel,
                        sim_idx,
                        bootstrapped_seq_ids.tolist(),  # type: ignore
                        activity_series,
                        naturalness_ensemble_df,
                        embedding_series,
                        sim_random_seed=random_seed + sim_idx,
                        round_size=round_size,
                        config=model_config,
                        max_rounds=max_rounds,
                        wt_aa_seq=wt_aa_seq,
                    )
                )
            single_model_campaign_results = [f.result() for f in futures]

        campaign_result.config_results.append(
            SingleConfigCampaignResult(
                config=model_config,
                simulation_results=single_model_campaign_results,
            )
        )

    return campaign_result


def simulate_campaigns(name: str, dms_ids: List[str], **kwargs) -> ModelEvaluation:
    """Run a list of campaigns."""
    results = ModelEvaluation(name=name, campaign_results=[])
    for dms_id in dms_ids:
        results.campaign_results.append(simulate_campaign(dms_id, **kwargs))
    return results


# --------------------------------------------------------------------------- #
# New, config‑centric checkpointing helper
# --------------------------------------------------------------------------- #
def simulate_campaigns_with_config_checkpoints(
    eval_prefix: str,
    dms_ids: List[str],
    config_list: List[FolDEModelConfig],
    checkpoint_dir: str,
    overwrite: bool = False,
    **kwargs,
) -> Dict[str, ModelEvaluation]:
    """Run campaigns with *per-config* checkpoint files.

    Each ``FolDEModelConfig`` gets its own ``ModelEvaluation`` JSON file named
    ``{eval_prefix}_{config_name}.json`` in ``checkpoint_dir``.  The outer loop
    iterates over configs so the expensive dataset loading happens only once
    per config.

    Parameters
    ----------
    eval_prefix
        Prefix for checkpoint filenames.
    dms_ids
        List of DMS dataset identifiers to evaluate.
    config_list
        Ordered list of model configs to evaluate.
    checkpoint_dir
        Where to store ``*.json`` checkpoint files.
    overwrite
        If *True*, always start fresh for every config even if a checkpoint is
        present.
    **kwargs
        Passed straight through to :func:`simulate_campaign`.  Must include
        ``round_size`` and ``number_of_simulations`` at minimum.

    Returns
    -------
    Dict[str, ModelEvaluation]
        Mapping ``config.name -> ModelEvaluation`` with all results.
    """
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # Validate config names early
    for cfg in config_list:
        cfg_name = cfg.name
        if not re.match(r"^[A-Za-z0-9.\-]+$", cfg_name):
            raise ValueError(
                f"Config name '{cfg_name}' contains invalid characters. "
                "Allowed characters are A-Z, a-z, 0-9, hyphen (-) and period (.). "
                "Underscores and other characters are not permitted."
            )

    # ------------------------------------------------------------------ #
    # First pass – validate existing checkpoints so we fail fast on
    # conflicts *before* starting any expensive work.
    # ------------------------------------------------------------------ #
    for cfg in config_list:
        cfg_name = cfg.name
        cp_path = cp_dir / f"{eval_prefix}_{cfg_name}.json"

        if cp_path.exists() and not overwrite:
            with cp_path.open() as f:
                data = json.load(f)
            eval_obj = ModelEvaluation.model_validate(data)

            # Sanity‑check that the stored config matches exactly
            if not eval_obj.campaign_results:
                raise ValueError(f"Checkpoint {cp_path} exists but contains no campaign_results.")
            stored_cfg = eval_obj.campaign_results[0].config_results[0].config
            if stored_cfg.model_dump() != cfg.model_dump():
                # Get the model dumps for comparison
                stored_dump = stored_cfg.model_dump()
                current_dump = cfg.model_dump()

                # Find differences
                differences = []
                all_keys = set(stored_dump.keys()) | set(current_dump.keys())

                for key in sorted(all_keys):
                    if key not in stored_dump:
                        differences.append(
                            f"  - '{key}': missing in stored config, current value: {current_dump[key]}"
                        )
                    elif key not in current_dump:
                        differences.append(
                            f"  - '{key}': missing in current config, stored value: {stored_dump[key]}"
                        )
                    elif stored_dump[key] != current_dump[key]:
                        differences.append(
                            f"  - '{key}': stored={stored_dump[key]}, current={current_dump[key]}"
                        )

                diff_msg = (
                    "\n".join(differences)
                    if differences
                    else "No specific differences found (possibly nested object differences)"
                )

                raise ValueError(
                    f"Config mismatch for checkpoint {cp_path}.\n"
                    f"Differences found:\n{diff_msg}\n"
                    "Pass overwrite=True or pick a new prefix."
                )

            # Ensure stored DMS IDs are a subset of the requested list
            stored_dms = {cr.dms_id for cr in eval_obj.campaign_results}
            if not stored_dms.issubset(set(dms_ids)):
                raise ValueError(
                    f"Checkpoint {cp_path} contains DMS IDs not requested in this run "
                    f"({sorted(stored_dms - set(dms_ids))})."
                )

    # ------------------------------------------------------------------ #
    # Second pass – run / resume each config
    # ------------------------------------------------------------------ #
    all_evals: Dict[str, ModelEvaluation] = {}
    for cfg in config_list:
        cfg_name = cfg.name
        cp_path = cp_dir / f"{eval_prefix}_{cfg_name}.json"

        if cp_path.exists() and not overwrite:
            with cp_path.open() as f:
                data = json.load(f)
            eval_obj = ModelEvaluation.model_validate(data)
            completed_dms = {cr.dms_id for cr in eval_obj.campaign_results}
            logger.info(
                f"Resuming config '{cfg_name}' with {len(completed_dms)} / "
                f"{len(dms_ids)} DMS datasets complete."
            )
        else:
            eval_obj = ModelEvaluation(name=f"{eval_prefix}_{cfg_name}", campaign_results=[])
            completed_dms: set[str] = set()

        # Inner loop over DMS datasets
        for dms_id in dms_ids:
            if dms_id in completed_dms:
                logger.info(f"[{cfg_name}] Skipping already‑completed DMS '{dms_id}'.")
                continue

            logger.info(f"[{cfg_name}] Simulating DMS '{dms_id}'.")
            eval_obj.campaign_results.append(
                simulate_campaign(
                    dms_id,
                    config_list=[cfg],
                    **kwargs,
                )
            )

            # Write / update checkpoint
            with cp_path.open("w") as f:
                json.dump(eval_obj.model_dump(), f, indent=2)

        all_evals[cfg_name] = eval_obj

    return all_evals
