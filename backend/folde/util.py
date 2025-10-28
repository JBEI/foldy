import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame
from pandas.core.frame import DataFrame
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsRegressor

from app.helpers.sequence_util import is_homolog_seq_id
from folde.types import FolDEModelConfig, ModelDiff, ModelEvaluation

DMS_SHORTNAMES = {
    "BLAT_ECOLX_Firnberg_2014": "BLAT_ECOLX",
    "ANCSZ_Hobbs_2022": "ANCSZ",
    "HXK4_HUMAN_Gersing_2022_activity": "HXK4_HUMAN",
    "OXDA_RHOTO_Vanella_2023_activity": "OXDA_RHOTO",
    "SHOC2_HUMAN_Kwon_2022": "SHOC2_HUMAN",
    "ADRB2_HUMAN_Jones_2020": "ADRB2_HUMAN",
    "CBS_HUMAN_Sun_2020": "CBS_HUMAN",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin": "P53_Null",
    "HSP82_YEAST_Flynn_2019": "HSP82_YEAST",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin": "P53_WT",
    "HEM3_HUMAN_Loggerenberg_2023": "HEM3_HUMAN",
    "PPM1D_HUMAN_Miller_2022": "PPM1D_HUMAN",
    "SPG1_STRSG_Olson_2014": "SPG1",
    "ADRB2_HUMAN_Jones_2020": "ADRB2_HUMAN",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin": "P53_HUMAN_Null",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin": "P53_HUMAN_WT",
    "MK01_HUMAN_Brenan_2016": "MK01_HUMAN",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function": "KCNJ2_MOUSE",
    "CAS9_STRP1_Spencer_2017_positive": "CAS9_STRP1",
    "SC6A4_HUMAN_Young_2021": "SC6A4_HUMAN",
    "PTEN_HUMAN_Mighell_2018": "PTEN_HUMAN",
    "S22A1_HUMAN_Yee_2023_activity": "S22A1_HUMAN",
    "KKA2_KLEPN_Melnikov_2014": "KKA2_KLEPN",
    "PPARG_HUMAN_Majithia_2016": "PPARG_HUMAN",
    "MET_HUMAN_Estevam_2023": "MET_HUMAN",
    "MTHR_HUMAN_Weile_2021": "MTHR_HUMAN",
    "LGK_LIPST_Klesmith_2015": "LGK_LIPST",
    "AMIE_PSEAE_Wrenbeck_2017": "AMIE_PSEAE",
    "SPG1_STRSG_Olson_2014": "SPG1_Olson",
    "PABP_YEAST_Melamed_2013": "PABP_YEAST",
    "GRB2_HUMAN_Faure_2021": "GRB2_HUMAN",
    "PAI1_HUMAN_Huttinger_2021": "PAI1_HUMAN",
    "A4GRB6_PSEAI_Chen_2020": "A4GRB6_PSEAI",
    "MSH2_HUMAN_Jia_2020": "MSH2_HUMAN",
    "MLAC_ECOLI_MacRae_2023": "MLAC_ECOLI",
    "RNC_ECOLI_Weeks_2023": "RNC_ECOLI",
    "HMDH_HUMAN_Jiang_2019": "HMDH_HUMAN",
}


def get_consensus_scores(pred_list: List[pd.Series], decision_mode: str) -> pd.Series:
    """Get the prediction of an ensemble using deicision mode (often max or median)."""
    pred_arr = np.stack([preds.to_numpy() for preds in pred_list])
    if decision_mode == "max":
        consensus_score_arr = np.max(pred_arr, axis=0)
    elif decision_mode == "ucb":
        consensus_score_arr = np.mean(pred_arr, axis=0) + np.std(pred_arr, axis=0)
    elif decision_mode == "median":
        consensus_score_arr = np.median(pred_arr, axis=0)
    elif decision_mode == "mean":
        consensus_score_arr = np.mean(pred_arr, axis=0)
    else:
        raise ValueError(f"Invalid decision mode {decision_mode}")
    return pd.Series(consensus_score_arr, index=pred_list[0].index).astype(float)  # type: ignore


def internal_sample_n_indices(
    scores: Union[List[float], NDArray[np.float64]],
    n: int,
    temperature: float = 0.0,
    epsilon: float = 0.0,
) -> List[int]:
    """Given a list of scores and a number of samples: draw n samples as indices.

    Arguments:
      scores: Array of scores for each item
      n: Number of items to sample
      temperature: if nonzero, do boltzmann sampling.
      epsilon: if nonzero, do epsilon sampling.
    """

    chosen_indices = []

    if epsilon > 0:
        # Draw some samples randomly.
        epsilon_choices = np.random.choice(len(scores), size=int(epsilon * n), replace=False)
        chosen_indices.extend(epsilon_choices.tolist())  # type: ignore

    if temperature < 1e-6:
        # Deterministic top-N selection (argmax without replacement)
        for top_ranked_index in np.argsort(scores)[::-1]:
            if top_ranked_index not in chosen_indices:
                chosen_indices.append(top_ranked_index)
                if len(chosen_indices) == n:
                    break
    else:
        try:
            scores = np.array(scores)
            remaining_indices = [ii for ii in range(len(scores)) if ii not in chosen_indices]
            while len(chosen_indices) < n:
                # Compute softmax over remaining scores
                # Convert the list to an array for proper indexing
                remaining_indices_arr = np.array(remaining_indices)
                remaining_scores = scores[remaining_indices_arr]
                probs = softmax(remaining_scores / temperature)

                # Sample one index
                choice = np.random.choice(len(remaining_indices), p=probs)
                chosen_indices.append(remaining_indices[choice])

                # Remove selected index
                del remaining_indices[choice]
        except Exception as e:
            logging.error(f"Error in boltzmann sampling. Returning top {n} variants: {e}")
            # Fallback to deterministic top-N selection
            for top_ranked_index in np.argsort(scores)[::-1]:
                if top_ranked_index not in chosen_indices:
                    chosen_indices.append(top_ranked_index)
                    if len(chosen_indices) == n:
                        break

    assert (
        len(chosen_indices) == n
    ), f"This code should always produce n ({n}) indices, but produced {len(chosen_indices)}"

    assert len(set(chosen_indices)) == len(
        chosen_indices
    ), f"chosen_indices must be unique, got {chosen_indices}"

    return chosen_indices


def constant_liar_sample(
    ensemble_preds: np.ndarray,
    seq_ids: np.ndarray,
    q_slate_size: int,
    lie_noise_stddev_multiplier: float,
    choice_of_baseline: str = "min",
    ucb_beta: float = 2.0,
) -> list[str]:
    """
    The "Constant Liar” approximation to the parallel EI acquisition function.

    Args:
        ensemble_preds: np.ndarray, shape (S, N)
        seq_ids: np.ndarray, shape (N,)
        q_slate_size: int, number of samples to draw
        lie_noise_stddev_multiplier: float, multiplier for lie noise stddev
        choice_of_baseline: str, 'min', 'mean', or 'max'
        ucb_beta: float, beta parameter for UCB
    """
    MAX_POINTS_TO_CONSIDER = 5000
    if ensemble_preds.ndim != 2:
        raise ValueError(f"ensemble_preds must be a 2D array, got shape {ensemble_preds.shape}")
    if ensemble_preds.shape[0] != len(seq_ids):
        raise ValueError(
            f"ensemble_preds must have the same number of rows as seq_ids, got {ensemble_preds.shape[0]} and {len(seq_ids)}"
        )
    if ensemble_preds.shape[0] > MAX_POINTS_TO_CONSIDER:
        raise ValueError(
            f"ensemble_preds must have at most {MAX_POINTS_TO_CONSIDER} rows, got {ensemble_preds.shape[0]}"
        )
    if ensemble_preds.shape[0] < q_slate_size:
        raise ValueError(
            f"ensemble_preds must have at least q_slate_size rows, got {ensemble_preds.shape[0]} vs {q_slate_size}"
        )
    if ensemble_preds.shape[1] < 3:
        raise ValueError(
            f"Calculating a good variance requires at least 3 models, got {ensemble_preds.shape[1]}"
        )

    pred_tensor = torch.tensor(ensemble_preds.T, dtype=torch.float64)  # (S, N)

    S, N = pred_tensor.shape
    lie_noise_variance = (lie_noise_stddev_multiplier * pred_tensor.std(dim=0).median().item()) ** 2

    # ────────────────────── compute empirical prior mean/covariance ─────────────
    prior_mean = pred_tensor.mean(dim=0)  # μ₀, shape (N,)
    devs = pred_tensor - prior_mean  # centred matrix (S, N)
    Cov = (devs.T @ devs) / S  # empirical covariance, rank ≤ S
    Cov += lie_noise_variance * torch.eye(N)  # ← “nugget” ensures PSD & noise floor

    if choice_of_baseline == "min":
        L = prior_mean.min().item()
    elif choice_of_baseline == "mean":
        L = prior_mean.mean().item()
    elif choice_of_baseline == "max":
        L = prior_mean.max().item()
    else:
        raise ValueError(f"Invalid choice of baseline {choice_of_baseline}")

    # Diagonal variance and standard deviation (already ≥ SIGMA_N2)
    vars = Cov.diag()
    sigmas = vars.sqrt()

    # ───────────────────────────── constant-liar loop ───────────────────────────
    selected = []  # indices of chosen items

    for _ in range(q_slate_size):

        # 1) Upper-confidence-bound score for every unpicked item
        ucb = prior_mean + ucb_beta * sigmas
        ucb[selected] = -torch.inf  # mask already-selected indices

        # 2) Greedily take the arg-max
        idx = int(torch.argmax(ucb))
        selected.append(idx)
        # print(f"Picked {seq_ids[idx]}  with UCB={ucb[idx]:.3f}")
        logging.info(
            f"Selecting {seq_ids[idx]} (original rank {idx+1}), score: {ucb[idx]} = {prior_mean[idx]} + {ucb_beta} * {sigmas[idx]}"
        )

        # 3) Single-point GP update with *fake* observation y=L at index idx
        k_i = Cov[:, idx].clone()  # column vector k(·, x_i)
        v_i = vars[idx].item()  # marginal var at x_i  (≥ σ_n²)

        # Posterior mean:   μ ← μ + k_i (L − μ_i) / v_i
        delta = (L - prior_mean[idx]) / v_i
        prior_mean = prior_mean + k_i * delta

        # Posterior covariance (rank-1 Downdate):  Σ ← Σ − k_i k_iᵀ / v_i
        Cov = Cov - torch.outer(k_i, k_i) / v_i
        Cov = 0.5 * (Cov + Cov.T)  # re-symmetrise to kill FP drift

        # Refresh variance/std-dev
        vars = Cov.diag()
        sigmas = vars.sqrt()

    # ─────────────────────────── map indices back to IDs ────────────────────────
    constant_liar_chosen_seq_ids = seq_ids[selected]  # final slate of length Q
    return constant_liar_chosen_seq_ids.tolist()


def top_k_mask(series: pd.Series, percentile: float) -> pd.Series:
    k = max(1, int(np.ceil(len(series) * percentile / 100)))
    top_idx = series.nlargest(k).index  # strict ranking
    out = pd.Series(False, index=series.index)
    out.loc[top_idx] = True
    return out


def get_top_percentile_recall_score(target: np.ndarray, pred: np.ndarray, pct: float) -> float:
    target = np.asarray(target).ravel()  # <-- makes it 1-D
    pred = np.asarray(pred).ravel()
    assert target.size == pred.size, "arrays must be same length"

    n = target.size
    k = max(1, int(np.ceil(n * pct / 100)))
    assert (
        k <= n
    ), f"k must be less than or equal to n, got k={k} and n={n}. target shape {target.shape}, pred shape {pred.shape} pct {pct}"

    top_tgt = np.argpartition(target, n - k)[n - k :]
    top_prd = np.argpartition(pred, n - k)[n - k :]

    # recall = |intersection| / k
    recall = np.intersect1d(top_tgt, top_prd).size / k
    # print(f'n: {n}, k: {k}, top_tgt size: {top_tgt.size}, top_prd size: {top_prd.size}, overlap size: {np.intersect1d(top_tgt, top_prd).size}, recall: {recall}')
    return recall


def get_top_percentile_recall_score_slate(
    target: np.ndarray, pred: np.ndarray, pct: float, slate_size: int
) -> float:
    target = np.asarray(target).ravel()  # <-- makes it 1-D
    pred = np.asarray(pred).ravel()
    assert target.size == pred.size, "arrays must be same length"

    n = target.size
    num_top_pct_mutants = max(1, int(np.ceil(n * pct / 100)))
    assert (
        num_top_pct_mutants <= n
    ), f"k must be less than or equal to n, got k={num_top_pct_mutants} and n={n}. target shape {target.shape}, pred shape {pred.shape} pct {pct}"
    assert (
        num_top_pct_mutants >= slate_size
    ), f"num_top_pct_mutants must be greater than or equal to slate_size, got k={num_top_pct_mutants} and slate_size={slate_size}. target shape {target.shape}, pred shape {pred.shape} pct {pct}"

    top_tgt = np.argpartition(target, n - num_top_pct_mutants)[n - num_top_pct_mutants :]
    top_prd = np.argpartition(pred, n - slate_size)[n - slate_size :]

    # recall = |intersection| / slate_size
    return float(np.intersect1d(top_tgt, top_prd).size) / float(slate_size)


def convert_compaign_result_collection_to_df(
    model_evaluation: ModelEvaluation,
) -> Tuple[DataFrame, DataFrame]:
    """Convert a CampaignResultCollection to mutant and round metrics dataframes.

    Args:
        model_evaluation: The ModelEvaluation object containing campaign results

    Returns:
        A tuple containing:
            - DataFrame with mutant metrics
            - DataFrame with round metrics
    """
    mutant_metrics_df_list = []
    round_metrics_df_list = []
    for campaign_result in model_evaluation.campaign_results:
        dms_id = campaign_result.dms_id
        for result in campaign_result.config_results:
            config = result.config
            for sim_num, sim_result in enumerate(result.simulation_results):
                mutant_metric_list = []
                for mutant_metric in sim_result.mutant_metrics:
                    mutant_metric_list.append(
                        {
                            "dms_id": dms_id,
                            "config_name": config.name,
                            "sim_num": sim_num,
                            **mutant_metric.model_dump(),
                        }
                    )
                mutant_metric_df = pd.DataFrame(mutant_metric_list)
                mutant_metrics_df_list.append(mutant_metric_df)

                round_metrics_list = []
                for round_metrics in sim_result.round_metrics:
                    round_num = round_metrics.round_num
                    mutants_this_round = mutant_metric_df[mutant_metric_df.round_found == round_num]
                    normalized_best_activity_this_round = (
                        mutants_this_round.activity.max() - campaign_result.min_activity
                    ) / (campaign_result.max_activity - campaign_result.min_activity)
                    mutants_so_far = mutant_metric_df[mutant_metric_df.round_found <= round_num]
                    mutants_this_round = mutant_metric_df[mutant_metric_df.round_found == round_num]
                    best_activity_so_far = mutants_so_far.activity.max()
                    normalized_best_activity_so_far = (
                        best_activity_so_far - campaign_result.min_activity
                    ) / (campaign_result.max_activity - campaign_result.min_activity)
                    round_metrics_list.append(
                        {
                            "dms_id": dms_id,
                            "config_name": config.name,
                            "sim_num": sim_num,
                            "variant_pool_size": sim_result.variant_pool_size,
                            "best_activity_this_round": mutants_this_round.activity.max(),
                            "best_percentile_this_round": normalized_best_activity_this_round,
                            "best_activity_so_far": best_activity_so_far,
                            "normalized_best_activity_so_far": normalized_best_activity_so_far,
                            "best_percentile_so_far": mutants_so_far.percentile.max(),
                            "cumulative_1pct_hits": (mutants_so_far.percentile >= 0.99).sum(),
                            "cumulative_10pct_hits": (mutants_so_far.percentile >= 0.90).sum(),
                            "new_1pct_hits": (mutants_this_round.percentile >= 0.99).sum(),
                            "frac_1pct_hits": (mutants_this_round.percentile >= 0.99).mean(),
                            "new_10pct_hits": (mutants_this_round.percentile >= 0.90).sum(),
                            "frac_10pct_hits": (mutants_this_round.percentile >= 0.90).mean(),
                            "has_found_top_1pct": (mutants_so_far.percentile >= 0.99).sum() > 0,
                            **round_metrics.model_dump(),
                            **round_metrics.misc,
                        }
                    )
                round_metrics_df = pd.DataFrame(round_metrics_list)
                round_metrics_df_list.append(round_metrics_df)

    mutant_df = pd.concat(mutant_metrics_df_list)
    round_metrics_df = pd.concat(round_metrics_df_list)

    return mutant_df, round_metrics_df


def get_training_loss_df(results: ModelEvaluation, round_idx: int) -> pd.DataFrame:
    train_df = pd.DataFrame()
    for campaign_results in results.campaign_results:
        for config_results in campaign_results.config_results:
            for sim_idx, sim_result in enumerate(config_results.simulation_results):
                few_shot_info = sim_result.round_metrics[round_idx].misc["few_shot_debug_info"]
                for model_idx in range(len(few_shot_info["pretrain_metrics"])):
                    pretrain_train_loss_list = few_shot_info["pretrain_metrics"][model_idx][
                        "train_loss"
                    ]
                    pretrain_val_loss_list = few_shot_info["pretrain_metrics"][model_idx][
                        "val_loss"
                    ]
                    finetune_train_loss_list = few_shot_info["finetune_metrics"][model_idx][
                        "train_loss"
                    ]
                    finetune_val_loss_list = few_shot_info["finetune_metrics"][model_idx][
                        "val_loss"
                    ]
                    finetune_test_recall_1pct_list = few_shot_info["finetune_metrics"][model_idx][
                        "test_recall_1pct"
                    ]
                    model_train_df = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "loss_type": "pretrain_train",
                                    "log_step": list(range(len(pretrain_train_loss_list))),
                                    "loss": pretrain_train_loss_list,
                                }
                            ),
                            pd.DataFrame(
                                {
                                    "loss_type": "pretrain_val",
                                    "log_step": list(range(len(pretrain_val_loss_list))),
                                    "loss": pretrain_val_loss_list,
                                }
                            ),
                            pd.DataFrame(
                                {
                                    "loss_type": "finetune_train",
                                    "log_step": list(range(len(finetune_train_loss_list))),
                                    "loss": finetune_train_loss_list,
                                }
                            ),
                            pd.DataFrame(
                                {
                                    "loss_type": "finetune_val",
                                    "log_step": list(range(len(finetune_val_loss_list))),
                                    "loss": finetune_val_loss_list,
                                }
                            ),
                            pd.DataFrame(
                                {
                                    "loss_type": "finetune_recall_1pct",
                                    "log_step": list(range(len(finetune_test_recall_1pct_list))),
                                    "loss": finetune_test_recall_1pct_list,
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

                    model_train_df["model_idx"] = model_idx
                    model_train_df["sim_idx"] = sim_idx
                    model_train_df["dms_id"] = campaign_results.dms_id
                    model_train_df["config_name"] = config_results.config.name
                    train_df = pd.concat([train_df, model_train_df])
    return train_df


def apply_diff_to_dict_recursive(
    config_dict: dict[str, Any],
    path_components: list[str],
    new_value: Any,
) -> None:
    assert isinstance(path_components, list)
    if len(path_components) == 1:
        config_dict[path_components[0]] = new_value
        return
    else:
        next_component = path_components[0]
        if next_component not in config_dict:
            raise ValueError(f"Component {next_component} not found in config_dict")
        if type(config_dict[next_component]) is not dict:
            raise ValueError(f"Component {next_component} is not a dict")
        apply_diff_to_dict_recursive(config_dict[next_component], path_components[1:], new_value)


def apply_diff_list_to_config(
    folde_model_config_base: FolDEModelConfig,
    model_diffs: List[ModelDiff],
    exclude_base_config: bool = False,
) -> List[FolDEModelConfig]:
    original_config = folde_model_config_base.model_copy(
        deep=True, update={"name": folde_model_config_base.name + "-base"}
    )
    if exclude_base_config:
        config_list = []
    else:
        config_list = [original_config]
    for model_diff in model_diffs:
        folde_model_config_dict = folde_model_config_base.model_dump()
        for param_path, new_value in model_diff.diffs.items():
            apply_diff_to_dict_recursive(
                folde_model_config_dict,
                param_path.split("."),
                new_value,
            )
        folde_model_config = FolDEModelConfig(**folde_model_config_dict)
        folde_model_config.name = f"{folde_model_config_base.name}-{model_diff.name}"
        config_list.append(folde_model_config)
    return config_list


def cluster_sort_seq_ids(
    chosen_pred_df: pd.DataFrame,
    method: str = "average",
) -> list[str]:
    """
    Sort the rows of a prediction DataFrame by hierarchical clustering on the
    pair-wise Pearson correlation between mutants.

    Parameters
    ----------
    chosen_pred_df : pd.DataFrame
        One row per mutant (index = seq_id) and columns ``model_0 … model_N``
        containing the ensemble’s activity predictions.
    method : str, optional
        Linkage method passed straight through to
        ``scipy.cluster.hierarchy.linkage`` (default ``"average"``).

    Returns
    -------
    list[str]
        The seq_ids ordered according to the dendrogram’s leaf order.
    """
    # Trivial cases ─ nothing to cluster
    if len(chosen_pred_df) <= 1:
        return chosen_pred_df.index.tolist()

    # 1. Pair-wise Pearson r between mutants (rows)
    corr = chosen_pred_df.T.corr()  # (n_mutants × n_mutants)

    # 2. Convert to condensed distance vector: d = 1 − r
    dist_condensed = squareform(1.0 - corr.values, checks=False)

    # 3. Hierarchical clustering with optimal leaf ordering
    Z = linkage(dist_condensed, method=method, optimal_ordering=True)

    # 4. Leaf order → index positions → seq_id list
    leaf_order = leaves_list(Z)  # ndarray of row indices
    return chosen_pred_df.index[leaf_order].tolist()


def load_checkpointed_model_eval(
    checkpoint_dir,
    eval_prefix: str,
    other_eval_prefixes: list[str] = [],
) -> ModelEvaluation:
    """Load a model evaluation from a checkpoint directory.

    Args:
        checkpoint_dir: The directory containing the checkpoint files.
        eval_prefix: The prefix of the checkpoint files.
        other_eval_prefixes: A list of other prefixes to load.
    """
    result_list = []
    for json_path in Path(checkpoint_dir).glob(f"{eval_prefix}*"):
        with open(json_path, "r") as f:
            result_list.append(ModelEvaluation(**json.load(f)))

    for other_eval_prefix in other_eval_prefixes:
        for json_path in Path(checkpoint_dir).glob(f"{other_eval_prefix}*"):
            with open(json_path, "r") as f:
                result_list.append(ModelEvaluation(**json.load(f)))
    if len(result_list) == 0:
        raise ValueError(f"Failed to find any matching files.")

    all_dms_ids = set()
    for result in result_list:
        for campaign_result in result.campaign_results:
            all_dms_ids.add(campaign_result.dms_id)
    print(f"Loading results for DMSs: {all_dms_ids}")

    composite_result = None
    for single_result in result_list:
        single_result_dms_list = [cr.dms_id for cr in single_result.campaign_results]
        single_result_has_all_dms = all(dms_id in single_result_dms_list for dms_id in all_dms_ids)
        if not single_result_has_all_dms:
            print(f"Skipping {single_result.name} because it is missing some DMS IDs.")
            continue

        if composite_result is None:
            composite_result = single_result
            continue

        # Insert all campaign results into the appropriate list.
        for single_campaign_result in single_result.campaign_results:
            for composite_campaign_result in composite_result.campaign_results:
                if single_campaign_result.dms_id == composite_campaign_result.dms_id:
                    composite_campaign_result.config_results.extend(
                        single_campaign_result.config_results
                    )
                    break
            else:
                raise ValueError(
                    f"This should not have happened. We started with one of the results that had all DMSs. But one of the following results had a new DMS id: {single_campaign_result.dms_id}"
                )

    if not composite_result:
        raise ValueError(f"Failed to find any matching files.")

    return composite_result


class NaturalnessImputer(object):
    def __init__(self):
        self.is_pretrained = False
        self.single_mutant_naturalness_df: Optional[pd.DataFrame] = None
        self.knns: Dict[str, KNeighborsRegressor] = {}

    def pretrain(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series):
        if self.is_pretrained:
            raise ValueError("Model is already pretrained.")

        self.knns = {}
        for naturalness_column in naturalness_df.columns:
            naturalness_series = naturalness_df[naturalness_column]
            assert naturalness_series.index.equals(embedding_series.index)
            assert embedding_series.index.is_unique, "embedding_series contains duplicate indices"

            assert not naturalness_series.isna().any(), "naturalness_series contains NANs"

            X = np.array([np.array(emb) for emb in embedding_series.values])
            y = naturalness_series.values

            # Fit KNN regressor
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X, y)
            self.knns[naturalness_column] = knn

        # Also store the naturalness series for prediction later.
        self.single_mutant_naturalness_df = naturalness_df
        self.is_pretrained = True

    def impute(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        assert self.single_mutant_naturalness_df is not None
        assert len(naturalness_df.columns) == len(self.knns)
        assert set(naturalness_df.columns) == set(self.single_mutant_naturalness_df.columns)

        # Get the ensemble of naturalness scores with missing values imputed.

        ensemble_of_computed_naturalness: List[pd.Series] = []

        for naturalness_column in naturalness_df.columns:

            naturalness_series = naturalness_df[naturalness_column]
            single_mutant_naturalness_series = self.single_mutant_naturalness_df[naturalness_column]
            knn = self.knns[naturalness_column]

            def get_naturalness(seq_id, direct_naturalness) -> float:
                """Try computing naturalness for mutants even if none was provided by extrapolating for multimutants."""
                if direct_naturalness is not None and not pd.isna(direct_naturalness):
                    return direct_naturalness

                if is_homolog_seq_id(seq_id):
                    return np.nan

                # Break it down into single mutants.
                seq_id_parts = seq_id.split("_")

                # For multimutants, we compute naturalness as the product of the naturalness of the single mutants.
                computed_naturalness = single_mutant_naturalness_series.loc[seq_id_parts].sum()
                if pd.isna(computed_naturalness):
                    raise ValueError(
                        f"Computed naturalness is NAN for {seq_id} with parts {seq_id_parts}"
                    )
                return computed_naturalness

            naturalness_series.index.name = "seq_id"
            computed_naturalness_series = naturalness_series.reset_index(name="wt_marginal").apply(
                lambda r: get_naturalness(r.seq_id, r.wt_marginal), axis=1
            )
            computed_naturalness_series.index = naturalness_series.index

            # Do KNN imputation to fill in NANs from homologs.
            if computed_naturalness_series.isna().any():
                if not self.is_pretrained:
                    raise ValueError(
                        "Model is not pretrained, so cannot fill in NANs from homologs."
                    )

                logging.info(
                    f"Filling in NANs from homologs for {computed_naturalness_series.isna().sum()}/{len(computed_naturalness_series)} naturalness values."
                )
                assert embedding_series is not None
                embedding_array = np.array([np.array(emb) for emb in embedding_series.values])
                naturalness_array = computed_naturalness_series.values

                # Find indices for known and missing
                missing_mask = computed_naturalness_series.isna().to_numpy()
                X_missing = embedding_array[missing_mask]

                imputed_values = knn.predict(X_missing)

                # Fill in the missing values
                imputed_naturalness = naturalness_array.copy()
                imputed_naturalness[missing_mask] = imputed_values

                # Convert back to Series
                computed_naturalness_series = pd.Series(
                    imputed_naturalness, index=naturalness_series.index
                )

            if computed_naturalness_series.isna().any():
                raise ValueError(
                    f"Computed naturalness series still has NANs: {computed_naturalness_series.isna().sum()}/{len(computed_naturalness_series)}"
                )
            ensemble_of_computed_naturalness.append(computed_naturalness_series)
        return ensemble_of_computed_naturalness
