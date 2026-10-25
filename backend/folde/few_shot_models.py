"""
Machine learning models for protein engineering prediction tasks.

This module provides scikit-learn compatible models for predicting
protein properties from embeddings. Models are thin wrappers around
scikit-learn implementations with additional protein-specific functionality.
"""

import logging
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

from app.helpers.preference_ranking import (
    BradleyTerryMLP,
    PreferenceTrainer,
    create_preference_model,
)
from app.helpers.sequence_util import sort_seq_id_list
from folde.data import OneHotVector
from folde.util import (
    NaturalnessImputer,
    cluster_sort_seq_ids,
    constant_liar_sample,
    get_consensus_scores,
    internal_sample_n_indices,
)


def build_single_mutant_naturalness_lookup(series: pd.Series) -> Dict[str, float]:
    """Map single-mutant seq_id -> naturalness, for additive imputation of multi-mutants."""
    is_single = ~pd.Index(series.index.astype(str)).str.contains("_")
    singles = series[is_single & ~series.isna()]
    return {str(k): float(v) for k, v in singles.items()}


def additively_impute_naturalness(
    series: pd.Series,
    single_allele_scores: Dict[str, float],
    context: str,
    column: str,
) -> pd.Series:
    """Fill NaN naturalness as the sum over constituent single mutants.

    Mirrors NaturalnessImputer.impute(): a multi-mutant's score is the sum of its singles'
    log wt-marginals (equivalently the product of their probabilities). We do NOT median-fill
    -- that was a real bug in the one-hot path, because a median-filled value looks like a
    real number and silently ties every multi-mutant. A missing constituent raises rather
    than being guessed at.
    """
    missing_mask = series.isna()
    n_missing = int(missing_mask.sum())
    logging.info(
        f"{context}: naturalness column '{column}' arrived with "
        f"{n_missing}/{len(series)} NaN ({100.0 * n_missing / max(len(series), 1):.4f}%)"
    )
    if n_missing == 0:
        return series

    filled = series.copy()
    for seq_id in series.index[missing_mask]:
        total = 0.0
        for part in str(seq_id).split("_"):
            if part not in single_allele_scores:
                raise ValueError(
                    f"{context}: cannot impute naturalness for '{seq_id}' -- constituent "
                    f"single mutant '{part}' has no naturalness in column '{column}'. "
                    "Refusing to guess a value for a model input."
                )
            total += single_allele_scores[part]
        filled.loc[seq_id] = total
    return filled


# Registry of available models
_FEW_SHOT_MODELS = {}


class FewShotModel(ABC):
    """Abstract base class for few-shot protein property prediction models."""

    def __init__(
        self,
        wt_aa_seq: str,
        lie_noise_stddev_multiplier: float | None = None,
        lie_noise_stddev_multiplier_schedule: list[float] | None = None,
        decision_mode: str = "median",
        ucb_beta: float | None = None,
        temperature: float = 0.0,
        epsilon: float = 0.0,
    ):
        self.wt_aa_seq = wt_aa_seq
        self.lie_noise_stddev_multiplier = lie_noise_stddev_multiplier
        self.lie_noise_stddev_multiplier_schedule = lie_noise_stddev_multiplier_schedule
        self.decision_mode = decision_mode
        self.ucb_beta = ucb_beta
        self.temperature = temperature
        self.epsilon = epsilon
        self.selection_debug_info: dict[str, Any] = {}

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "FewShotModel":
        """Optional method to pretrain the model on naturalness and embedding data."""
        return self

    @abstractmethod
    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
    ) -> "FewShotModel":
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        """Make predictions using the trained model."""
        pass

    def get_top_n(
        self,
        n: int,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        round_number: int | None = None,
    ) -> Tuple[List[str], List[pd.Series]]:
        """Get the top N variants predicted by the model.

        This method combines features from naturalness and embedding dataframes,
        makes predictions, and returns the sequence IDs of the top N variants.

        Args:
            n: Number of top variants to return
            naturalness_df: DataFrame of naturalness scores indexed by seq_id, one column per ensemble member
            embedding_series: Series of embeddings indexed by seq_id
            round_number: Number of the round (zero-based)

        Returns:
            Tuple of
              * List of sequence IDs for the top N variants.
              * List of Series of predictions for all input variants with seq_id as index.

        Raises:
            ValueError: If the model is not fitted or if required columns are missing
        """
        assert naturalness_df.index.equals(embedding_series.index)

        # Convert list of embeddings to numpy array
        ensemble_of_predictions = self.predict(naturalness_df, embedding_series)

        self.selection_debug_info = {"sorts": {}}

        if self.decision_mode == "constantliar" or self.decision_mode == "krigingbeliever":
            if self.lie_noise_stddev_multiplier is not None:
                assert self.lie_noise_stddev_multiplier_schedule is None
                lie_noise_stddev_multiplier = self.lie_noise_stddev_multiplier

            elif self.lie_noise_stddev_multiplier_schedule is not None:
                assert self.lie_noise_stddev_multiplier is None
                assert round_number is not None
                lie_noise_stddev_multiplier = self.lie_noise_stddev_multiplier_schedule[
                    round_number
                ]
                logging.info(
                    f"Using lie_noise_stddev_multiplier {lie_noise_stddev_multiplier} for this round and {self.lie_noise_stddev_multiplier_schedule} remaining."
                )

            else:
                raise ValueError(
                    "Either lie_noise_stddev_multiplier or lie_noise_stddev_multiplier_schedule must be set."
                )

            ensemble_scores = get_consensus_scores(ensemble_of_predictions, "mean")
            pred_df = {
                f"model_{ii}": ensemble_of_predictions[ii]
                for ii in range(len(ensemble_of_predictions))
            }
            pred_df["score"] = ensemble_scores
            pred_df = pd.DataFrame(pred_df, index=ensemble_of_predictions[0].index).sort_values(
                "score", ascending=False
            )
            cl_considerations = pred_df.drop(columns=["score"]).iloc[:5000]
            chosen_seq_ids = constant_liar_sample(
                cl_considerations.to_numpy(),
                cl_considerations.index.to_numpy(),
                n,
                lie_noise_stddev_multiplier=lie_noise_stddev_multiplier,
                choice_of_baseline="min" if self.decision_mode == "constantliar" else "mean",
                ucb_beta=self.ucb_beta if self.ucb_beta is not None else 0.0,
            )

            self.selection_debug_info["sorts"]["selection_order"] = chosen_seq_ids
            try:
                self.selection_debug_info["sorts"]["cluster_order"] = cluster_sort_seq_ids(
                    pred_df.loc[pd.Index(chosen_seq_ids)]
                )
            except Exception as e:
                logging.error(f"Error clustering seq_ids: {e}")

            self.selection_debug_info["sorts"]["seq_id_order"] = sort_seq_id_list(
                self.wt_aa_seq, chosen_seq_ids
            )
        else:
            ensemble_scores = get_consensus_scores(ensemble_of_predictions, self.decision_mode)
            chosen_indices = internal_sample_n_indices(
                ensemble_scores.to_numpy(),
                n,
                temperature=self.temperature,
                epsilon=self.epsilon,
            )
            chosen_seq_ids = naturalness_df.index[chosen_indices].tolist()
            self.selection_debug_info["sorts"]["selection_order"] = chosen_seq_ids

        return (
            chosen_seq_ids,
            ensemble_of_predictions,
        )

    @abstractmethod
    def get_debug_info(self) -> Dict[str, Any]:
        """Get model-specific debug information."""
        return self.selection_debug_info


def register_few_shot_model(model_class: Type[FewShotModel]) -> Type[FewShotModel]:
    _FEW_SHOT_MODELS[model_class.__name__] = model_class
    return model_class


# def register_few_shot_model(name: str):
#     """Register a model in the global registry."""

#     def decorator(cls):
#         _FEW_SHOT_MODELS[name] = cls
#         return cls

#     return decorator


@register_few_shot_model
class RandomFewShotModel(FewShotModel):
    """Just guess random activity."""

    def __init__(self, random_state: int, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
    ) -> "RandomFewShotModel":
        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        np.random.seed(self.random_state)
        return [pd.Series(np.random.rand(naturalness_df.shape[0]), index=naturalness_df.index)]

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


@register_few_shot_model
class NaturalnessFewShotModel(FewShotModel):
    """Use naturalness scores as activity scores."""

    def __init__(self, random_state: int, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state
        self.naturalness_imputer = NaturalnessImputer()

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
    ) -> "NaturalnessFewShotModel":
        return self

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "NaturalnessFewShotModel":
        self.naturalness_imputer.pretrain(naturalness_df, embedding_series)
        return self

    def predict(
        self, naturalness_df: pd.DataFrame, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Predict using naturalness scores.

        Args:
            naturalness_series: Series containing naturalness scores, some of which may be NAN.
            embedding_series: Optional Series containing protein embeddings

        Returns:
            Array of prediction scores based on naturalness
        """
        assert embedding_series is not None
        return self.naturalness_imputer.impute(naturalness_df, embedding_series)

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


# TODO(jacob): Implement GP Model
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html


@register_few_shot_model
class MLPFewShotModel(FewShotModel):
    """Multi-layer Perceptron regressor for protein property prediction.

    Thin wrapper around sklearn's MLPRegressor. See scikit-learn documentation
    for full parameter details: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    def __init__(self, ensemble_size: int = 1, **kwargs):
        """Initialize the MLP regressor with any parameters supported by sklearn's MLPRegressor."""
        super().__init__(
            wt_aa_seq=kwargs.pop("wt_aa_seq"),
            lie_noise_stddev_multiplier=kwargs.pop("lie_noise_stddev_multiplier", 4.0),
            decision_mode=kwargs.pop("decision_mode", "median"),
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )
        self.ensemble_size = ensemble_size
        self.base_random_state = kwargs.pop("random_state", 0)
        self.is_fitted: bool = False

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
        **kwargs,
    ) -> "MLPFewShotModel":
        """Train the MLP regressor.

        Args:
            naturalness_df: DataFrame of ALL mutants' naturalness scores indexed by seq_id, one column per ensemble member
            embedding_series: Series of ALL mutants' embeddings indexed by seq_id
            measured_activity_series: Series of measured activity measurements indexed by seq_id
            test_activity_series: Optional series of test activity measurements indexed by seq_id
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        assert naturalness_df.index.equals(embedding_series.index)

        # Reset state! We might have called "fit" already and should blank that out.
        self.models: List[SklearnMLPRegressor] = [
            SklearnMLPRegressor(**kwargs, random_state=self.base_random_state + ii)
            for ii in range(self.ensemble_size)
        ]
        self.metrics_: Dict[str, float] = {}

        measured_embedding_series = embedding_series.loc[measured_activity_series.index]
        X = np.array([np.array(emb) for emb in measured_embedding_series.values])
        y = measured_activity_series.to_numpy()
        # Train the model
        for model in self.models:
            model.fit(X, y, **kwargs)
        self.is_fitted = True

        # Calculate training metrics
        y_train_pred = get_consensus_scores(
            [pd.Series(m.predict(X), index=naturalness_df.index) for m in self.models], "median"
        )
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if test_activity_series is not None:
            assert False, "TODO: IMPLEMENT"
            # X_val, y_val = validation_data
            # y_val_pred = get_ensemble_prediction(self.models, X_val, "median")
            # self.metrics_.update(
            #     {
            #         "val_mse": mean_squared_error(y_val, y_val_pred),
            #         "val_r2": r2_score(y_val, y_val_pred),
            #         "val_mae": mean_absolute_error(y_val, y_val_pred),
            #     }
            # )

        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        """Make predictions using the trained MLP."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])

        return [pd.Series(m.predict(X), embedding_series.index) for m in self.models]

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the MLP."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Basic debug info
        debug_info: Dict[str, Any] = {
            "metrics": self.metrics_,
        }

        # Get properties from the first model in the ensemble
        if len(self.models) > 0:
            first_model = self.models[0]

            if hasattr(first_model, "n_layers_"):
                debug_info["n_layers"] = first_model.n_layers_

            if hasattr(first_model, "n_outputs_"):
                debug_info["n_outputs"] = first_model.n_outputs_

            if hasattr(first_model, "n_iter_"):
                debug_info["n_iter"] = first_model.n_iter_

            if hasattr(first_model, "loss_curve_"):
                debug_info["loss_curve"] = first_model.loss_curve_

            # Network structure
            if hasattr(first_model, "coefs_") and hasattr(first_model, "intercepts_"):
                network_structure = []
                for i, (coef, intercept) in enumerate(
                    zip(first_model.coefs_, first_model.intercepts_)
                ):
                    layer_info = {
                        "layer": i + 1,
                        "shape": coef.shape,
                        "n_params": coef.size + intercept.size,
                    }
                    network_structure.append(layer_info)
                debug_info["network_structure"] = network_structure

        return debug_info


@register_few_shot_model
class RandomForestFewShotModel(FewShotModel):
    """Random Forest regressor for protein property prediction.

    Thin wrapper around sklearn's RandomForestRegressor. See scikit-learn documentation
    for full parameter details: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    def __init__(self, ensemble_size: int = 1, **kwargs):
        """Initialize the Random Forest regressor with any parameters supported by sklearn's RandomForestRegressor."""
        super().__init__(
            wt_aa_seq=kwargs.pop("wt_aa_seq"),
            lie_noise_stddev_multiplier=kwargs.pop("lie_noise_stddev_multiplier", 4.0),
            decision_mode=kwargs.pop("decision_mode", "median"),
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )

        self.ensemble_size = ensemble_size
        self.base_random_state = kwargs.pop("random_state", 0)
        self.is_fitted: bool = False

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: pd.Series | None = None,
        **kwargs,
    ) -> "RandomForestFewShotModel":
        """Train the Random Forest regressor.

        Args:
            naturalness_df: DataFrame of ALL mutants' naturalness scores indexed by seq_id, one column per ensemble member
            embedding_series: Series of ALL mutants' embeddings indexed by seq_id
            measured_activity_series: Series of measured activity measurements indexed by seq_id
            test_activity_series: Optional series of test activity measurements indexed by seq_id
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        assert naturalness_df.index.equals(embedding_series.index)

        # Reset state! We might have called "fit" already and should blank that out.
        self.models: List[SklearnRandomForestRegressor] = [
            SklearnRandomForestRegressor(**kwargs, random_state=self.base_random_state + ii)
            for ii in range(self.ensemble_size)
        ]
        self.metrics_: Dict[str, float] = {}

        measured_embedding_series = embedding_series.loc[measured_activity_series.index]
        X = np.array([np.array(emb) for emb in measured_embedding_series.values])
        y = measured_activity_series.to_numpy()

        # Train the model
        for model in self.models:
            model.fit(X, y, **kwargs)
        self.is_fitted = True

        # Calculate training metrics
        y_train_pred = get_consensus_scores(
            [pd.Series(m.predict(X), index=measured_activity_series.index) for m in self.models],
            "median",
        )
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if test_activity_series is not None:
            # TODO(jbr): Maybe utilize the test series?
            pass

        # Add OOB score if available
        if hasattr(self.models, "oob_score_"):
            self.metrics_["oob_score"] = self.models[0].oob_score_

        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        """Make predictions using the trained Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])
        return [pd.Series(m.predict(X), index=embedding_series.index) for m in self.models]

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        debug_info: Dict[str, Any] = super().get_debug_info()
        debug_info["metrics"] = self.metrics_

        # Get properties from the first model in the ensemble
        if len(self.models) > 0:
            first_model = self.models[0]

            # Get feature importances if available
            if hasattr(first_model, "feature_importances_"):
                # 7/13/25: Had to can the feature importances, it was way too big for model evals.
                # feature_importances = {"mean": first_model.feature_importances_.tolist()}
                # # Add standard deviation if we have estimators
                # if hasattr(first_model, "estimators_") and len(first_model.estimators_) > 0:
                #     feature_importances["std"] = np.std(
                #         [tree.feature_importances_ for tree in first_model.estimators_],
                #         axis=0,
                #     ).tolist()
                # debug_info["feature_importances"] = feature_importances

                # Basic tree info (limit to first 5 trees to avoid excessive info)
                if hasattr(first_model, "estimators_"):
                    tree_info = []
                    for i, tree in enumerate(first_model.estimators_[:5]):
                        if hasattr(tree, "tree_"):
                            tree_info.append(
                                {
                                    "tree_idx": i,
                                    "n_nodes": tree.tree_.node_count,
                                    "max_depth": tree.tree_.max_depth,
                                }
                            )

                    if tree_info:
                        debug_info["tree_info"] = tree_info

                    debug_info["n_estimators"] = len(first_model.estimators_)

        return debug_info


@register_few_shot_model
class TorchMLPFewShotModel(FewShotModel):
    """Custom, torch-backed MLP model."""

    def __init__(
        self,
        wt_aa_seq: str,
        random_state: int,
        hidden_dims: list[int] = [100, 50],
        dropout: float = 0.1,
        device: str | None = None,
        ensemble_size: int = 1,
        enable_ensemble_devariancing: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrain: bool = False,
        pretrain_epochs: int = 10,
        pretrain_patience: int = 20,
        pretrain_val_frequency: int = 5,
        shrink_and_perturb_params: tuple[float, float] | None = None,
        train_epochs: int = 50,
        train_patience: int | None = None,
        val_frequency: int = 10,
        use_mse_loss: bool = False,
        do_holdout_validation: bool = False,
        do_validation_with_pair_fraction: float | None = None,
        importance_sampling_reweighting_strat: str | None = None,
        importance_sampling_temperature: float | None = None,
        use_exponential_learning_rate_decay: bool = False,
        use_plateau_learning_rate_decay: bool = False,
        embedding_dim: int | None = None,  # DEPRECATED
        **kwargs,
    ):
        super().__init__(wt_aa_seq, **kwargs)

        self.base_random_state = random_state

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.ensemble_size = ensemble_size
        self.enable_ensemble_devariancing = enable_ensemble_devariancing
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.should_pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_patience = pretrain_patience
        self.pretrain_val_frequency = pretrain_val_frequency
        self.shrink_and_perturb_params = shrink_and_perturb_params
        self.train_epochs = train_epochs
        self.train_patience = train_patience
        self.val_frequency = val_frequency
        self.use_mse_loss = use_mse_loss
        self.do_holdout_validation = do_holdout_validation
        self.do_validation_with_pair_fraction = do_validation_with_pair_fraction
        self.importance_sampling_reweighting_strat = importance_sampling_reweighting_strat
        self.importance_sampling_temperature = importance_sampling_temperature
        self.use_exponential_learning_rate_decay = use_exponential_learning_rate_decay
        self.use_plateau_learning_rate_decay = use_plateau_learning_rate_decay

        self.pretrained_model_state_dicts = []

        self.finetuned_model_and_trainer_list = []

        if embedding_dim is not None:
            logging.warning("embedding_dim is deprecated and ignored.")

        self.is_pretrained = False
        self.pretrain_metrics: list[dict[str, Any]] = []
        self.is_fitted = False
        self.finetune_metrics: list[dict[str, Any]] = []

    def _create_model_ensemble(
        self, embedding_dim: int
    ) -> list[tuple[BradleyTerryMLP, PreferenceTrainer]]:
        return [
            create_preference_model(
                embedding_dim=embedding_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                device=self.device,
                random_state=self.base_random_state + ii,
            )
            for ii in range(self.ensemble_size)
        ]

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "TorchMLPFewShotModel":
        assert naturalness_df.index.equals(embedding_series.index)
        assert embedding_series.index.is_unique, "embedding_series contains duplicate indices"

        if not self.should_pretrain:
            return self

        if self.is_pretrained:
            raise ValueError("Model is already pretrained.")

        embedding_dim = embedding_series.iloc[0].shape[0]

        # Create pretrained version of each model.
        for ii, (model, trainer) in enumerate(self._create_model_ensemble(embedding_dim)):
            # Choose the naturalness column for this model.
            naturalness_series = naturalness_df[
                naturalness_df.columns[ii % len(naturalness_df.columns)]
            ]
            has_naturalness_data = ~naturalness_series.isna()
            naturalness_series_with_data = naturalness_series[has_naturalness_data]
            embedding_series_with_data = embedding_series[has_naturalness_data]

            # if ii == 0:
            #     logging.info(
            #         f"Pretraining model with naturalness data with {sum(has_naturalness_data)} naturalness measurements embedding dimension {embedding_dim}."
            #     )

            rng = np.random.RandomState(self.base_random_state + ii)
            validation_seqids = rng.choice(
                embedding_series_with_data.index,
                size=int(embedding_series_with_data.shape[0] * 0.2),
                replace=False,
            )
            train_seqids = np.setdiff1d(embedding_series_with_data.index, validation_seqids)

            X_train = np.array(
                [np.array(emb) for emb in embedding_series_with_data[train_seqids].values]
            )
            X_val = np.array(
                [np.array(emb) for emb in embedding_series_with_data[validation_seqids].values]
            )
            y_train = naturalness_series_with_data[train_seqids].to_numpy()
            y_val = naturalness_series_with_data[validation_seqids].to_numpy()

            self.pretrain_metrics.append(
                trainer.train(
                    train_embeddings=X_train,
                    train_activity_labels=y_train,
                    val_embeddings=X_val,
                    val_activity_labels=y_val,
                    batch_size=256,  # Increased batch size 32->256, speeding up training quite a bit.
                    epochs=self.pretrain_epochs,
                    patience=self.pretrain_patience,
                    use_mse_loss=self.use_mse_loss,
                    val_frequency=self.pretrain_val_frequency,
                    learning_rate=1e-4 * 8,  # Increased LR to compensate for larger batches.
                    weight_decay=1e-5,
                    importance_sampling_reweighting_strat=None,
                    importance_sampling_temperature=None,
                    use_exponential_learning_rate_decay=False,
                    use_plateau_learning_rate_decay=False,
                )
            )
            self.pretrained_model_state_dicts.append(model.state_dict())

        self.is_pretrained = True

        train_loss_start = np.mean([m["train_loss"][0] for m in self.pretrain_metrics])
        train_loss_end = np.mean([m["train_loss"][-1] for m in self.pretrain_metrics])
        val_loss_start = np.mean([m["val_loss"][0] for m in self.pretrain_metrics])
        val_loss_end = np.mean([m["val_loss"][-1] for m in self.pretrain_metrics])
        logging.info(
            f"Pretrain loss: train {train_loss_start:.4f} -> {train_loss_end:.4f}, val {val_loss_start:.4f} -> {val_loss_end:.4f} after {len(self.pretrain_metrics[0]['train_loss']) * self.val_frequency} epochs"
        )

        return self

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: pd.Series | None = None,
    ) -> "TorchMLPFewShotModel":
        """Train the TorchMLPFewShotModel.

        Args:
            naturalness_df: DataFrame of ALL mutants' naturalness scores indexed by seq_id, one column per ensemble member
            embedding_series: Series of ALL mutants' embeddings indexed by seq_id
            measured_activity_series: Series of measured activity measurements indexed by seq_id
            test_activity_series: Optional series of test activity measurements indexed by seq_id
        """
        assert naturalness_df.index.equals(embedding_series.index)
        assert embedding_series.index.is_unique, "embedding_series contains duplicate indices"

        if test_activity_series is not None:
            assert test_embedding_series is not None
            assert test_naturalness_df is not None
            assert test_embedding_series.index.equals(test_naturalness_df.index)
            assert test_embedding_series.index.equals(test_activity_series.index)

        # Reset state! We might have called "fit" already and should blank that out.
        embedding_dim = embedding_series.iloc[0].shape[0]
        self.finetuned_model_and_trainer_list = self._create_model_ensemble(embedding_dim)
        self.finetune_metrics = []

        if self.should_pretrain:
            if not self.is_pretrained:
                raise ValueError("Model is not pretrained. Call pretrain() first.")
            if not len(self.pretrained_model_state_dicts) == self.ensemble_size:
                raise ValueError(
                    f"Number of pretrained models does not match ensemble size {len(self.pretrained_model_state_dicts)} != {self.ensemble_size}."
                )
            for idx, (model, trainer) in enumerate(self.finetuned_model_and_trainer_list):
                model.load_state_dict(self.pretrained_model_state_dicts[idx])

                if self.shrink_and_perturb_params:
                    beta, sigma = self.shrink_and_perturb_params
                    for p in model.parameters():
                        p.data.mul_(beta)  # shrink
                        p.data.add_(torch.randn_like(p) * sigma)  # perturb

        kf_splits = None
        if self.do_holdout_validation:
            kf = KFold(
                n_splits=self.ensemble_size, shuffle=True, random_state=self.base_random_state
            )
            kf_splits = list(kf.split(measured_activity_series.index))

        for model_idx, (model, trainer) in enumerate(self.finetuned_model_and_trainer_list):
            X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None

            if self.do_holdout_validation:
                assert (
                    self.do_validation_with_pair_fraction is None
                ), f"Cannot do both holdout validation and specify a pair fraction to hold out."
                assert kf_splits is not None, "kf_splits is not set"
                # Do the train / test split.
                train_indices, val_indices = kf_splits[model_idx]
                train_seqids = measured_activity_series.index[train_indices]
                val_seqids = measured_activity_series.index[val_indices]

                X_train = np.array([np.array(emb) for emb in embedding_series[train_seqids].values])
                y_train = measured_activity_series[train_seqids].to_numpy()
                X_val = np.array([np.array(emb) for emb in embedding_series[val_seqids].values])
                y_val = measured_activity_series[val_seqids].to_numpy()
            else:
                X_train = np.array(
                    [
                        np.array(emb)
                        for emb in embedding_series[measured_activity_series.index].values
                    ]
                )
                y_train = measured_activity_series.to_numpy()

            if (
                test_activity_series is not None
                and test_embedding_series is not None
                and test_naturalness_df is not None
            ):
                X_test = np.array(
                    [
                        np.array(emb)
                        for emb in test_embedding_series[test_activity_series.index].values
                    ]
                )
                y_test = test_activity_series.to_numpy()

            self.finetune_metrics.append(
                trainer.train(
                    train_embeddings=X_train,
                    train_activity_labels=y_train,
                    val_embeddings=X_val,
                    val_activity_labels=y_val,
                    test_embeddings=X_test,
                    test_activity_labels=y_test,
                    batch_size=max(16, y_train.shape[0]),
                    epochs=self.train_epochs,
                    patience=self.train_patience,
                    use_mse_loss=self.use_mse_loss,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    val_frequency=self.val_frequency,
                    do_validation_with_pair_fraction=self.do_validation_with_pair_fraction,
                    importance_sampling_reweighting_strat=self.importance_sampling_reweighting_strat,
                    importance_sampling_temperature=self.importance_sampling_temperature,
                    use_exponential_learning_rate_decay=self.use_exponential_learning_rate_decay,
                    use_plateau_learning_rate_decay=self.use_plateau_learning_rate_decay,
                )
            )
            if (
                "val_loss" in self.finetune_metrics[-1]
                and "train_loss" in self.finetune_metrics[-1]
            ):
                train_loss_list = self.finetune_metrics[-1]["train_loss"]
                val_loss_list = [
                    v for v in self.finetune_metrics[-1]["val_loss"] if not np.isnan(v)
                ]
                if len(train_loss_list) > 0 and len(val_loss_list) > 0:
                    logging.info(
                        f"Finetune improvement: train loss ({train_loss_list[0]:.4f} -> {train_loss_list[-1]:.4f}) val loss ({val_loss_list[0]:.4f} -> {val_loss_list[-1]:.4f})"
                    )

        self.is_fitted = True

        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        """Make predictions using the trained Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])
        pred_series_list = []
        for _, t in self.finetuned_model_and_trainer_list:
            score_array = t.predict_scores(X)
            score_array = score_array - score_array.mean(axis=0)  #  / score_array.std(axis=0)
            if self.enable_ensemble_devariancing:
                score_array = score_array / score_array.std(axis=0)
            pred_series_list.append(pd.Series(score_array, index=embedding_series.index))
        return pred_series_list

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        debug_info = super().get_debug_info()
        debug_info["pretrain_metrics"] = self.pretrain_metrics
        debug_info["pretrain_val_frequency"] = self.pretrain_val_frequency
        debug_info["finetune_metrics"] = self.finetune_metrics
        debug_info["finetune_val_frequency"] = self.val_frequency
        return debug_info


@register_few_shot_model
class ALDEFewShotModel(FewShotModel):
    """Reimplementation of the ALDE surrogate of Yang et al. (2025), as a FolDE baseline.

    Reference:
      Yang, Lal, Bowden, Astudillo, Hameedi, Kaur, Hill, Yue & Arnold,
      "Active learning-assisted directed evolution", Nature Communications 16, 714 (2025).
      Code: https://github.com/jsunn-y/ALDE

    This reproduces the ``DNN_ENSEMBLE`` surrogate with the ``GREEDY`` / ``UCB`` / ``TS``
    acquisition functions from ``src/`` of that repository. Their defaults, which we follow:

      * 5 feed-forward nets, each trained on an independent random 90% subsample of the
        measured data (a subsample, NOT a bootstrap resample).
      * Architecture ``[d, 30, 30, 1]`` with LeakyReLU, Adam at lr 1e-3, MSE loss,
        full-batch, up to 300 iterations with a training-loss early stop over a 30-iteration
        window.
      * Targets normalized by the running maximum of the observed labels, recomputed each
        round.
      * Inputs min-max normalized to [0, 1] using the per-dimension range of the candidate
        pool.
      * Batch selection is top-B on a single acquisition surface computed once per round --
        ALDE performs no joint batch acquisition and no fantasizing. Their code notes that
        batch expected improvement "ran extremely slowly without noticeable improvement in
        performance".

    Deviations from the reference implementation, all deliberate:

      * Their pool min-max normalization produces NaN for any feature column that is
        constant across the pool. That never fires on a combinatorially complete library
        (every one-hot column contains both 0 and 1) but fires on nearly every column of a
        ProteinGym pool of singles and doubles. We leave constant columns at zero instead.
      * Their target normalization divides by ``max(y)``, which is degenerate when the
        running max is non-positive -- possible with ProteinGym DMS scores, impossible with
        the non-negative fitnesses they benchmark on. We divide by ``max(|y|)``, which is
        identical whenever ``max(y) > 0``.

    This model ignores naturalness entirely, matching the base ALDE repository, where round 1
    is random rather than zero-shot. Pair it with ``RandomZeroShotModel`` for a faithful
    comparison, or with ``NaturalnessZeroShotModel`` for the head-to-head that gives ALDE the
    same round-1 advantage FolDE has (compare their ``alde4ssmula`` "focused training" option).
    """

    ACQUISITIONS = ("greedy", "ucb", "ts")

    def __init__(
        self,
        ensemble_size: int = 5,
        hidden_dims: Optional[List[int]] = None,
        bootstrap_size: float = 0.9,
        train_iter: int = 300,
        learning_rate: float = 1e-3,
        early_stop_window: int = 30,
        negative_slope: float = 0.01,
        acquisition: str = "greedy",
        ucb_xi: float = 4.0,
        **kwargs,
    ):
        super().__init__(
            wt_aa_seq=kwargs.pop("wt_aa_seq"),
            decision_mode=kwargs.pop("decision_mode", "mean"),
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )
        if acquisition not in self.ACQUISITIONS:
            raise ValueError(
                f"Unknown ALDE acquisition {acquisition!r}. Options: {self.ACQUISITIONS}"
            )
        self.ensemble_size = ensemble_size
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [30, 30]
        self.bootstrap_size = bootstrap_size
        self.train_iter = train_iter
        self.learning_rate = learning_rate
        self.early_stop_window = early_stop_window
        self.negative_slope = negative_slope
        self.acquisition = acquisition
        self.ucb_xi = ucb_xi
        self.base_random_state = kwargs.pop("random_state", 0)

        self.models: List[torch.nn.Module] = []
        self.domain_min_: NDArray[np.float64] | None = None
        self.domain_scale_: NDArray[np.float64] | None = None
        self.metrics_: Dict[str, Any] = {}
        self.is_fitted: bool = False

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _stack_embeddings(embedding_series: pd.Series) -> NDArray[np.float32]:
        """Stack embeddings into one float32 array, preallocated to avoid a 2x peak.

        float32 rather than ALDE's float64: one-hot encodings over a full DMS pool are far
        wider than the fixed-site libraries they benchmark on (tens of thousands of columns
        for a large protein), and the pool array is the dominant memory cost. The values are
        exactly representable either way.
        """
        values = embedding_series.values
        n = len(values)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        first = np.asarray(values[0], dtype=np.float32)
        out = np.empty((n, first.shape[0]), dtype=np.float32)
        out[0] = first
        for i in range(1, n):
            out[i] = np.asarray(values[i], dtype=np.float32)
        return out

    def _set_domain_from_pool(self, embedding_series: pd.Series) -> None:
        """Record the pool's per-dimension min/max without materializing the pool.

        For sparse one-hot variants (``OneHotVector``) the min/max follow from how many rows
        set each column, which we count from the indices directly -- so the domain costs one
        pass over small index arrays instead of a dense pool-sized array. Dense embeddings
        fall back to the straightforward computation.
        """
        values = embedding_series.values
        if len(values) == 0:
            raise ValueError("Cannot fit on an empty candidate pool")

        first = values[0]
        if isinstance(first, OneHotVector):
            dim = first.dim
            counts = np.zeros(dim, dtype=np.int64)
            for v in values:
                counts[v.indices] += 1
            n = len(values)
            lo = (counts == n).astype(np.float32)  # column set in every row -> min 1
            hi = (counts > 0).astype(np.float32)  # column set in any row -> max 1
        else:
            X_pool = self._stack_embeddings(embedding_series)
            lo = X_pool.min(axis=0)
            hi = X_pool.max(axis=0)

        span = hi - lo
        # Constant columns would give NaN in the reference implementation. Hold them at 0.
        span[span <= 0] = 1.0
        self.domain_min_ = lo.astype(np.float32)
        self.domain_scale_ = span.astype(np.float32)

    def _normalize(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.domain_min_ is not None and self.domain_scale_ is not None
        X -= self.domain_min_
        X /= self.domain_scale_
        return X

    def _build_net(self, input_dim: int, generator: torch.Generator) -> torch.nn.Module:
        layers: List[torch.nn.Module] = []
        prev = input_dim
        for width in self.hidden_dims:
            linear = torch.nn.Linear(prev, width)
            with torch.no_grad():
                # Re-draw the default init under our seeded generator so ensemble members
                # differ reproducibly.
                bound = 1.0 / np.sqrt(prev)
                linear.weight.uniform_(-bound, bound, generator=generator)
                linear.bias.uniform_(-bound, bound, generator=generator)
            layers.append(linear)
            layers.append(torch.nn.LeakyReLU(negative_slope=self.negative_slope))
            prev = width
        head = torch.nn.Linear(prev, 1)
        with torch.no_grad():
            bound = 1.0 / np.sqrt(prev)
            head.weight.uniform_(-bound, bound, generator=generator)
            head.bias.uniform_(-bound, bound, generator=generator)
        layers.append(head)
        return torch.nn.Sequential(*layers)

    # -- FewShotModel interface ------------------------------------------------

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
    ) -> "ALDEFewShotModel":
        assert naturalness_df.index.equals(embedding_series.index)

        # ``embedding_series`` here is the whole candidate pool, which is what defines the
        # normalization domain in ALDE.
        self._set_domain_from_pool(embedding_series)

        measured_embeddings = embedding_series.loc[measured_activity_series.index]
        X = self._normalize(self._stack_embeddings(measured_embeddings))
        y_raw = measured_activity_series.to_numpy(dtype=np.float64)

        # ALDE's running-max target normalization (see class docstring for the guard).
        denom = float(np.max(np.abs(y_raw))) if y_raw.size else 1.0
        if denom < 1e-12:
            denom = 1.0
        self.y_denominator_ = denom
        y = y_raw / denom

        n_samples, input_dim = X.shape
        n_subsample = max(2, int(round(self.bootstrap_size * n_samples)))

        X_t = torch.from_numpy(X).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32).unsqueeze(1)

        self.models = []
        losses: List[float] = []
        iters: List[int] = []
        for member in range(self.ensemble_size):
            seed = self.base_random_state + member
            generator = torch.Generator().manual_seed(seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_samples, size=min(n_subsample, n_samples), replace=False)

            net = self._build_net(input_dim, generator)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            loss_fn = torch.nn.MSELoss()

            Xi, yi = X_t[idx], y_t[idx]
            best_loss = float("inf")
            best_iter = 0
            last_loss = float("inf")
            for it in range(self.train_iter):
                optimizer.zero_grad()
                loss = loss_fn(net(Xi), yi)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
                if last_loss < best_loss - 1e-6:
                    best_loss = last_loss
                    best_iter = it
                elif it - best_iter >= self.early_stop_window:
                    break

            net.eval()
            self.models.append(net)
            losses.append(last_loss)
            iters.append(it + 1)

        self.is_fitted = True
        self.metrics_ = {
            "n_train": int(n_samples),
            "input_dim": int(input_dim),
            "final_train_mse_per_member": losses,
            "iterations_per_member": iters,
            "y_denominator": denom,
        }
        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        """Score the pool, in chunks.

        ALDE chunks pool inference too (``predict_batched_gpu``, 1000 rows at a time). Here it
        also keeps peak memory bounded: a one-hot pool for a large protein can be tens of
        thousands of columns wide, and materializing a second normalized copy of it was enough
        to exhaust memory on the largest benchmark datasets.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        n = len(embedding_series)
        chunk_size = 2048
        outputs = [np.empty(n, dtype=np.float64) for _ in self.models]
        values = embedding_series.values
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                block = np.empty(
                    (stop - start, len(self.domain_min_)),  # type: ignore[arg-type]
                    dtype=np.float32,
                )
                for offset, i in enumerate(range(start, stop)):
                    block[offset] = np.asarray(values[i], dtype=np.float32)
                block = self._normalize(block)
                block_t = torch.from_numpy(block)
                for model_idx, net in enumerate(self.models):
                    outputs[model_idx][start:stop] = net(block_t).squeeze(-1).numpy()

        return [pd.Series(o, index=embedding_series.index) for o in outputs]

    def get_top_n(
        self,
        n: int,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        round_number: int | None = None,
    ) -> Tuple[List[str], List[pd.Series]]:
        """Select a batch the way ALDE does: top-B on one acquisition surface.

        For ``greedy`` and ``ucb`` the surface is computed once and the top B distinct
        candidates are taken. For ``ts`` a fresh ensemble member is drawn per batch element,
        which -- as in the reference implementation -- means a batch of B is drawn from only
        ``ensemble_size`` distinct rankings.
        """
        assert naturalness_df.index.equals(embedding_series.index)
        ensemble_of_predictions = self.predict(naturalness_df, embedding_series)
        pred_arr = np.stack([p.to_numpy() for p in ensemble_of_predictions])
        index = ensemble_of_predictions[0].index

        self.selection_debug_info = {"sorts": {}, "acquisition": self.acquisition}

        if self.acquisition == "ts":
            rng = np.random.default_rng(self.base_random_state + (round_number or 0))
            chosen_positions: List[int] = []
            for _ in range(min(n, len(index))):
                member = int(rng.integers(0, pred_arr.shape[0]))
                surface = pred_arr[member].copy()
                if chosen_positions:
                    surface[np.asarray(chosen_positions)] = -np.inf
                chosen_positions.append(int(np.argmax(surface)))
        else:
            if self.acquisition == "ucb":
                surface = pred_arr.mean(axis=0) + np.sqrt(self.ucb_xi) * pred_arr.std(axis=0)
            else:
                surface = pred_arr.mean(axis=0)
            chosen_positions = np.argsort(surface)[::-1][:n].tolist()

        chosen_seq_ids = index[chosen_positions].tolist()
        self.selection_debug_info["sorts"]["selection_order"] = chosen_seq_ids
        try:
            self.selection_debug_info["sorts"]["seq_id_order"] = sort_seq_id_list(
                self.wt_aa_seq, chosen_seq_ids
            )
        except Exception as e:  # pragma: no cover - debug info only
            logging.error(f"Error sorting seq_ids: {e}")

        return chosen_seq_ids, ensemble_of_predictions

    def get_debug_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        debug_info = dict(self.selection_debug_info)
        debug_info["metrics"] = self.metrics_
        return debug_info


def is_valid_few_shot_model_name(model_name: str) -> bool:
    return model_name in _FEW_SHOT_MODELS


def get_few_shot_model(model_name: str, **kwargs) -> FewShotModel:
    """Get a few-shot model by name with the provided parameters.

    Args:
        model_name: Name of the model to create ('mlp' or 'random_forest')
        **kwargs: Parameters passed directly to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in _FEW_SHOT_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(_FEW_SHOT_MODELS.keys())}"
        )

    model_class = _FEW_SHOT_MODELS[model_name]
    return model_class(**kwargs)


# ##########################################################################################
# NATURALNESS-AS-INPUT ALTERNATIVES TO WARM-STARTING ########################################
#
# Reviewer 1 comment 6 asks whether warm-starting the MLP on naturalness is the best way to
# exploit the prior. These two classes are the alternatives screened against it:
#   * NaturalnessFeatureMLPFewShotModel   -- naturalness as an explicit input feature
#   * NaturalnessResidualMLPFewShotModel  -- naturalness as a prior the MLP corrects
# Both are ALTERNATIVES to warm-starting, so both require pretrain=False.
#
# Both work by appending the z-scored naturalness columns to each embedding vector and then
# delegating to TorchMLPFewShotModel.fit/predict unchanged. The per-ensemble-member choice of
# naturalness column happens inside the torch module (via nat_offset) rather than by splitting
# the training loop, so the parent's fit() -- holdout validation, pair-fraction validation,
# test-set plumbing, importance sampling, LR schedules -- is preserved exactly as written.


class _NaturalnessFeatureBTMLP(BradleyTerryMLP):
    """Bradley-Terry MLP whose input is [embedding, naturalness_z_for_this_member].

    Input carries every naturalness column; this member reads only column ``nat_offset``, so
    one augmented embedding series serves the whole ensemble.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        dropout: float,
        nat_offset: int,
    ):
        # The net sees the embedding plus exactly one naturalness scalar.
        super().__init__(embedding_dim=embedding_dim + 1, hidden_dims=hidden_dims, dropout=dropout)
        self.raw_embedding_dim = embedding_dim
        self.nat_offset = nat_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.raw_embedding_dim
        nat = x[:, dim + self.nat_offset : dim + self.nat_offset + 1]
        return self.mlp(torch.cat([x[:, :dim], nat], dim=1))


class _NaturalnessResidualBTMLP(BradleyTerryMLP):
    """Bradley-Terry MLP computing f(x) = naturalness_z(x) + g(x).

    ``g`` is the ordinary MLP over the embedding, with its output layer zero-initialized, so
    at initialization f equals the naturalness prior exactly and training only learns the
    correction. The prior enters as an additive offset rather than an input feature, so
    gradients flow through g alone.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        dropout: float,
        nat_offset: int,
    ):
        # g sees only the embedding; naturalness is added to g's output, not fed into it.
        super().__init__(embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.raw_embedding_dim = embedding_dim
        self.nat_offset = nat_offset

        output_layer = self.mlp[-1]
        assert isinstance(output_layer, torch.nn.Linear), (
            "expected BradleyTerryMLP to end in nn.Linear so the residual head can be "
            f"zero-initialized, got {type(output_layer)}"
        )
        assert (
            output_layer.out_features == 1
        ), f"expected a scalar output layer, got out_features={output_layer.out_features}"
        # f == prior exactly at initialization.
        torch.nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            torch.nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.raw_embedding_dim
        nat = x[:, dim + self.nat_offset : dim + self.nat_offset + 1]
        return self.mlp(x[:, :dim]) + nat


class _NaturalnessInputMLPFewShotModel(TorchMLPFewShotModel):
    """Shared plumbing for the two naturalness-as-input models.

    Holds the parts that are delicate enough that duplicating them across both subclasses
    would be the larger risk: per-member column selection, additive NaN imputation, and
    train/inference-consistent standardization.
    """

    def __init__(
        self,
        wt_aa_seq: str,
        random_state: int,
        constant_naturalness_for_validation: bool = False,
        **kwargs,
    ):
        super().__init__(wt_aa_seq=wt_aa_seq, random_state=random_state, **kwargs)
        if self.should_pretrain:
            raise ValueError(
                f"{type(self).__name__} requires pretrain=False: it is an alternative to "
                "warm-starting, not an addition to it. Pretraining would also size the "
                "ensemble against the un-augmented embedding dimension."
            )
        # Plumbing check only: forces the naturalness feature to a constant, under which the
        # feature-concat model should reproduce hotstart-0epochs.
        self.constant_naturalness_for_validation = constant_naturalness_for_validation

        self._naturalness_columns: List[str] = []
        self._naturalness_mean: Dict[str, float] = {}
        self._naturalness_std: Dict[str, float] = {}
        self._single_mutant_naturalness: Dict[str, Dict[str, float]] = {}
        # Captured in pretrain() from the ENTIRE pool's single mutants; kept separate so each
        # fit can union it with the per-fit frame rather than be overwritten by it.
        self._pretrain_single_naturalness: Dict[str, Dict[str, float]] = {}
        self._naturalness_state_is_set = False

    # -- naturalness preparation ---------------------------------------------------------

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "_NaturalnessInputMLPFewShotModel":
        """Capture the complete single-mutant naturalness table.

        pretrain() receives the single mutants of the ENTIRE pool, whereas fit() only sees the
        bootstrapped world -- half of it -- so the constituent singles of a world multi-mutant
        are frequently absent there and additive imputation cannot resolve them. This never
        fired on the 9-dataset training benchmark (all single-mutation, so imputation never
        runs) but breaks every multi-mutation dataset. NaturalnessImputer gets its table the
        same way, via pretrain().

        Does NOT warm-start anything: these models run with pretrain=False by construction, and
        the superclass call below returns immediately. Standardization statistics are
        deliberately left to fit(), computed over the candidate pool it is given.
        """
        for column in naturalness_df.columns:
            self._pretrain_single_naturalness[str(column)] = build_single_mutant_naturalness_lookup(
                naturalness_df[column]
            )
        logging.info(
            f"{type(self).__name__}: pretrain captured single-mutant naturalness for "
            f"{len(self._pretrain_single_naturalness)} column(s), "
            f"{sum(len(v) for v in self._pretrain_single_naturalness.values())} values"
        )
        return cast(
            "_NaturalnessInputMLPFewShotModel", super().pretrain(naturalness_df, embedding_series)
        )

    def _impute_additively(self, series: pd.Series, column: str) -> pd.Series:
        """Fill NaN naturalness as the sum over constituent single mutants.

        Mirrors NaturalnessImputer.impute(): a multi-mutant's score is the sum of its
        singles' log wt-marginals (equivalently the product of their probabilities). We do
        NOT median-fill -- that was a real bug in the one-hot path, because a median-filled
        value looks like a real number and silently ties every multi-mutant.
        """
        return additively_impute_naturalness(
            series,
            self._single_mutant_naturalness[column],
            context=type(self).__name__,
            column=column,
        )

    def _set_naturalness_state(self, naturalness_df: pd.DataFrame) -> None:
        """Compute standardization stats ONCE, over the full candidate pool.

        Recomputing these per call would standardize train and inference against different
        index subsets, desynchronizing them. No labels are involved, so using the whole pool
        leaks nothing.
        """
        self._naturalness_columns = list(naturalness_df.columns)
        self._single_mutant_naturalness = {}
        self._naturalness_mean = {}
        self._naturalness_std = {}

        for column in self._naturalness_columns:
            series = naturalness_df[column]
            # Single mutants carry direct scores and are the basis for additive imputation.
            # Union with pretrain's entire-pool table: this frame is only the bootstrapped
            # world, so on a multi-mutation dataset it lacks many constituent singles.
            lookup = build_single_mutant_naturalness_lookup(series)
            lookup.update(self._pretrain_single_naturalness.get(column, {}))
            self._single_mutant_naturalness[column] = lookup

            imputed = self._impute_additively(series, column)
            assert (
                not imputed.isna().any()
            ), f"{type(self).__name__}: {int(imputed.isna().sum())} NaNs remain in '{column}'"

            mean = float(imputed.mean())
            std = float(imputed.std())
            if not np.isfinite(std) or std == 0.0:
                # Happens under constant_naturalness_for_validation; z becomes identically 0.
                logging.info(
                    f"{type(self).__name__}: naturalness column '{column}' has zero variance "
                    "over the pool; using std=1.0 so the feature is a constant zero"
                )
                std = 1.0
            self._naturalness_mean[column] = mean
            self._naturalness_std[column] = std

        self._naturalness_state_is_set = True

    def _naturalness_matrix(self, naturalness_df: pd.DataFrame) -> NDArray[np.float32]:
        """Z-scored, imputed naturalness aligned to naturalness_df.index, one column each."""
        if not self._naturalness_state_is_set:
            raise ValueError("Naturalness state is not set. Call fit() first.")
        if list(naturalness_df.columns) != self._naturalness_columns:
            raise ValueError(
                f"Naturalness columns changed since fit(): expected "
                f"{self._naturalness_columns}, got {list(naturalness_df.columns)}"
            )
        columns = []
        for column in self._naturalness_columns:
            imputed = self._impute_additively(naturalness_df[column], column)
            assert not imputed.isna().any(), f"NaNs remain in '{column}' at inference"
            z = (
                imputed.to_numpy(dtype=np.float64) - self._naturalness_mean[column]
            ) / self._naturalness_std[column]
            columns.append(z)
        return np.asarray(np.stack(columns, axis=1), dtype=np.float32)

    def _augment(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> pd.Series:
        """Append the naturalness columns to every embedding vector."""
        assert naturalness_df.index.equals(embedding_series.index)
        nat = self._naturalness_matrix(naturalness_df)
        if self.constant_naturalness_for_validation:
            # Zero AFTER standardization, so the appended column is exactly 0 rather than the
            # constant (0 - mean)/std. The output layers have no bias, so an all-zero feature
            # contributes nothing at all -- which is what makes this a sharp plumbing check.
            nat = np.zeros_like(nat)
        vectors = [
            np.concatenate([np.asarray(emb, dtype=np.float32), nat[row_idx]])
            for row_idx, emb in enumerate(embedding_series.values)
        ]
        return pd.Series(vectors, index=embedding_series.index)

    # -- ensemble construction -----------------------------------------------------------

    def _make_module(self, raw_embedding_dim: int, nat_offset: int) -> BradleyTerryMLP:
        raise NotImplementedError

    def _create_model_ensemble(
        self, embedding_dim: int
    ) -> list[tuple[BradleyTerryMLP, PreferenceTrainer]]:
        """Build the ensemble over augmented inputs.

        ``embedding_dim`` arrives as raw_dim + n_naturalness_columns, because fit() measures
        it from the augmented series.
        """
        n_columns = len(self._naturalness_columns)
        assert n_columns > 0, "Naturalness state is not set; call fit() first."
        raw_embedding_dim = embedding_dim - n_columns

        ensemble = []
        for ii in range(self.ensemble_size):
            random_state = self.base_random_state + ii
            # Mirror create_preference_model's seeding so initialization matches the base model.
            torch.manual_seed(random_state)
            random.seed(random_state)
            np.random.seed(random_state)
            # Same rule pretrain() uses to pick a naturalness column per ensemble member.
            nat_offset = ii % n_columns
            model = self._make_module(raw_embedding_dim, nat_offset)
            trainer = PreferenceTrainer(model=model, random_state=random_state, device=self.device)
            ensemble.append((model, trainer))
        return ensemble

    # -- overrides -----------------------------------------------------------------------

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: pd.Series | None = None,
    ) -> "_NaturalnessInputMLPFewShotModel":
        # naturalness_df here is the whole candidate world, which is what we standardize over.
        self._set_naturalness_state(naturalness_df)

        augmented = self._augment(naturalness_df, embedding_series)
        augmented_test = None
        if test_embedding_series is not None and test_naturalness_df is not None:
            augmented_test = self._augment(test_naturalness_df, test_embedding_series)

        super().fit(
            naturalness_df,
            augmented,
            measured_activity_series,
            test_naturalness_df,
            augmented_test,
            test_activity_series,
        )
        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        return super().predict(naturalness_df, self._augment(naturalness_df, embedding_series))


@register_few_shot_model
class NaturalnessFeatureMLPFewShotModel(_NaturalnessInputMLPFewShotModel):
    """Naturalness appended to the embedding as one explicit input feature."""

    def _make_module(self, raw_embedding_dim: int, nat_offset: int) -> BradleyTerryMLP:
        return _NaturalnessFeatureBTMLP(
            embedding_dim=raw_embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            nat_offset=nat_offset,
        )


@register_few_shot_model
class NaturalnessResidualMLPFewShotModel(_NaturalnessInputMLPFewShotModel):
    """Naturalness as an additive prior the MLP learns a correction to."""

    def _make_module(self, raw_embedding_dim: int, nat_offset: int) -> BradleyTerryMLP:
        return _NaturalnessResidualBTMLP(
            embedding_dim=raw_embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            nat_offset=nat_offset,
        )


# ##########################################################################################
# AUGMENTED RIDGE BASELINES (Hsu et al. 2022) ##############################################
#
# Reviewer 2 comment 1 asks for baselines built on top of the zero-shot predictions: a linear
# model instead of a NN, and a model combining zero-shot predictions with measurements but
# without ESM representations. These are the "augmented" regression baselines of
#
#   Hsu, Nisonoff, Fannjiang & Listgarten, "Learning protein fitness models from evolutionary
#   and assay-limited data", Nature Biotechnology 40, 1114-1122 (2022)
#
# i.e. ridge regression on sequence features with the density-model (zero-shot) score included
# as one additional feature. Their evaluation at 48-240 labeled examples brackets our 3-round
# budget of 48 exactly.
#
# Ridge is deterministic, so there is no ensemble and no constant liar: these arms use plain
# top-N selection and therefore differ from FolDE in the ACQUISITION as well as the surrogate.


@register_few_shot_model
class RidgeFewShotModel(FewShotModel):
    """Ridge regression on sequence features, optionally augmented with the zero-shot score.

    Args:
        augment: append the naturalness (zero-shot) score as one additional feature, which is
            the augmentation of Hsu et al. (2022).
        two_step: instead of one joint fit, regress activity on naturalness alone (OLS) and
            ridge the residuals on the remaining features. Reported alongside the joint fit
            when the joint fit's naturalness coefficient comes out negligible.
        alphas: ridge penalty grid searched by leave-one-out.
        chunk_size: rows per batch when featurizing the pool. Ridge is linear so batching
            cannot shift predictions relative to each other -- unlike the torch models, whose
            predict() centers scores by the mean of whatever batch it is handed.
    """

    def __init__(
        self,
        wt_aa_seq: str,
        random_state: int,
        augment: bool = False,
        two_step: bool = False,
        alphas: Optional[List[float]] = None,
        chunk_size: int = 8192,
        **kwargs,
    ):
        super().__init__(wt_aa_seq, **kwargs)
        self.random_state = random_state
        self.augment = augment
        self.two_step = two_step
        self.alphas: List[float] = (
            alphas if alphas is not None else list(np.logspace(-3.0, 6.0, 19))
        )
        self.chunk_size = chunk_size

        self.is_fitted = False
        self._feature_mean: Optional[NDArray[np.float64]] = None
        self._feature_std: Optional[NDArray[np.float64]] = None
        self._pool_signature: Optional[Tuple[int, int]] = None
        self._single_mutant_naturalness: Dict[str, float] = {}
        self._naturalness_column: Optional[str] = None
        self._model: Any = None
        self._naturalness_only_model: Any = None
        self._fit_info: Dict[str, Any] = {}

    # -- featurization -------------------------------------------------------------------

    def _naturalness_series(self, naturalness_df: pd.DataFrame) -> pd.Series:
        """Imputed naturalness for this frame, as a Series aligned to its index."""
        assert self._naturalness_column is not None
        return additively_impute_naturalness(
            naturalness_df[self._naturalness_column],
            self._single_mutant_naturalness,
            context=type(self).__name__,
            column=self._naturalness_column,
        )

    def _raw_features(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        rows: pd.Index,
    ) -> NDArray[np.float64]:
        """Unstandardized feature block for `rows`: embedding, plus naturalness if augmenting."""
        embeddings = np.asarray(
            [np.asarray(emb, dtype=np.float64) for emb in embedding_series.loc[rows].values],
            dtype=np.float64,
        )
        if not self.augment:
            return embeddings
        nat = self._naturalness_series(naturalness_df.loc[rows]).to_numpy(dtype=np.float64)
        return np.concatenate([embeddings, nat.reshape(-1, 1)], axis=1)

    def _standardize(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._feature_mean is not None and self._feature_std is not None
        return (raw - self._feature_mean) / self._feature_std

    def _compute_pool_statistics(
        self, naturalness_df: pd.DataFrame, embedding_series: pd.Series
    ) -> None:
        """Mean/std per feature over the FULL candidate pool, computed once and stored.

        Streamed in chunks rather than materializing the pool matrix: SPG1's pool is 268,481
        variants, which would be several GB densified. No labels are involved, so using the
        whole pool is not leakage. Recomputing these per call would standardize train and
        inference against different index subsets and desynchronize them.
        """
        index = embedding_series.index
        # fit() is called afresh every round, but the candidate pool is the same object each
        # time, so the statistics are identical. Recomputing them means another full densifying
        # pass over the pool per round -- material on the multi-mutation pools. Reuse them when
        # the pool is provably unchanged (length plus a content hash, not a heuristic).
        signature = (len(index), int(pd.util.hash_pandas_object(index, index=False).sum()))
        if self._pool_signature == signature and self._feature_mean is not None:
            return
        total = 0
        feature_sum: Optional[NDArray[np.float64]] = None
        feature_sq_sum: Optional[NDArray[np.float64]] = None
        for start in range(0, len(index), self.chunk_size):
            rows = index[start : start + self.chunk_size]
            block = self._raw_features(naturalness_df, embedding_series, rows)
            if feature_sum is None:
                feature_sum = np.zeros(block.shape[1], dtype=np.float64)
                feature_sq_sum = np.zeros(block.shape[1], dtype=np.float64)
            assert feature_sq_sum is not None
            feature_sum += block.sum(axis=0)
            feature_sq_sum += (block**2).sum(axis=0)
            total += block.shape[0]
        assert feature_sum is not None and feature_sq_sum is not None and total > 0

        mean = feature_sum / total
        variance = np.maximum(feature_sq_sum / total - mean**2, 0.0)
        std = np.sqrt(variance)
        # A pool-constant column carries no information; leave it at zero rather than dividing
        # by zero. One-hot pools have many such columns.
        n_constant = int((std == 0.0).sum())
        std = np.where(std == 0.0, 1.0, std)
        self._feature_mean = mean
        self._feature_std = std
        self._pool_signature = signature
        self._fit_info["n_features"] = int(mean.shape[0])
        self._fit_info["n_pool_constant_features"] = n_constant
        self._fit_info["pool_size_for_standardization"] = total

    # -- FewShotModel interface ----------------------------------------------------------

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "RidgeFewShotModel":
        """Capture the complete single-mutant naturalness table.

        pretrain() receives the single mutants of the ENTIRE pool, whereas fit() only sees the
        bootstrapped world -- half the pool -- so the constituent singles of a world
        multi-mutant are frequently absent there. Building the additive-imputation table from
        fit() alone therefore fails on multi-mutation datasets. NaturalnessImputer gets its
        table the same way, via pretrain().
        """
        if len(naturalness_df.columns) == 0:
            return self
        self._naturalness_column = str(naturalness_df.columns[0])
        self._single_mutant_naturalness = build_single_mutant_naturalness_lookup(
            naturalness_df[self._naturalness_column]
        )
        logging.info(
            f"{type(self).__name__}: pretrain captured "
            f"{len(self._single_mutant_naturalness)} single-mutant naturalness values"
        )
        return self

    def fit(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        test_naturalness_df: pd.DataFrame | None = None,
        test_embedding_series: pd.Series | None = None,
        test_activity_series: Optional[pd.Series] = None,
    ) -> "RidgeFewShotModel":
        assert naturalness_df.index.equals(embedding_series.index)
        assert embedding_series.index.is_unique, "embedding_series contains duplicate indices"

        # Naturalness bookkeeping, needed whenever naturalness enters the features.
        self._naturalness_column = str(naturalness_df.columns[0])
        if self.augment or self.two_step:
            # Union with whatever pretrain() captured. pretrain() holds the entire pool's
            # singles; this frame is only the bootstrapped world, so it can add nothing the
            # pretrain table lacks -- but merging keeps the model correct if pretrain was
            # never called, and the values agree wherever both have an entry.
            merged = build_single_mutant_naturalness_lookup(
                naturalness_df[self._naturalness_column]
            )
            merged.update(self._single_mutant_naturalness)
            self._single_mutant_naturalness = merged

        self._compute_pool_statistics(naturalness_df, embedding_series)

        rows = pd.Index(measured_activity_series.index)
        X = self._standardize(self._raw_features(naturalness_df, embedding_series, rows))
        # Raw activity: every measurement within a dataset comes from one assay.
        y = measured_activity_series.to_numpy(dtype=np.float64)
        assert not np.isnan(X).any(), f"{int(np.isnan(X).sum())} NaNs in X before fitting"
        assert not np.isnan(y).any(), "NaNs in y before fitting"

        # Leave-one-out is RidgeCV's default and is honest at n=16 without a holdout split.
        self._model = SklearnRidgeCV(alphas=self.alphas)
        if self.two_step:
            # OLS on naturalness alone, then ridge the residuals on the remaining features.
            nat_column = X[:, -1].reshape(-1, 1) if self.augment else None
            if nat_column is None:
                nat_raw = self._naturalness_series(naturalness_df.loc[rows]).to_numpy(
                    dtype=np.float64
                )
                nat_column = nat_raw.reshape(-1, 1)
            self._naturalness_only_model = SklearnLinearRegression().fit(nat_column, y)
            residual = y - self._naturalness_only_model.predict(nat_column)
            features = X[:, :-1] if self.augment else X
            self._model.fit(features, residual)
        else:
            self._model.fit(X, y)

        self._fit_info["n_train"] = int(len(y))
        self._fit_info["alpha"] = float(getattr(self._model, "alpha_", float("nan")))
        self._fit_info["alpha_at_grid_floor"] = bool(
            np.isclose(self._fit_info["alpha"], min(self.alphas))
        )
        self._fit_info["alpha_at_grid_ceiling"] = bool(
            np.isclose(self._fit_info["alpha"], max(self.alphas))
        )
        # The naturalness coefficient and the embedding-coefficient scale are logged on EVERY
        # augmented fit, two-step included. Without this the two-step arm could not be checked
        # against the validation gate (is the penalty still reaching naturalness?), which is
        # exactly how the first augmented run shipped invalid.
        if self.two_step:
            # Unpenalized OLS slope on the (standardized) naturalness column, then the ridge
            # coefficients fitted to the residual on the embedding block.
            ridge_coefficients = np.asarray(self._model.coef_, dtype=np.float64)
            self._fit_info["naturalness_coef"] = float(
                np.asarray(self._naturalness_only_model.coef_, dtype=np.float64).ravel()[0]
            )
            self._fit_info["mean_abs_embedding_coef"] = float(np.abs(ridge_coefficients).mean())
            self._fit_info["max_abs_embedding_coef"] = float(np.abs(ridge_coefficients).max())
        elif self.augment:
            coefficients = np.asarray(self._model.coef_, dtype=np.float64)
            self._fit_info["naturalness_coef"] = float(coefficients[-1])
            self._fit_info["mean_abs_embedding_coef"] = float(np.abs(coefficients[:-1]).mean())
            self._fit_info["max_abs_embedding_coef"] = float(np.abs(coefficients[:-1]).max())
        if "naturalness_coef" in self._fit_info:
            self._fit_info["nat_to_mean_emb_ratio"] = float(
                abs(self._fit_info["naturalness_coef"])
                / max(self._fit_info["mean_abs_embedding_coef"], 1e-300)
            )
        self.is_fitted = True
        return self

    def predict(self, naturalness_df: pd.DataFrame, embedding_series: pd.Series) -> List[pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        assert naturalness_df.index.equals(embedding_series.index)

        index = embedding_series.index
        scores = np.empty(len(index), dtype=np.float64)
        for start in range(0, len(index), self.chunk_size):
            rows = index[start : start + self.chunk_size]
            X = self._standardize(self._raw_features(naturalness_df, embedding_series, rows))
            if self.two_step:
                nat_column = (
                    X[:, -1].reshape(-1, 1)
                    if self.augment
                    else self._naturalness_series(naturalness_df.loc[rows])
                    .to_numpy(dtype=np.float64)
                    .reshape(-1, 1)
                )
                features = X[:, :-1] if self.augment else X
                block = self._naturalness_only_model.predict(nat_column) + self._model.predict(
                    features
                )
            else:
                block = self._model.predict(X)
            scores[start : start + len(rows)] = block
        # Deterministic, so a single-member "ensemble".
        return [pd.Series(scores, index=index)]

    def get_debug_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        debug_info = super().get_debug_info()
        debug_info["ridge_fit_info"] = dict(self._fit_info)
        debug_info["augment"] = self.augment
        debug_info["two_step"] = self.two_step
        return debug_info
