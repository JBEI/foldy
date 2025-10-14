"""
Machine learning models for protein engineering prediction tasks.

This module provides scikit-learn compatible models for predicting
protein properties from embeddings. Models are thin wrappers around
scikit-learn implementations with additional protein-specific functionality.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import torch
from app.helpers.preference_ranking import (
    BradleyTerryMLP,
    PreferenceTrainer,
    create_preference_model,
)
from app.helpers.sequence_util import sort_seq_id_list
from folde.util import (
    cluster_sort_seq_ids,
    constant_liar_sample,
    get_consensus_scores,
    internal_sample_n_indices,
)
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

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
