"""
Implementation of zero-shot protein prediction models.

This module contains models that can predict protein properties
without training on labeled data. These models are useful for
low-N protein engineering campaigns where little training data
is available.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from app.helpers.sequence_util import is_homolog_seq_id
from folde.util import constant_liar_sample, get_consensus_scores, internal_sample_n_indices
from sklearn.neighbors import KNeighborsRegressor

# Registry of available zero-shot models
_ZERO_SHOT_MODELS = {}


# def register_zeroshot_model(name):
#     """Decorator to register a zero-shot model class."""

#     def decorator(cls):
#         _ZERO_SHOT_MODELS[name] = cls
#         return cls

#     return decorator


class ZeroShotModel(ABC):
    """Abstract base class for zero-shot prediction models.

    Zero-shot models can predict protein properties without
    requiring training on labeled fitness/activity data.
    """

    def __init__(
        self,
        decision_mode: str = "mean",
        lie_noise_stddev_multiplier: float = 0.25,
        ucb_beta: float = 0.0,
        temperature: float = 0.0,
        epsilon: float = 0.0,
    ):
        """Initialize the zero-shot model.

        Args:
            **kwargs: Model-specific parameters
        """
        self.decision_mode = decision_mode
        self.lie_noise_stddev_multiplier = lie_noise_stddev_multiplier
        self.ucb_beta = ucb_beta
        self.temperature = temperature
        self.epsilon = epsilon

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "ZeroShotModel":
        """Optional method to pretrain the model on naturalness and embedding data."""
        return self

    @abstractmethod
    def predict(
        self, naturalness_df: pd.DataFrame, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Make predictions for protein variants.

        Args:
            naturalness_df: DataFrame containing possibly multiple columns of naturalness scores
            embedding_series: Optional Series containing protein embeddings

        Returns:
            Array of prediction scores for each variant
        """
        pass

    def get_top_n(
        self,
        n: int,
        naturalness_df: pd.DataFrame,
        embedding_series: Optional[pd.Series] = None,
    ) -> Tuple[List[str], List[pd.Series]]:
        """Get the top N variants predicted by the model.

        This method predicts scores for all variants and returns the
        sequence IDs of the top N variants by predicted score.

        Args:
            n: Number of top variants to return
            naturalness_df: DataFrame containing naturalness scores
            embedding_df: Optional DataFrame containing protein embeddings

        Returns:
            Tuple of
              * List of sequence IDs for the top N variants.
              * Series of predictions for all input variants with seq_id as index.
        """

        assert embedding_series is not None
        assert naturalness_df.index.equals(embedding_series.index)

        # Get predictions
        ensemble_of_predictions = self.predict(naturalness_df, embedding_series)

        self.selection_debug_info = {"sorts": {}}

        if self.decision_mode == "constantliar" or self.decision_mode == "krigingbeliever":
            assert self.lie_noise_stddev_multiplier is not None

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

            logging.info(
                f"Choosing {n} sequences ZERO SHOT with lie_noise_stddev_multiplier {self.lie_noise_stddev_multiplier} and ucb_beta {self.ucb_beta}"
            )
            chosen_seq_ids = constant_liar_sample(
                cl_considerations.to_numpy(),
                cl_considerations.index.to_numpy(),
                n,
                lie_noise_stddev_multiplier=self.lie_noise_stddev_multiplier,
                choice_of_baseline="min" if self.decision_mode == "constantliar" else "mean",
                ucb_beta=self.ucb_beta if self.ucb_beta is not None else 0.0,
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

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters and other debug info
        """
        # Default implementation without model_params
        return {"model_type": self.__class__.__name__}


def register_zeroshot_model(model_class: Type[ZeroShotModel]) -> Type[ZeroShotModel]:
    _ZERO_SHOT_MODELS[model_class.__name__] = model_class
    return model_class


@register_zeroshot_model
class RandomZeroShotModel(ZeroShotModel):
    """Zero-shot prediction model based on sequence naturalness scores.

    This model uses sequence naturalness (log-likelihood) scores
    directly as the prediction, optionally with some transformation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(
        self, naturalness_df: pd.DataFrame, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Predict using naturalness scores.

        Args:
            naturalness_df: DataFrame containing 'wt_marginal' column
            embedding_df: Not used by this model, but included for API consistency

        Returns:
            Array of prediction scores based on naturalness
        """
        return [pd.Series(np.random.rand(naturalness_df.shape[0]), index=naturalness_df.index)]

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters
        """
        return {}


@register_zeroshot_model
class NaturalnessZeroShotModel(ZeroShotModel):
    """Zero-shot prediction model based on sequence naturalness scores.

    This model uses sequence naturalness (log-likelihood) scores
    directly as the prediction, optionally with some transformation.
    """

    def __init__(self, **kwargs):
        """Initialize the naturalness-based model.

        Args:
            transformation: Transformation to apply to naturalness scores.
                Options: "identity", "log", "exp", "neg"
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.is_pretrained = False
        self.single_mutant_naturalness_df: Optional[pd.DataFrame] = None
        self.knns: Dict[str, KNeighborsRegressor] = {}

    def pretrain(
        self,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
    ) -> "NaturalnessZeroShotModel":
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

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters
        """
        info = super().get_debug_info()
        return info


def get_zero_shot_model(model_name: str, **kwargs) -> ZeroShotModel:
    """Get a zero-shot model instance by name.

    Args:
        model_name: Name of the zero-shot model to instantiate
        **kwargs: Parameters to pass to the model constructor

    Returns:
        Instantiated zero-shot model

    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in _ZERO_SHOT_MODELS:
        available_models = list(_ZERO_SHOT_MODELS.keys())
        raise ValueError(
            f"Unknown zero-shot model: {model_name}. Available models: {available_models}"
        )

    model_class = _ZERO_SHOT_MODELS[model_name]
    return model_class(**kwargs)
