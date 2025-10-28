"""
Tests for the testing utilities.

This module contains tests for the test_utils module to ensure that
the testing utilities work correctly.
"""

import numpy as np
import pandas as pd
import pytest

from folde.tests.test_utils import (
    MockFewShotModel,
    MockZeroShotModel,
    create_simulated_protein_dataset,
)


def test_create_simulated_protein_dataset():
    """Test the simulated dataset creation."""
    num_samples = 50
    embedding_dim = 8
    random_seed = 42

    # Create the dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(
        num_samples=num_samples, embedding_dim=embedding_dim, random_seed=random_seed
    )

    # Verify the datasets
    assert len(activity_df) == num_samples
    assert len(naturalness_df) == num_samples
    assert len(embedding_df) == num_samples

    # Verify the structure of each dataframe
    assert "seq_id" in activity_df.columns
    assert "DMS_score" in activity_df.columns
    assert "mutant" in activity_df.columns
    assert activity_df.index.name == "seq_id"

    assert "seq_id" in naturalness_df.columns
    assert "wt_marginal" in naturalness_df.columns
    assert naturalness_df.index.name == "seq_id"

    assert "seq_id" in embedding_df.columns
    assert "embedding" in embedding_df.columns
    assert embedding_df.index.name == "seq_id"

    # Check that embeddings have the correct dimension
    assert len(embedding_df.embedding.iloc[0]) == embedding_dim

    # Check that random seed works by creating datasets with different seeds
    activity_df2, _, _ = create_simulated_protein_dataset(num_samples=num_samples, random_seed=43)

    # The datasets should be different with different seeds
    assert not np.array_equal(activity_df.DMS_score.values, activity_df2.DMS_score.values)  # type: ignore[reportArgumentType]

    # The datasets should be the same with the same seed
    activity_df3, _, _ = create_simulated_protein_dataset(num_samples=num_samples, random_seed=42)
    assert np.array_equal(activity_df.DMS_score.values, activity_df3.DMS_score.values)  # type: ignore[reportArgumentType]


def test_mock_zero_shot_model():
    """Test the mock zero-shot model."""
    # Create a simulated dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(random_seed=42)

    # Create a mock model
    model = MockZeroShotModel(temperature=0.1)

    # Extract series from dataframes
    naturalness_series = naturalness_df.wt_marginal
    embedding_series = embedding_df.embedding if "embedding" in embedding_df.columns else None

    # Test prediction
    predictions = model.predict(naturalness_series, embedding_series)
    assert len(predictions) == len(naturalness_series)
    assert np.array_equal(predictions, naturalness_series.values)  # type: ignore[reportArgumentType]

    # Test get_top_n
    top_n = 5
    top_seq_ids, pred_series = model.get_top_n(top_n, naturalness_series, embedding_series)
    assert len(top_seq_ids) == top_n
    assert len(pred_series) == len(naturalness_series)

    # With temperature=0, top_n should be deterministic
    model.temperature = 0.0
    top_seq_ids_det, _ = model.get_top_n(top_n, naturalness_series, embedding_series)

    # Should get the highest wt_marginal values
    top_by_naturalness = naturalness_df.sort_values("wt_marginal", ascending=False)
    assert set(top_seq_ids_det) == set(top_by_naturalness.head(top_n).index)


def test_mock_few_shot_model():
    """Test the mock few-shot model."""
    # Create a simulated dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(random_seed=42)

    # Create a mock model
    # TODO(jacob): Fix MockFewShotModel constructor in test refactor - missing required wt_aa_seq parameter
    model = MockFewShotModel()  # type: ignore[reportCallIssue]

    # Extract the series for activity, naturalness and embedding
    activity_series = activity_df.DMS_score
    naturalness_series = naturalness_df.wt_marginal
    embedding_series = embedding_df.embedding

    # Test fit (split data for train/test)
    half_size = len(naturalness_series) // 2
    train_naturalness = naturalness_series.iloc[:half_size]
    train_embedding = embedding_series.iloc[:half_size]
    train_activity = activity_series.iloc[:half_size]
    test_naturalness = naturalness_series.iloc[half_size:]
    test_embedding = embedding_series.iloc[half_size:]

    model.fit(train_naturalness, train_embedding, train_activity, test_naturalness, test_embedding)

    # Verify fit was called - we're using as_any() to tell the type checker
    # that we know what we're doing when accessing implementation-specific attributes
    from typing import Any, cast

    mock_model = cast(Any, model)
    assert mock_model.fit_called
    assert len(mock_model.fit_inputs) == 1

    # Test prediction
    predictions = model.predict(test_naturalness, test_embedding)
    assert len(predictions) == 1  # Should return a list with one series
    assert len(predictions[0]) == len(test_embedding)

    # Test get_top_n
    top_n = 5
    top_seq_ids, pred_series = model.get_top_n(top_n, naturalness_series, embedding_series)
    assert len(top_seq_ids) == top_n
    assert isinstance(pred_series, list)

    # Verify debug info
    debug_info = model.get_debug_info()
    assert debug_info["model_type"] == "MockFewShotModel"
