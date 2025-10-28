"""
Tests for the campaign simulation functionality.

This module tests the protein engineering campaign simulation logic,
including:
- CampaignWorldState class
- _run_single_simulation function
- _evaluate_metrics function
- simulate_campaign function
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from folde.campaign import (
    CampaignWorldState,
    _evaluate_metrics,
    _run_single_simulation,
    simulate_campaign,
)
from folde.few_shot_models import FewShotModel
from folde.types import FolDEModelConfig, ModelEvaluation, SimulationResult
from folde.zero_shot_models import ZeroShotModel


class TestDatasetGeneration:
    """Tests for dataset generation utilities."""

    def test_create_simulated_dataset(self):
        """Test creating a simulated dataset for testing campaign simulations."""
        # Create simulated dataset
        num_samples = 100
        embedding_dim = 10

        # Create sequence IDs
        seq_ids = [f"seq_{i}" for i in range(num_samples)]

        # Create activity data
        activity_values = np.random.normal(0, 1, num_samples)
        activity_df = pd.DataFrame(
            {
                "seq_id": seq_ids,
                "DMS_score": activity_values,
                "mutant": [f"A{i}G" for i in range(num_samples)],
            }
        )
        activity_df = activity_df.set_index("seq_id", drop=False)

        # Create naturalness data
        naturalness_values = np.random.normal(-10, 2, num_samples)
        naturalness_df = pd.DataFrame({"seq_id": seq_ids, "wt_marginal": naturalness_values})
        naturalness_df = naturalness_df.set_index("seq_id", drop=False)

        # Create embedding data
        embeddings = [np.random.normal(0, 1, embedding_dim) for _ in range(num_samples)]
        embedding_df = pd.DataFrame({"seq_id": seq_ids, "embedding": embeddings})
        embedding_df = embedding_df.set_index("seq_id", drop=False)

        # Verify the datasets
        assert len(activity_df) == num_samples
        assert len(naturalness_df) == num_samples
        assert len(embedding_df) == num_samples
        assert activity_df.index.name == "seq_id"
        assert naturalness_df.index.name == "seq_id"
        assert embedding_df.index.name == "seq_id"

        # This is a helper function other tests will use
        self.activity_df = activity_df
        self.naturalness_df = naturalness_df
        self.embedding_df = embedding_df


class MockZeroShotModel(ZeroShotModel):
    """Mock ZeroShotModel for testing campaign simulations."""

    def __init__(self, temperature=0.0, return_values=None):
        """Initialize mock model.

        Args:
            temperature: Temperature parameter for Boltzmann sampling
            return_values: Optional fixed values to return for predictions
        """
        super().__init__(temperature=temperature)
        self.return_values = return_values
        self.predict_called = False
        self.predict_inputs = []

    def predict(self, naturalness_series, embedding_series=None):
        """Mock predict method."""
        self.predict_called = True
        self.predict_inputs.append((naturalness_series, embedding_series))

        if self.return_values is not None:
            return self.return_values

        # By default, return a list with a single series
        return [pd.Series(naturalness_series.values, index=naturalness_series.index)]


class MockFewShotModel(FewShotModel):
    """Mock FewShotModel for testing campaign simulations."""

    def __init__(
        self,
        return_values=None,
        decision_mode="median",
        temperature=0.0,
        epsilon=0.0,
        random_state=42,
    ):
        """Initialize mock model.

        Args:
            return_values: Optional fixed values to return for predictions
            decision_mode: How to combine ensemble predictions
            temperature: Temperature for sampling
            epsilon: Epsilon for epsilon-greedy exploration
        """
        super().__init__(decision_mode=decision_mode, temperature=temperature, epsilon=epsilon)
        self.return_values = return_values
        self.random_state = random_state
        self.fit_called = False
        self.fit_inputs = []
        self.predict_called = False
        self.predict_inputs = []

    def fit(
        self,
        naturalness_series,
        embedding_series,
        measured_activity_series,
        validation_activity_series=None,
        **kwargs,
    ):
        """Mock fit method."""
        self.fit_called = True
        self.fit_inputs.append((naturalness_series, embedding_series, measured_activity_series))
        return self

    def predict(self, naturalness_series, embedding_series):
        """Mock predict method."""
        self.predict_called = True
        self.predict_inputs.append((naturalness_series, embedding_series))

        if self.return_values is not None:
            if isinstance(self.return_values, list):
                return self.return_values
            else:
                # Convert to a list with a single Series
                return [pd.Series(self.return_values, index=embedding_series.index)]

        # Return a list with a single Series of random values
        np.random.seed(self.random_state)
        return [pd.Series(np.random.rand(len(embedding_series)), index=embedding_series.index)]

    def get_debug_info(self):
        """Mock debug info method."""
        return {"model_type": "MockFewShotModel"}


class TestCampaignWorldState:
    """Tests for the CampaignWorldState class."""

    def test_initialization(self):
        """Test initializing the CampaignWorldState."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Initialize world state with series
        world_state = CampaignWorldState(
            dataset_generator.activity_df["DMS_score"],  # Converting to Series
            dataset_generator.naturalness_df["wt_marginal"],  # Converting to Series
            dataset_generator.embedding_df["embedding"],  # Converting to Series
        )

        # Verify initialization
        assert world_state.golden_activity_series.equals(dataset_generator.activity_df["DMS_score"])
        assert world_state.naturalness_series.equals(
            dataset_generator.naturalness_df["wt_marginal"]
        )
        assert world_state.embedding_series.equals(dataset_generator.embedding_df["embedding"])
        assert world_state.measured_seq_ids == []

    def test_measure_variant_activities(self):
        """Test measuring variant activities."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Initialize world state
        world_state = CampaignWorldState(
            dataset_generator.activity_df,
            dataset_generator.naturalness_df,
            dataset_generator.embedding_df,
        )

        # Measure some variants
        seq_ids_to_measure = ["seq_0", "seq_5", "seq_10"]
        world_state.measure_variant_activities(seq_ids_to_measure)

        # Verify measured variants
        assert world_state.measured_seq_ids == seq_ids_to_measure

        # Measure more variants
        more_seq_ids = ["seq_15", "seq_20"]
        world_state.measure_variant_activities(more_seq_ids)

        # Verify all measured variants
        assert world_state.measured_seq_ids == seq_ids_to_measure + more_seq_ids

    def test_get_unmeasured_variants(self):
        """Test getting unmeasured variants."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Get the series
        activity_series = dataset_generator.activity_df["DMS_score"]
        naturalness_series = dataset_generator.naturalness_df["wt_marginal"]
        embedding_series = dataset_generator.embedding_df["embedding"]

        # Initialize world state
        world_state = CampaignWorldState(activity_series, naturalness_series, embedding_series)

        # Measure some variants
        seq_ids_to_measure = ["seq_0", "seq_5", "seq_10"]
        world_state.measure_variant_activities(seq_ids_to_measure)

        # Get unmeasured variants
        unmeasured_activity_series = world_state.get_unmeasured_variants_activity_df()
        unmeasured_naturalness_series = world_state.get_unmeasured_naturalness_series()
        unmeasured_embeddings_series = world_state.get_unmeasured_embeddings_series()

        # Verify unmeasured variants
        assert len(unmeasured_activity_series) == len(activity_series) - len(seq_ids_to_measure)
        assert len(unmeasured_naturalness_series) == len(naturalness_series) - len(
            seq_ids_to_measure
        )
        assert len(unmeasured_embeddings_series) == len(embedding_series) - len(seq_ids_to_measure)

        # Verify measured variants are excluded
        for seq_id in seq_ids_to_measure:
            assert seq_id not in unmeasured_activity_series.index
            assert seq_id not in unmeasured_naturalness_series.index
            assert seq_id not in unmeasured_embeddings_series.index

    def test_get_measured_variants(self):
        """Test getting measured variants."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Get the series
        activity_series = dataset_generator.activity_df["DMS_score"]
        naturalness_series = dataset_generator.naturalness_df["wt_marginal"]
        embedding_series = dataset_generator.embedding_df["embedding"]

        # Initialize world state
        world_state = CampaignWorldState(activity_series, naturalness_series, embedding_series)

        # Measure some variants
        seq_ids_to_measure = ["seq_0", "seq_5", "seq_10"]
        world_state.measure_variant_activities(seq_ids_to_measure)

        # Get measured variants
        measured_activity_series = world_state.get_measured_activity_series()
        measured_naturalness_series = world_state.get_measured_naturalness_series()
        measured_embeddings_series = world_state.get_measured_embeddings_series()

        # Verify measured variants
        assert len(measured_activity_series) == len(seq_ids_to_measure)
        assert len(measured_naturalness_series) == len(seq_ids_to_measure)
        assert len(measured_embeddings_series) == len(seq_ids_to_measure)

        # Verify only measured variants are included
        for seq_id in seq_ids_to_measure:
            assert seq_id in measured_activity_series.index
            assert seq_id in measured_naturalness_series.index
            assert seq_id in measured_embeddings_series.index


class TestEvaluateMetrics:
    """Tests for the _evaluate_metrics function."""

    def test_evaluate_metrics(self):
        """Test evaluating metrics."""
        # Create sample data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.2, 1.8, 3.1, 3.9, 5.2])

        # Calculate metrics
        metrics = _evaluate_metrics(y_true, y_pred)

        # Verify metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "pearson" in metrics
        assert "spearman" in metrics

        # Verify metric values
        assert metrics["mse"] == pytest.approx(0.03, abs=0.01)
        assert metrics["rmse"] == pytest.approx(0.173, abs=0.01)
        assert metrics["pearson"] == pytest.approx(0.996, abs=0.01)
        assert metrics["spearman"] == pytest.approx(1.0, abs=0.01)

    def test_evaluate_metrics_degenerate(self):
        """Test evaluating metrics with degenerate predictions."""
        # Create sample data with degenerate predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # All same value

        # Calculate metrics
        metrics = _evaluate_metrics(y_true, y_pred)

        # Verify metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert metrics["pearson"] is None
        assert metrics["spearman"] is None

        # Verify metric values
        # MSE = (1-2)² + (2-2)² + (3-2)² + (4-2)² + (5-2)² = 1 + 0 + 1 + 4 + 9 = 15
        # MSE = 15/5 = 3.0
        assert metrics["mse"] == pytest.approx(3.0, abs=0.01)
        assert metrics["rmse"] == pytest.approx(1.732, abs=0.01)  # sqrt(3.0)


@patch("folde.campaign.get_zero_shot_model")
@patch("folde.campaign.get_few_shot_model")
class TestRunSingleSimulation:
    """Tests for the _run_single_simulation function."""

    def test_run_single_simulation_first_round(self, mock_get_few_shot, mock_get_zero_shot):
        """Test running a single simulation for the first round only."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Get the series and list of available seq_ids
        available_seq_ids = dataset_generator.activity_df.index.tolist()
        activity_series = dataset_generator.activity_df["DMS_score"]
        naturalness_series = dataset_generator.naturalness_df["wt_marginal"]
        embedding_series = dataset_generator.embedding_df["embedding"]

        # Create model configuration
        model_config = FolDEModelConfig(
            name="test_config",
            naturalness_model_id="test_naturalness",
            embedding_model_id="test_embedding",
            zero_shot_model_name="NaturalnessZeroShotModel",
            zero_shot_model_params={"temperature": 0.1},
            few_shot_model_name="MLPFewShotModel",
            few_shot_model_params={},
        )

        # Create mock models
        mock_zero_shot_model = MockZeroShotModel()
        mock_get_zero_shot.return_value = mock_zero_shot_model

        # Run simulation for 1 round only
        result = _run_single_simulation(
            available_seq_ids,
            activity_series,
            naturalness_series,
            embedding_series,
            round_size=5,
            config=model_config,
            max_rounds=1,
            random_seed=42,
        )

        # Verify simulation results
        assert result.rounds == 1
        assert result.variant_pool_size == len(dataset_generator.activity_df)
        assert len(result.mutant_metrics) == 5  # round_size
        assert len(result.round_metrics) == 1

        # Verify zero-shot model was used
        mock_get_zero_shot.assert_called_once_with(
            model_config.zero_shot_model_name, **model_config.zero_shot_model_params
        )
        assert mock_zero_shot_model.predict_called

        # Verify few-shot model was not used
        mock_get_few_shot.assert_not_called()

        # Verify round metrics
        round_metric = result.round_metrics[0]
        assert round_metric.round_num == 1
        assert isinstance(round_metric.model_spearman, float)

        # Verify mutant metrics
        for mutant_metric in result.mutant_metrics:
            assert mutant_metric.round_found == 1
            assert mutant_metric.seq_id in dataset_generator.activity_df.index
            assert isinstance(mutant_metric.activity, float)
            assert isinstance(mutant_metric.predicted_activity, float)
            assert isinstance(mutant_metric.percentile, float)

    def test_run_single_simulation_multiple_rounds(self, mock_get_few_shot, mock_get_zero_shot):
        """Test running a single simulation for multiple rounds."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Get the series and list of available seq_ids
        available_seq_ids = dataset_generator.activity_df.index.tolist()
        activity_series = dataset_generator.activity_df["DMS_score"]
        naturalness_series = dataset_generator.naturalness_df["wt_marginal"]
        embedding_series = dataset_generator.embedding_df["embedding"]

        # Create model configuration
        model_config = FolDEModelConfig(
            name="test_config",
            naturalness_model_id="test_naturalness",
            embedding_model_id="test_embedding",
            zero_shot_model_name="NaturalnessZeroShotModel",
            zero_shot_model_params={"temperature": 0.1},
            few_shot_model_name="MLPFewShotModel",
            few_shot_model_params={},
        )

        # Create mock models
        mock_zero_shot_model = MockZeroShotModel()
        mock_few_shot_model = MockFewShotModel()
        mock_get_zero_shot.return_value = mock_zero_shot_model
        mock_get_few_shot.return_value = mock_few_shot_model

        # Run simulation for multiple rounds
        result = _run_single_simulation(
            available_seq_ids,
            activity_series,
            naturalness_series,
            embedding_series,
            round_size=5,
            config=model_config,
            max_rounds=3,
            random_seed=42,
        )

        # Verify simulation results
        assert result.rounds == 3
        assert result.variant_pool_size == len(dataset_generator.activity_df)
        assert len(result.mutant_metrics) == 15  # 5 per round * 3 rounds
        assert len(result.round_metrics) == 3

        # Verify zero-shot model was used in first round
        mock_get_zero_shot.assert_called_once_with(
            model_config.zero_shot_model_name, **model_config.zero_shot_model_params
        )
        assert mock_zero_shot_model.predict_called

        # Verify few-shot model was used in subsequent rounds
        mock_get_few_shot.assert_called_with(
            model_config.few_shot_model_name, **model_config.few_shot_model_params
        )
        assert mock_few_shot_model.fit_called
        assert mock_few_shot_model.predict_called

        # Verify round metrics
        for i, round_metric in enumerate(result.round_metrics):
            assert round_metric.round_num == i + 1
            assert isinstance(round_metric.model_spearman, float)

        # Verify mutant metrics by round
        round_1_mutants = [m for m in result.mutant_metrics if m.round_found == 1]
        round_2_mutants = [m for m in result.mutant_metrics if m.round_found == 2]
        round_3_mutants = [m for m in result.mutant_metrics if m.round_found == 3]
        assert len(round_1_mutants) == 5
        assert len(round_2_mutants) == 5
        assert len(round_3_mutants) == 5

        # Verify all seq_ids are unique (no repeat selections)
        all_seq_ids = [m.seq_id for m in result.mutant_metrics]
        assert len(all_seq_ids) == len(set(all_seq_ids))


@patch("folde.campaign.get_proteingym_dataset")
class TestSimulateCampaignSimple:
    """Tests for the simulate_campaign function using a simplified approach."""

    def test_simulate_campaign_simple(self, mock_get_dataset):
        """Test simulating a campaign with a simplified approach to avoid ProcessPoolExecutor issues."""
        # Create test datasets
        dataset_generator = TestDatasetGeneration()
        dataset_generator.test_create_simulated_dataset()

        # Mock dataset retrieval
        mock_get_dataset.return_value = (
            dataset_generator.naturalness_df,
            dataset_generator.embedding_df,
            dataset_generator.activity_df,
        )

        # Create model configuration
        model_config = FolDEModelConfig(
            name="test_config",
            naturalness_model_id="test_naturalness",
            embedding_model_id="test_embedding",
            zero_shot_model_name="NaturalnessZeroShotModel",
            zero_shot_model_params={"temperature": 0.1},
            few_shot_model_name="MLPFewShotModel",
            few_shot_model_params={},
        )

        # Since this function is complex to mock, let's just verify that our mocks are called
        # correctly and that the function doesn't raise exceptions
        with patch("folde.campaign.ProcessPoolExecutor") as mock_executor:
            # Setup a mock executor context
            mock_context = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_context

            # Setup a mock Future and SimulationResult
            future = MagicMock()
            sim_result = SimulationResult(
                config=model_config,
                rounds=3,
                variant_pool_size=len(dataset_generator.activity_df),
                mutant_metrics=[],
                round_metrics=[],
            )
            future.result.return_value = sim_result
            mock_context.submit.return_value = future

            # Run the function with minimal settings
            result = simulate_campaign(
                dms_id="test_dms",
                round_size=5,
                number_of_simulations=1,  # Keep it simple with 1 simulation
                config_list=[model_config],
                max_rounds=3,
            )

        # Just verify the mock was called and basic structure
        assert mock_get_dataset.called
        assert result.dms_id == "test_dms"
        # The test passed without exceptions!


@patch("folde.campaign.simulate_campaign")
class TestSimulateCampaigns:
    """Tests for the simulate_campaigns function."""

    def test_simulate_campaigns(self, mock_simulate_campaign):
        """Test simulating multiple campaigns."""
        # Create model configuration
        model_config = FolDEModelConfig(
            name="test_config",
            naturalness_model_id="test_naturalness",
            embedding_model_id="test_embedding",
            zero_shot_model_name="NaturalnessZeroShotModel",
            zero_shot_model_params={"temperature": 0.1},
            few_shot_model_name="MLPFewShotModel",
            few_shot_model_params={},
        )

        # Create mocked campaign results
        mock_campaign_1 = MagicMock()
        mock_campaign_2 = MagicMock()
        mock_simulate_campaign.side_effect = [mock_campaign_1, mock_campaign_2]

        # Run campaign simulations
        from folde.campaign import simulate_campaigns

        result = simulate_campaigns(
            name="test_eval",
            dms_ids=["test_dms_1", "test_dms_2"],
            round_size=5,
            number_of_simulations=2,
            config_list=[model_config],
        )

        # Verify results
        assert result.name == "test_eval"
        assert len(result.campaign_results) == 2
        assert result.campaign_results[0] == mock_campaign_1
        assert result.campaign_results[1] == mock_campaign_2

        # Verify calls to simulate_campaign
        assert mock_simulate_campaign.call_count == 2
        mock_simulate_campaign.assert_any_call(
            "test_dms_1",
            round_size=5,
            number_of_simulations=2,
            config_list=[model_config],
        )
        mock_simulate_campaign.assert_any_call(
            "test_dms_2",
            round_size=5,
            number_of_simulations=2,
            config_list=[model_config],
        )
