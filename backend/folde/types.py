from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FolDEModelConfig(BaseModel):
    name: str
    data_split_mode: Optional[str] = None  # Can be "1-VS-REST" or "2-VS-REST" etc.
    one_mutation_at_a_time: bool = False
    naturalness_model_id: str
    naturalness_columns: Optional[List[str]] = None
    embedding_model_id: str
    embedding_column: Optional[str] = None
    zero_shot_model_name: str
    zero_shot_model_params: Dict[str, Any]
    few_shot_model_name: str
    few_shot_model_params: Dict[str, Any]

    # DEPRECATED.
    few_shot_naturalness_column: Optional[str] = None


class MutantMetrics(BaseModel):
    """Stores dense information about each mutants tested in the simulation."""

    seq_id: str
    round_found: int
    activity: float
    predicted_activity: float
    predicted_activity_stddev: float | None = None
    percentile: float
    relevant_mutants: List[str]


class RoundMetrics(BaseModel):
    """Stores expensive-to-compute per-round metrics, such as model characterization."""

    round_num: int
    model_spearman: float
    misc: Dict[str, Any]


class SimulationResult(BaseModel):
    rounds: int
    variant_pool_size: int
    round_metrics: List[RoundMetrics]
    mutant_metrics: List[MutantMetrics]


class SingleConfigCampaignResult(BaseModel):
    config: FolDEModelConfig
    simulation_results: List[SimulationResult]


class CampaignResult(BaseModel):
    dms_id: str
    round_size: int
    number_of_simulations: int
    activity_column: str
    min_activity: float
    median_activity: float
    max_activity: float
    max_rounds: int
    random_seed: int
    config_results: List[SingleConfigCampaignResult]


class ModelEvaluation(BaseModel):
    name: str
    campaign_results: List[CampaignResult]


class ModelDiff(BaseModel):
    name: str
    # key is the path to the parameter, value is the new value
    diffs: dict[str, Any]
