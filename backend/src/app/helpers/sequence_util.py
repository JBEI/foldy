import re
import random
from collections import defaultdict
from re import fullmatch
import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any, DefaultDict

from dnachisel import biotools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from werkzeug.exceptions import BadRequest


VALID_AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
# We are tolerant of selenocysteine and other non-standard amino acids.
EXPANSIVE_VALID_AMINO_ACIDS: str = "ACDEFGHIKLMNOPQRSTUVWY"


def get_locus_from_allele_id(allele_id: str) -> int:
    """Returns the 1-based index of the locus from the allele id."""
    return int(allele_id[1:-1])


def get_loci_set(seq_id: str) -> Set[int]:
    """Get the set of loci (1-based indices) from a sequence ID.
    
    Args:
        seq_id: A sequence ID string (e.g., "G3W_A12T" or "WT")
    
    Returns:
        A set of locus indices
    """
    if seq_id == "WT":
        return set()
    return {get_locus_from_allele_id(allele) for allele in seq_id.split("_")}


def allele_set_to_seq_id(allele_set: Set[str]) -> str:
    """Converts the allele set to a standard ID (eg: {A12T, G3W}->"G3W_A12T").
    
    Args:
        allele_set: Set of allele strings
        
    Returns:
        A standardized sequence ID
    """
    if allele_set == {""} or len(allele_set) == 0:
        return "WT"
    allele_list = sorted(
        list(allele_set), key=lambda allele: (int(allele[1:-1]), allele[-1])
    )
    return "_".join(allele_list)


def maybe_get_allele_id_error_message(wt_aa_seq: str, allele_id: str) -> Optional[str]:
    """Returns an error message if allele id is invalid, otherwise None.
    
    Args:
        wt_aa_seq: The wild-type amino acid sequence
        allele_id: The allele ID to validate (e.g., "A12T")
        
    Returns:
        An error message string if the allele ID is invalid, None otherwise
    """
    assert type(wt_aa_seq) == str, f"wt_aa_seq must be a string, got {type(wt_aa_seq)}"
    fn = re.compile(r"([A-Z])(\d+)([A-Z])")
    m = fn.match(allele_id)
    if not m:
        return f"Allele is improperly formatted {allele_id}"

    # Check amino acids.
    if not m.groups()[0] in EXPANSIVE_VALID_AMINO_ACIDS:
        return f'Invalid allele "{allele_id}": first character must be a valid amino acid, got "{m.groups()[0]}"'
    if not m.groups()[2] in EXPANSIVE_VALID_AMINO_ACIDS:
        return f'Invalid allele "{allele_id}": third character must be a valid amino acid, got "{m.groups()[2]}"'

    allele_idx = int(m.groups()[1]) - 1
    if allele_idx < 0 or allele_idx >= len(wt_aa_seq):
        return f"Allele is out of bounds (got {allele_idx+1} but protein only has {len(wt_aa_seq)} AAs)."
    if wt_aa_seq[allele_idx] != m.groups()[0]:  # Changed 'is not' to '!=' for string comparison
        return f"Allele does not correspond to WT sequence: wt residue at {m.groups()[1]} is {wt_aa_seq[int(m.groups()[1])-1]} but allele was {allele_id}"
    
    return None


def maybe_get_seq_id_error_message(wt_aa_seq: str, seq_id: Any) -> Optional[str]:
    """Returns an error message if seq_id is invalid, otherwise None.

    Checks that:
    * seq_id has no duplicate loci
    * loci are sorted
    * alleles are valid
    
    Args:
        wt_aa_seq: The wild-type amino acid sequence
        seq_id: The sequence ID to validate
        
    Returns:
        An error message string if the sequence ID is invalid, None otherwise
    """
    if type(seq_id) != str:
        return f"seq_id must be a string, got {seq_id} with type {type(seq_id)}"
    if seq_id == "WT":
        return None
    allele_list = seq_id.split("_")
    for allele in allele_list:
        error_msg = maybe_get_allele_id_error_message(wt_aa_seq, allele)
        if error_msg:
            return error_msg
    is_sorted = all(
        [
            get_locus_from_allele_id(left) <= get_locus_from_allele_id(right)
            for left, right in zip(allele_list[:-1], allele_list[1:])
        ]
    )
    if not is_sorted:
        return "Loci are not sorted"
    return None


def get_seq_ids_for_deep_mutational_scan(
    wt_aa_seq: str, dms_starting_seq_ids: List[str], extra_seq_ids: List[str]
) -> List[str]:
    """Do a DMS starting with a few mutants of the provided protein.

    The base of the mutational scan is conducted based on the "starting_seq_id",
    seq IDs are of the form A23T_Y45G, where the alleles are sorted by locus.

    For each starting_sequence
       for each locus
           if there's already a mutation at that locus:
               delete the mutation at that locus and consider any other mutation, including WT
           if there's not:
               consider all possible mutations besides WT
               
    Args:
        wt_aa_seq: The wild-type amino acid sequence
        dms_starting_seq_ids: List of starting sequence IDs for deep mutational scan
        extra_seq_ids: Additional sequence IDs to include
        
    Returns:
        List of all sequence IDs generated by the deep mutational scan
    """

    def assert_valid_seq_id(seq_id: str) -> None:
        """Validate a sequence ID, raising ValueError if invalid.
        
        Args:
            seq_id: The sequence ID to validate
        
        Raises:
            ValueError: If the sequence ID is invalid
        """
        seq_id_error_msg = maybe_get_seq_id_error_message(wt_aa_seq, seq_id)
        if seq_id_error_msg:
            raise ValueError(f"Invalid seq_id '{seq_id}': {seq_id_error_msg}")

    def allele_is_at_locus(allele_id: str, locus: int) -> bool:
        """Returns true if the allele (1-based) is at that locus (1-based).
        
        Args:
            allele_id: The allele ID (e.g., "A12T")
            locus: The 1-based locus index
            
        Returns:
            True if the allele is at the specified locus, False otherwise
        """
        return int(allele_id[1:-1]) == locus

    def seq_id_to_allele_list(seq_id: str) -> List[str]:
        """Converts the seq_id to an allele list (eg: "G3W_A12T"->["G3W", "A12T"]).
        
        Args:
            seq_id: The sequence ID
            
        Returns:
            List of allele strings
        """
        if seq_id == "WT":
            return []
        return seq_id.split("_")

    assert type(wt_aa_seq) == str, f"wt_aa_seq must be a string, got {type(wt_aa_seq)}"

    # Validate inputs.
    for starting_seq_id in dms_starting_seq_ids:
        assert_valid_seq_id(starting_seq_id)
    for extra_seq_id in extra_seq_ids:
        assert_valid_seq_id(extra_seq_id)

    seq_id_set: Set[str] = set()

    for starting_seq_id in dms_starting_seq_ids:
        starting_seq_allele_list = seq_id_to_allele_list(starting_seq_id)

        # Make sure to normalize the starting seq id before including in set.
        seq_id_set.add(allele_set_to_seq_id(set(starting_seq_allele_list)))

        for aa_idx in range(len(wt_aa_seq)):
            if any(
                [
                    get_locus_from_allele_id(allele) == aa_idx + 1
                    for allele in starting_seq_allele_list
                ]
            ):
                # The case where this locus is already mutated.

                seq_base_allele_list = [
                    allele
                    for allele in starting_seq_allele_list
                    if get_locus_from_allele_id(allele) != aa_idx + 1
                ]
                for alternative_aa in VALID_AMINO_ACIDS:
                    if wt_aa_seq[aa_idx] == alternative_aa:
                        seq_id_set.add(allele_set_to_seq_id(set(seq_base_allele_list)))
                    else:
                        new_mut_id = f"{wt_aa_seq[aa_idx]}{aa_idx+1}{alternative_aa}"
                        seq_id_set.add(
                            allele_set_to_seq_id(set(seq_base_allele_list + [new_mut_id]))
                        )

            else:
                # The case where this locus is not mutated.
                for alternative_aa in VALID_AMINO_ACIDS:
                    if wt_aa_seq[aa_idx] == alternative_aa:
                        continue
                    else:
                        new_mut_id = f"{wt_aa_seq[aa_idx]}{aa_idx+1}{alternative_aa}"
                        seq_id_set.add(
                            allele_set_to_seq_id(
                                set(starting_seq_allele_list + [new_mut_id])
                            )
                        )

    for extra_seq_id in extra_seq_ids:
        extra_seq_allele_list = seq_id_to_allele_list(extra_seq_id)

        # Make sure to normalize the starting seq id before including in set.
        seq_id_set.add(allele_set_to_seq_id(set(extra_seq_allele_list)))

    return list(seq_id_set)


def seq_id_to_seq(wt_aa_seq: str, seq_id: str) -> str:
    """Convert the seq ID into a sequence.
    
    Args:
        wt_aa_seq: The wild-type amino acid sequence
        seq_id: The sequence ID to convert
        
    Returns:
        The amino acid sequence corresponding to the sequence ID
        
    Raises:
        AssertionError: If the sequence ID is invalid
    """
    if seq_id == "WT":
        return wt_aa_seq
    seq = wt_aa_seq
    for allele in seq_id.split("_"):
        assert len(allele) >= 3, f'Invalid seq_id, too short: "{seq_id}"'
        wt_allele = allele[0]
        idx = int(allele[1:-1]) - 1
        mut_allele = allele[-1]
        assert (
            seq[idx] == wt_allele
        ), f"Invalid seq_id '{seq_id}' specifically '{allele}': wt allele is {seq[idx]}"
        seq = seq[:idx] + mut_allele + seq[idx + 1 :]
    return seq


def process_and_validate_evolve_input_files(
    wt_aa_seq: str, 
    raw_activity_df: pd.DataFrame, 
    raw_embedding_df: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepares raw inputs for EvolvePRO logic, raises ValueError if inputs are invalid.
    
    Args:
        wt_aa_seq: The wild-type amino acid sequence
        raw_activity_df: DataFrame containing activity measurements
        raw_embedding_df: Optional DataFrame containing embeddings
        
    Returns:
        If raw_embedding_df is None, returns the validated activity DataFrame.
        Otherwise, returns a tuple of (activity_df, embedding_df).
        
    Raises:
        ValueError: If inputs are invalid
    """
    activity_df = raw_activity_df.copy()
    embedding_df = raw_embedding_df.copy() if raw_embedding_df is not None else None

    if "seq_id" not in activity_df.columns:
        raise ValueError(
            f"Activity file must contain a 'seq_id' column, got {activity_df.columns}"
        )
    if "activity" not in activity_df.columns:
        raise ValueError(
            f"Activity file must contain a 'activity' column, got {activity_df.columns}"
        )
    if embedding_df is not None:
        if "seq_id" not in embedding_df.columns:
            raise ValueError(
                f"Embedding file must contain a 'seq_id' column, got {embedding_df.columns}"
            )
        if "embedding" not in embedding_df.columns:
            raise ValueError(
                f"Embedding file must contain a 'embedding' column, got {embedding_df.columns}"
            )

    # activity_df.replace({"seq_id": {np.nan: ""}}, inplace=True)  # "WT": "",
    for seq_id in activity_df.seq_id:
        seq_id_error_msg = maybe_get_seq_id_error_message(wt_aa_seq, seq_id)
        if seq_id_error_msg:
            raise ValueError(f"Invalid seq_id '{seq_id}': {seq_id_error_msg}")

    # embedding_df.fillna({"seq_id": ""}, inplace=True)

    if embedding_df is None:
        return activity_df
    else:
        return activity_df, embedding_df


def get_measured_and_unmeasured_mutant_seq_ids(activity_df: pd.DataFrame, embedding_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns the measured and unmeasured sequence IDs from activity and embedding DataFrames.
    
    Args:
        activity_df: DataFrame containing measured activities
        embedding_df: DataFrame containing embeddings for all mutants
        
    Returns:
        Tuple containing:
        - measured_mutants: List of sequence IDs with both activity and embedding data
        - unmeasured_mutants: List of sequence IDs with embedding data but no activity data
        
    Raises:
        BadRequest: If any sequence ID is NaN
    """
    # Make a copy to avoid modifying the original
    activity_df_copy = activity_df.copy()
    embedding_df_copy = embedding_df.copy()
    
    # Set index properly
    activity_df_copy = activity_df_copy.set_index("seq_id")
    embedding_df_copy = embedding_df_copy.set_index("seq_id")

    activity_mutants: Set[Any] = set(activity_df["seq_id"])
    embedding_mutants: Set[Any] = set(embedding_df["seq_id"])

    if np.nan in activity_mutants:
        raise BadRequest("Activity file contains NaN seq_ids.")

    if np.nan in embedding_mutants:
        raise BadRequest("Embedding file contains NaN seq_ids.")

    # Calculate overlap and test sets
    measured_mutants = list(activity_mutants.intersection(embedding_mutants))
    unmeasured_mutants = list(embedding_mutants - activity_mutants)
    return measured_mutants, unmeasured_mutants


def train_and_predict_activities(
    activity_df: pd.DataFrame, embedding_df: pd.DataFrame, mode: str
) -> Tuple[List[str], List[str], Union[RandomForestRegressor, MLPRegressor], pd.DataFrame]:
    """Train a machine learning model on measured mutants and predict activities for all mutants.

    Args:
        activity_df: DataFrame containing measured activities with mutant seq_ids as index
        embedding_df: DataFrame containing embeddings for all mutants (measured + unmeasured)
        mode: Model type to use ("randomforest" or "mlp")

    Returns:
        Tuple containing:
        - measured_mutants: list of measured mutant seq_ids
        - unmeasured_mutants: list of unmeasured mutant seq_ids
        - model: trained model (RandomForestRegressor or MLPRegressor)
        - predicted_activity_df: DataFrame containing predictions for all mutants with columns:
          * seq_id: mutant identifier
          * predicted_activity: model predictions
          * relevant_measured_mutants: space-separated list of measured mutants sharing loci
          * actual_activity: measured activity (if available)
          
    Raises:
        ValueError: If an invalid model choice is provided
    """
    # Get measured and unmeasured mutant sets
    measured_mutants, unmeasured_mutants = get_measured_and_unmeasured_mutant_seq_ids(
        activity_df, embedding_df
    )

    # Prepare training data
    X_train = np.vstack(
        [json.loads(x) for x in embedding_df.loc[activity_df.index].embedding]
    )
    y_train = activity_df.activity.to_numpy()

    model: Union[RandomForestRegressor, MLPRegressor]
    if mode == "randomforest":
        model = RandomForestRegressor(
            n_estimators=100,
            criterion="friedman_mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
    elif mode == "mlp":
        model = MLPRegressor(
            random_state=1, max_iter=5000, hidden_layer_sizes=(100, 50)
        )
    else:
        raise ValueError(f"Invalid model choice: {mode}")
    model.fit(X_train, y_train)

    # Prepare prediction data for all mutants
    all_mutants_embedding_array = np.vstack(
        [
            json.loads(x)
            for x in embedding_df.loc[measured_mutants + unmeasured_mutants].embedding
        ]
    )

    # Make predictions
    y_all_pred = model.predict(all_mutants_embedding_array)

    # Create results DataFrame
    predicted_activity_df = pd.DataFrame(
        {
            "seq_id": measured_mutants + unmeasured_mutants,
            "predicted_activity": y_all_pred,
        }
    )
    predicted_activity_df = predicted_activity_df.set_index("seq_id")

    # Add relevant measured mutants - we need to get the index as a Series first
    idx_series = predicted_activity_df.index.to_series()
    predicted_activity_df["relevant_measured_mutants"] = (
        idx_series.apply(
            lambda seq_id: " ".join(
                [m for m in measured_mutants if get_loci_set(m) & get_loci_set(seq_id)]
            )
        )
    )

    # Add actual activities where available
    predicted_activity_df["actual_activity"] = predicted_activity_df.join(
        activity_df.groupby(level=0).activity.mean(), how="left"
    ).activity

    return (
        measured_mutants,
        unmeasured_mutants,
        model,
        predicted_activity_df.sort_values("predicted_activity", ascending=False),
    )


def get_cross_validation_holdout_sets(
    seq_ids: List[str], 
    min_held_out_mutants: int = 1, 
    max_held_out_mutants: Optional[int] = None
) -> Tuple[List[List[str]], List[int]]:
    """Returns a list of holdout sets, raises ValueError if impossible.

    We make splits in a "smart" way because we don't want leakage. Our approach is, for each
    split, we choose a single locus to be held out. For example, we might decide that any mutants
    with mutations at locus 15 are held out. Our intuition is that mutations at a specific locus
    correlate. We want each split to include at least one mutation in the test set that has never
    been seen before.

    For example: if seq_ids = ["A1B", "Y15G", "A1C_Y15P", "A1B_G22H", "G22H"], we might put
    ["Y15G", "A1C_Y15P"] in the holdout set, to test whether the model can predict the effect
    of mutations at locus 15.

    Args:
        seq_ids: List of sequence IDs to split
        min_held_out_mutants: Minimum number of mutants to include in the holdout set
        max_held_out_mutants: Maximum number of mutants to include in the holdout set

    Returns:
        Tuple containing:
        - List of holdout sets (each set is a list of sequence IDs)
        - List of viable loci used for creating the holdout sets
    """

    # Our approach is to choose loci to hold out that match the requirements.

    # Number of mutants with a mutation at each locus.
    loci_mutant_counts: DefaultDict[int, int] = defaultdict(int)
    for seq_id in seq_ids:
        if seq_id == "WT":
            continue
        for allele in seq_id.split("_"):
            locus = get_locus_from_allele_id(allele)
            loci_mutant_counts[locus] += 1

    # We want to hold out at least one mutant at each locus.
    viable_loci: List[int] = []
    for locus, count in loci_mutant_counts.items():
        if count < min_held_out_mutants:
            continue
        if max_held_out_mutants is not None and count > max_held_out_mutants:
            continue
        viable_loci.append(locus)

    holdout_lists: List[List[str]] = []
    for holdout_locus in viable_loci:
        holdout_list: List[str] = []
        for seq_id in seq_ids:
            if seq_id == "WT":
                # WT is never in the holdout set.
                continue
            if any(
                get_locus_from_allele_id(allele) == holdout_locus
                for allele in seq_id.split("_")
            ):
                holdout_list.append(seq_id)
        holdout_lists.append(holdout_list)
    return holdout_lists, viable_loci


def determine_quartile(value: float, quartiles: np.ndarray) -> int:
    """
    Determine which quartile a value falls into, based on precomputed quartiles.

    Args:
        value: The value to classify
        quartiles: Array of bin edges for quartile determination

    Returns:
        The quartile index (0, 1, 2, or 3 for quartiles, or appropriate bin index for other binning)
    """
    # Calculate the result with intermediate steps to avoid type errors
    digitize_result = np.digitize(value, quartiles, right=False)
    quartile_candidate = digitize_result - 1
    max_index = len(quartiles) - 2
    # Use Python's min to ensure we get a native Python int
    index = min(int(quartile_candidate), max_index)
    return index


def get_cross_validation_holdout_sets_with_stratification(
    seq_ids: List[str],
    target_values: List[float],
    N: int,
    min_data_per_quartile: int = 1,
    max_data_per_quartile: Optional[int] = None,
    max_num_iterations: int = 1000,
    max_loci_to_consider_each_round: int = 50,
    max_holdout_set_attempts: int = 10,
    num_bins: int = 4,
) -> Tuple[List[Set[str]], List[Set[int]]]:
    """
    Generates stratified cross-validation holdout sets based on loci and quartiles of target values.

    Args:
        seq_ids: List of sequence IDs
        target_values: Target values corresponding to each sequence ID
        N: Number of holdout sets to generate
        min_data_per_quartile: Minimum number of data points per quartile in each holdout set
        max_data_per_quartile: Maximum number of data points per quartile in each holdout set
        max_num_iterations: Maximum iterations to try building a valid holdout set
        max_loci_to_consider_each_round: Maximum loci to consider in each rejection sampling round
        max_holdout_set_attempts: Maximum attempts to build a valid holdout set
        num_bins: Number of bins to use for stratification

    Returns:
        Tuple containing:
        - List of holdout sets (each set is a set of sequence IDs)
        - List of holdout loci sets

    Raises:
        ValueError: If unable to generate holdout sets within constraints
        AssertionError: If holdout set creation fails after max attempts
    """
    # Compute equal-width bins of the target values
    min_value, max_value = min(target_values), max(target_values)
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    # Create loci_mutant_counts as a dict from locus to quartile counts
    loci_mutant_counts: DefaultDict[int, List[int]] = defaultdict(lambda: [0] * num_bins)  # List for quartiles
    for seq_id, target_value in zip(seq_ids, target_values):
        if seq_id == "WT":
            continue  # Skip WT
        for allele in seq_id.split("_"):
            locus = get_locus_from_allele_id(allele)
            quartile = determine_quartile(target_value, bin_edges)
            loci_mutant_counts[locus][quartile] += 1

    def get_quartile_counts(holdout_set: Set[int]) -> List[int]:
        """Get counts of data points in each quartile for a given holdout set.
        
        Args:
            holdout_set: Set of loci to include in the holdout
            
        Returns:
            List of counts for each quartile
        """
        return [
            sum(loci_mutant_counts[locus][quartile] for locus in holdout_set)
            for quartile in range(num_bins)
        ]

    def try_build_holdout_set() -> Set[int]:
        """Try to build a holdout set. Raise ValueError if we fail.
        
        Returns:
            A set of loci forming a valid holdout set
            
        Raises:
            ValueError: If unable to build a valid holdout set
        """
        holdout_set: Set[int] = set()
        iterations = 0
        while iterations < max_num_iterations:
            iterations += 1

            # Identify the quartile with the fewest data points
            holdout_quartile_counts = get_quartile_counts(holdout_set)
            min_quartile = holdout_quartile_counts.index(min(holdout_quartile_counts))
            # Filter loci with nonzero counts in min_quartile
            candidate_loci = [
                locus
                for locus, counts in loci_mutant_counts.items()
                if counts[min_quartile] > 0 and locus not in holdout_set
            ]
            if not candidate_loci:
                # No viable loci to sample, raise an error
                raise ValueError(
                    f"Cannot satisfy stratification requirements. Try relaxing constraints."
                )

            # Randomly sample a locus
            sampled_locus = None
            for _ in range(max_loci_to_consider_each_round):
                sampled_locus = random.choice(candidate_loci)

                tentative_holdout_set = holdout_set.union({sampled_locus})
                tentative_holdout_quartile_counts = get_quartile_counts(
                    tentative_holdout_set
                )

                if all(
                    count <= (max_data_per_quartile or float("inf"))
                    for count in tentative_holdout_quartile_counts
                ):
                    # Commit this locus to the holdout set
                    holdout_set = tentative_holdout_set
                    break  # Exit the rejection sampling loop

            # We've finished one attempt. Check if we've met the stratification
            # criteria. If not, we dump it and try again.
            if all(
                min_data_per_quartile
                <= count
                <= (max_data_per_quartile or float("inf"))
                for count in get_quartile_counts(holdout_set)
            ):
                break

        if iterations >= max_num_iterations:
            raise ValueError(
                "Exceeded maximum number of iterations while building holdout sets."
            )

        return holdout_set

    # Generate N holdout sets.
    holdout_sets: List[Set[int]] = []
    for _ in range(N):
        holdout_set = None
        # We are willing to try max_num_iterations times to build a valid holdout set.
        # Failures can happen because we happened to choose inviable loci to start,
        # or because of bad luck in sampling.
        for holdout_set_attempt in range(max_holdout_set_attempts):
            try:
                holdout_set_candidate = try_build_holdout_set()
                if holdout_set_candidate in holdout_sets:
                    print(
                        f"Discarding holdout set {holdout_set_candidate} because it is a duplicate"
                    )
                    holdout_set = None
                else:
                    holdout_set = holdout_set_candidate
                    break
            except ValueError as e:
                print(
                    f"Failed to build holdout set {holdout_set_attempt} time: {e}",
                    flush=True,
                )
        assert (
            holdout_set is not None
        ), f"Failed to build holdout set after {max_holdout_set_attempts} attempts"
        holdout_sets.append(holdout_set)

    # Convert holdout sets which contain loci to sets of seq_ids
    holdout_sets_seq_ids: List[Set[str]] = [
        {
            seq_id
            for seq_id in seq_ids
            if any(
                [
                    get_locus_from_allele_id(allele) in holdout_set
                    for allele in seq_id.split("_")
                ]
            )
        }
        for holdout_set in holdout_sets
    ]

    return holdout_sets_seq_ids, holdout_sets


def get_cv_splits(
    seq_id_list: List[str], 
    holdout_sets: List[Set[str]], 
    n_splits: int = 5
) -> List[Tuple[List[int], List[int]]]:
    """Returns a list of (train_mutants, test_mutants) tuples for cross_val_score.
    
    Args:
        seq_id_list: List of sequence IDs
        holdout_sets: List of holdout sets, each containing sequence IDs
        n_splits: Number of splits to generate (not used, kept for backwards compatibility)
        
    Returns:
        List of tuples, each containing:
        - List of training indices
        - List of testing indices
    """
    splits: List[Tuple[List[int], List[int]]] = []
    for holdout_set in holdout_sets:
        test_indices: List[int] = []
        train_indices: List[int] = []
        for ii, seq_id in enumerate(seq_id_list):
            if seq_id in holdout_set:
                test_indices.append(ii)
            else:
                train_indices.append(ii)
        splits.append((train_indices, test_indices))
    return splits


def back_translate(aa_seq: str) -> str:
    """Convert an amino acid sequence to a DNA sequence using bacterial codon usage.
    
    Selenocysteine (U) is replaced with cysteine (C) for the translation.
    
    Args:
        aa_seq: Amino acid sequence
        
    Returns:
        Corresponding DNA sequence using preferred bacterial codons
    """
    # Ignore selenocysteine...
    # https://www.frontiersin.org/articles/10.3389/fmolb.2020.00002/full
    aa_without_u = aa_seq.replace("U", "C")
    return biotools.reverse_translate(aa_without_u, table="Bacterial")
    # randomize_codons=True (not used)


def validate_aa_sequence(fold_name: str, sequence: str, af2_model_preset: str) -> None:
    """Validate amino acid sequence and fold name, raising BadRequest if invalid.
    
    Args:
        fold_name: Name of the fold
        sequence: Amino acid sequence to validate
        af2_model_preset: AlphaFold2 model preset ("multimer", "boltz", etc.)
        
    Raises:
        BadRequest: If the fold name or amino acid sequence is invalid
    """
    if not fullmatch(r"[a-zA-Z0-9\-_ ]+", fold_name):
        raise BadRequest(f'Fold name has invalid characters: "{fold_name}"')

    if ":" in sequence or ";" in sequence:
        if af2_model_preset not in ["multimer", "boltz"]:
            raise BadRequest(
                f'This sequence looks like a multimer. Multimers are only supported when using "AF2" and using the AF2 model preset "multimer" (see Advanced Options).'
            )
        chains: List[List[str]] = []
        for chain in sequence.split(";"):
            if len(chain.split(":")) != 2:
                raise BadRequest(
                    f'Each chain (separated by ";") must have a single ":".'
                )
            chain_parts = chain.split(":")
            chains.append([chain_parts[0], chain_parts[1]])
    else:
        chains = [["1", sequence]]

    if len(chains) != len(set([c[0] for c in chains])):
        raise BadRequest("A chain name is duplicated.")

    for chain_name, chain_seq in chains:
        if not fullmatch(r"[a-zA-Z0-9_\-]+", chain_name):
            raise BadRequest(f'Invalid chain name "{chain_name}"')
        for ii, aa in enumerate(chain_seq):
            if aa not in VALID_AMINO_ACIDS:
                raise BadRequest(
                    f'Invalid amino acid "{aa}" at index {ii+1} in chain {chain_name}. Valid amino acids are {"".join(VALID_AMINO_ACIDS)}.'
                )
