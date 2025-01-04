import re
from collections import defaultdict

import numpy as np

VALID_AMINO_ACIDS = "ACDEFGHIKLMNOPQRSTUVWY"


def get_locus_from_allele_id(allele_id):
    """Returns the 1-based index of the locus from the allele id."""
    return int(allele_id[1:-1])


def get_loci_set(seq_id):
    if seq_id == "":
        return set()
    return {get_locus_from_allele_id(allele) for allele in seq_id.split("_")}


def maybe_get_allele_id_error_message(wt_aa_seq, allele_id):
    """Returns an error message if allele id is invalid, otherwise None."""
    fn = re.compile(r"([ACDEFGHIKLMNOPQRSTUVWY])(\d+)[ACDEFGHIKLMNOPQRSTUVWY]")
    m = fn.match(allele_id)
    if not m:
        return f"Allele is improperly formatted {allele_id}"
    allele_idx = int(m.groups()[1]) - 1
    if allele_idx < 0 or allele_idx >= len(wt_aa_seq):
        return f"Allele is out of bounds (got {allele_idx+1} but protein only has {len(wt_aa_seq)} AAs)."
    if wt_aa_seq[allele_idx] is not m.groups()[0]:
        return f"Allele does not correspond to WT sequence: wt residue at {m.groups()[1]} is {wt_aa_seq[int(m.groups()[1])-1]} but allele was {allele_id}"


def maybe_get_seq_id_error_message(wt_aa_seq, seq_id):
    """Returns an error message if seq_id is invalid, otherwise None.

    Checks that:
    * seq_id has no duplicate loci
    * loci are sorted
    * alleles are valid
    """
    if seq_id == "":
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
    wt_aa_seq: str, dms_starting_seq_ids: list[str], extra_seq_ids: list[str]
) -> list[str]:
    """Do a DMS starting with a few mutants of the provided protein.

    The base of the mutational scan is conducted based on the "starting_seq_id",
    seq IDs are of the form A23T_Y45G, where the alleles are sorted by locus.

    For each starting_sequence
       for each locus
           if there's already a mutation at that locus:
               delete the mutation at that locus and consider any other mutation, including WT
           if there's not:
               consider all possible mutations besides WT
    """

    seq_id_set = set()

    def assert_valid_seq_id(seq_id):
        seq_id_error_msg = maybe_get_seq_id_error_message(wt_aa_seq, seq_id)
        if seq_id_error_msg:
            raise ValueError(f"Invalid seq_id {seq_id}: {seq_id_error_msg}")

    def allele_is_at_locus(allele_id, locus):
        """Returns true if the allele (1-based) is at that locus (1-based)."""
        return int(allele_id[1:-1]) == locus

    def allele_set_to_seq_id(allele_set):
        """Converts the allele set to a standard ID (eg: {A12T, G3W}->"G3W_A12T")."""
        if allele_set == {""}:
            return ""
        allele_list = sorted(
            list(allele_set), key=lambda allele: (int(allele[1:-1]), allele[-1])
        )
        return "_".join(allele_list)

    # Validate inputs.
    for starting_seq_id in dms_starting_seq_ids:
        assert_valid_seq_id(starting_seq_id)
    for extra_seq_id in extra_seq_ids:
        assert_valid_seq_id(extra_seq_id)

    for starting_seq_id in dms_starting_seq_ids:
        starting_seq_allele_list = starting_seq_id.split("_") if starting_seq_id else []

        # Make sure to normalize the starting seq id before including in set.
        seq_id_set.add(allele_set_to_seq_id(starting_seq_allele_list))

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
                        seq_id_set.add(allele_set_to_seq_id(seq_base_allele_list))
                    else:
                        new_mut_id = f"{wt_aa_seq[aa_idx]}{aa_idx+1}{alternative_aa}"
                        seq_id_set.add(
                            allele_set_to_seq_id(seq_base_allele_list + [new_mut_id])
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
                                starting_seq_allele_list + [new_mut_id]
                            )
                        )

    for extra_seq_id in extra_seq_ids:
        extra_seq_allele_list = extra_seq_id.split("_") if extra_seq_id else []

        # Make sure to normalize the starting seq id before including in set.
        seq_id_set.add(allele_set_to_seq_id(extra_seq_allele_list))

    return list(seq_id_set)


def seq_id_to_seq(wt_aa_seq, seq_id):
    """Convert the seq ID into a sequence."""
    if seq_id == "":
        return wt_aa_seq
    seq = wt_aa_seq
    for allele in seq_id.split("_"):
        wt_allele = allele[0]
        idx = int(allele[1:-1]) - 1
        mut_allele = allele[-1]
        assert (
            seq[idx] == wt_allele
        ), f"Invalid seq_id {seq_id} specifically {allele}: wt allele is {seq[idx]}"
        seq = seq[:idx] + mut_allele + seq[idx + 1 :]
    return seq


def process_and_validate_evolve_input_files(
    wt_aa_seq, raw_activity_df, raw_embedding_df
):
    """Prepares raw inputs for EvolvePRO logic, raises ValueError if inputs are invalid."""
    activity_df = raw_activity_df.copy()
    embedding_df = raw_embedding_df.copy()

    if "seq_id" not in activity_df.columns:
        raise ValueError(
            f"Activity file must contain a 'seq_id' column, got {activity_df.columns}"
        )
    if "activity" not in activity_df.columns:
        raise ValueError(
            f"Activity file must contain a 'activity' column, got {activity_df.columns}"
        )
    if "seq_id" not in embedding_df.columns:
        raise ValueError(
            f"Embedding file must contain a 'seq_id' column, got {embedding_df.columns}"
        )
    if "embedding" not in embedding_df.columns:
        raise ValueError(
            f"Embedding file must contain a 'embedding' column, got {embedding_df.columns}"
        )

    activity_df.replace({"seq_id": {"WT": ""}}, inplace=True)
    for seq_id in activity_df.seq_id:
        seq_id_error_msg = maybe_get_seq_id_error_message(wt_aa_seq, seq_id)
        if seq_id_error_msg:
            raise ValueError(f"Invalid seq_id {seq_id}: {seq_id_error_msg}")

    embedding_df.fillna({"seq_id": ""}, inplace=True)

    return activity_df, embedding_df


def get_measured_and_unmeasured_mutant_seq_ids(activity_df, embedding_df):
    """Returns the seq_ids"""
    activity_df.index = activity_df.seq_id
    embedding_df.index = embedding_df.seq_id

    activity_mutants = set(activity_df["seq_id"])
    embedding_mutants = set(embedding_df["seq_id"])

    if np.nan in activity_mutants:
        raise BadRequest("Activity file contains NaN seq_ids.")

    if np.nan in embedding_mutants:
        raise BadRequest("Embedding file contains NaN seq_ids.")

    # Calculate overlap and test sets
    measured_mutants = list(activity_mutants.intersection(embedding_mutants))
    unmeasured_mutants = list(embedding_mutants - activity_mutants)
    return measured_mutants, unmeasured_mutants


def get_cross_validation_holdout_sets(
    seq_ids, min_held_out_mutants=1, max_held_out_mutants=None
):
    """Returns a list of holdout sets, raises ValueError if impossible.

    We make splits in a "smart" way because we don't want leakage. Our approach is, for each
    split, we choose a single locus to be held out. For example, we might decide that any mutants
    with mutations at locus 15 are held out. Our intuition is that mutations at a specific locus
    correlate. We want each split to include at least one mutation in the test set that has never
    been seen before.

    For example: if seq_ids = ["A1B", "Y15G", "A1C_Y15P", "A1B_G22H", "G22H"], we might put
    ["Y15G", "A1C_Y15P"] in the holdout set, to test whether the model can predict the effect
    of mutations at locus 15.

    Arguments:
    * seq_ids: list of seq_ids to split
    * min_held_out_mutants: minimum number of mutants to include in the holdout set
    * max_held_out_mutants: maximum number of mutants to include in the holdout set

    Returns: (list of sets of seq_ids, list of viable loci)
    """

    # Our approach is to choose loci to hold out that match the requirements.

    # Number of mutants with a mutation at each locus.
    loci_mutant_counts = defaultdict(int)
    for seq_id in seq_ids:
        if seq_id == "":
            continue
        for allele in seq_id.split("_"):
            locus = get_locus_from_allele_id(allele)
            loci_mutant_counts[locus] += 1

    # We want to hold out at least one mutant at each locus.
    viable_loci = []
    for locus, count in loci_mutant_counts.items():
        if count < min_held_out_mutants:
            continue
        if max_held_out_mutants is not None and count > max_held_out_mutants:
            continue
        viable_loci.append(locus)

    holdout_lists = []
    for holdout_locus in viable_loci:
        holdout_list = []
        for seq_id in seq_ids:
            if seq_id == "":
                # WT is never in the holdout set.
                continue
            if any(
                get_locus_from_allele_id(allele) == holdout_locus
                for allele in seq_id.split("_")
            ):
                holdout_list.append(seq_id)
        holdout_lists.append(holdout_list)
    return holdout_lists, viable_loci


def get_cv_splits(seq_id_list, holdout_sets, n_splits=5):
    """Returns a list of (train_mutants, test_mutants) tuples for cross_val_score."""
    splits = []
    for holdout_set in holdout_sets:
        test_indices = []
        train_indices = []
        for ii, seq_id in enumerate(seq_id_list):
            if seq_id in holdout_set:
                test_indices.append(ii)
            else:
                train_indices.append(ii)
        splits.append((train_indices, test_indices))
    return splits
