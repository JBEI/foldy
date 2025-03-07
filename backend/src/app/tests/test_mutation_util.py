# backend/src/app/tests/test_mutation_util.py

import pytest
from app.helpers.sequence_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
    get_cross_validation_holdout_sets_with_stratification,
)
import pytest
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fixture to set random seeds before each test"""
    random.seed(42)
    np.random.seed(42)
    yield


@pytest.fixture
def wt_seq():
    return "ACD"  # Example wild-type sequence


def test_empty_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = []
    expected = []
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert result == expected, "Expected empty list when starting_seq_ids is empty."


def test_wt_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = ["WT"]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == 19 * 3 + 1
    ), "Expected one mutation per residue per alternative amino acid, plus WT."
    assert (
        "" not in result
    ), "There should not be an empty string in result, we use WT for wild type."
    assert "WT" in result, "Expected WT to be in the result."


def test_one_mutant_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = ["A1C"]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == 19 * 3 + 1
    ), "Expected one mutation per residue per amino acid."


def test_two_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = ["WT", "A1C"]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == (19 * 3 + 1) + (19 * 3 + 1) - 20
    ), "Expected the addition of two above tests, minus 20 for overlap (all single A1 mutations)."


def test_seq_ids_raises_error_on_bad_input(wt_seq):
    with pytest.raises(ValueError, match="improperly formatted"):
        get_seq_ids_for_deep_mutational_scan(wt_seq, ["3T"], [])


def test_seq_ids_raises_error_on_bad_input_2(wt_seq):
    with pytest.raises(ValueError, match="Allele does not correspond to WT sequence"):
        get_seq_ids_for_deep_mutational_scan(wt_seq, ["A3T"], [])


def test_seq_ids_raises_error_on_bad_input_3(wt_seq):
    with pytest.raises(ValueError, match="Allele is out of bounds"):
        get_seq_ids_for_deep_mutational_scan(wt_seq, ["A4T"], [])


def test_starting_seq_ids_works_with_extra_seq_ids(wt_seq):
    """Test that extra_seq_ids get added to starting_seq_ids."""
    starting_seq_ids = ["WT"]
    extra_seq_ids = ["A1T_D3T"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert len(result) == 19 * 3 + 2


def test_one_extra_seq_ids(wt_seq):
    """Test the function with just one extra."""
    starting_seq_ids = []
    extra_seq_ids = ["A1W"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert len(result) == 1, "Expected just one seq to come out, the one we put in."


def test_two_extra_seq_ids(wt_seq):
    """Test the function with two extras."""
    starting_seq_ids = []
    extra_seq_ids = ["A1W", "A1T_D3T"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert set(result) == {
        "A1W",
        "A1T_D3T",
    }, "Expected the two original seqs to pop out."


def test_extra_seq_ids_fails_if_not_canonical(wt_seq):
    """Test that extra_seq_ids get canonicalized."""
    starting_seq_ids = []
    extra_seq_ids = ["D3T_A1T"]
    with pytest.raises(ValueError, match="Loci are not sorted"):
        get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, extra_seq_ids)


def test_bad_starting_seq_ids_causes_error(wt_seq):
    """Test that extra_seq_ids get canonicalized."""
    starting_seq_ids = []
    extra_seq_ids = ["D3T_A1T"]
    with pytest.raises(ValueError, match="Loci are not sorted"):
        get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, extra_seq_ids)


def test_seq_id_to_seq_one_mut():
    assert seq_id_to_seq("ACTY", "A1C") == "CCTY"


def test_seq_id_to_seq_one_two_mut():
    assert seq_id_to_seq("ACTY", "T3A") == "ACAY"


def test_seq_id_to_seq_one_end_mut():
    assert seq_id_to_seq("ACTY", "Y4A") == "ACTA"


def test_get_cross_validation_holdout_sets_with_stratification():
    # My test case contains three that are necessary, and any one of the options from the first quartile.
    holdout_sets, holdout_loci = get_cross_validation_holdout_sets_with_stratification(
        ["A1C", "B2T", "C3A", "D4G", "E5Y", "F6K", "G7M", "H8S"],
        [1, 2, 3, 4, 1, 1, 1, 0.9],
        1,
        min_data_per_quartile=1,
        max_data_per_quartile=None,
        max_num_iterations=1000,
        max_loci_to_consider_each_round=50,
    )
    assert len(holdout_sets) == 1
    assert len(holdout_sets[0]) == 4
    # Any of four options could have been used to fill the zero-th quartile.
    assert "B2T" in holdout_sets[0]
    assert "C3A" in holdout_sets[0]
    assert "D4G" in holdout_sets[0]


def test_get_cross_validation_holdout_sets_with_stratification_2():
    # My test only contains one satisfactory option because two of the loci are coupled.
    holdout_sets, holdout_loci = get_cross_validation_holdout_sets_with_stratification(
        [
            "A1C",
            "A1C_B2T",
            "A1C_C3A",
            "A1C_D4G",
            "B2T_E5Y",
            "B2T_F6K",
            "B2T_G7M",
            "B2T_H8S",
        ],
        [1, 2, 3, 4, 1, 1, 1, 0.9],
        1,
        min_data_per_quartile=1,
        max_data_per_quartile=1,
        max_num_iterations=10,
        max_loci_to_consider_each_round=5,
    )
    assert len(holdout_sets) == 1
    assert len(holdout_sets[0]) == 4
    assert "A1C" in holdout_sets[0]
    assert "A1C_B2T" in holdout_sets[0]
    assert "A1C_C3A" in holdout_sets[0]
    assert "A1C_D4G" in holdout_sets[0]


def test_get_cross_validation_holdout_sets_with_stratification_3():
    # The only option is to take the first locus.
    holdout_sets, holdout_loci = get_cross_validation_holdout_sets_with_stratification(
        [
            "A1C",
            "A1G",
            "A1Y",
            "A1W",
            "B2T_E5Y",
            "B2T_F6K",
            "B2T_G7M",
            "B2T_H8S",
        ],
        [1, 2, 3, 4, 1, 1, 1, 0.9],
        1,
        min_data_per_quartile=1,
        max_data_per_quartile=1,
        max_num_iterations=10,
        max_loci_to_consider_each_round=5,
    )
    assert len(holdout_sets) == 1
    assert len(holdout_sets[0]) == 4
    assert holdout_loci == [{1}]
    assert "A1C" in holdout_sets[0]
    assert "A1G" in holdout_sets[0]
    assert "A1Y" in holdout_sets[0]
    assert "A1W" in holdout_sets[0]
