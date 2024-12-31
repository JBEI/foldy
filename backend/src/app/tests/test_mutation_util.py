# backend/src/app/tests/test_mutation_util.py

import pytest
from src.app.helpers.mutation_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
)

VALID_AMINO_ACIDS = "ACDEFGHIKLMNOPQRSTUVWY"


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
    starting_seq_ids = [""]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == 21 * 3 + 1
    ), "Expected one mutation per residue per alternative amino acid, plus WT."


def test_one_mutant_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = ["A1C"]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == 21 * 3 + 1
    ), "Expected one mutation per residue per amino acid."


def test_two_starting_seq_ids(wt_seq):
    """Test the function with empty starting sequence IDs."""
    starting_seq_ids = ["", "A1C"]
    result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids, [])
    assert (
        len(result) == (21 * 3 + 1) + (21 * 3 + 1) - 22
    ), "Expected the addition of two above tests, minus 22 for overlap (all single A1 mutations)."


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
    starting_seq_ids = [""]
    extra_seq_ids = ["A1T_D3T"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert len(result) == 21 * 3 + 2


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


def test_extra_seq_ids_canonicalizes_ids(wt_seq):
    """Test that extra_seq_ids get canonicalized."""
    starting_seq_ids = []
    extra_seq_ids = ["D3T_A1T"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert result == ["A1T_D3T"]


def test_bad_starting_seq_ids_causes_error(wt_seq):
    """Test that extra_seq_ids get canonicalized."""
    starting_seq_ids = []
    extra_seq_ids = ["D3T_A1T"]
    result = get_seq_ids_for_deep_mutational_scan(
        wt_seq, starting_seq_ids, extra_seq_ids
    )
    assert result == ["A1T_D3T"]


def test_seq_id_to_seq_one_mut():
    assert seq_id_to_seq("ACTY", "A1C") == "CCTY"


def test_seq_id_to_seq_one_two_mut():
    assert seq_id_to_seq("ACTY", "T3A") == "ACAY"


def test_seq_id_to_seq_one_end_mut():
    assert seq_id_to_seq("ACTY", "Y4A") == "ACTA"


# def test_single_mutation(wt_seq):
#     """Test the function with a single starting mutation."""
#     wt_seq = wt_seq  # "A", "C", "D"
#     starting_seq_ids = ["A1T"]

#     # Expected mutations:
#     # Starting mutation: "A1T"
#     # Possible mutations at positions 2 and 3
#     expected = set(
#         [
#             "A1T",
#             "A1T_C2A",
#             "A1T_C2C",
#             "A1T_C2D",
#             "A1T_C2E",
#             "A1T_C2F",
#             "A1T_C2G",
#             "A1T_C2H",
#             "A1T_C2I",
#             "A1T_C2K",
#             "A1T_C2L",
#             "A1T_C2M",
#             "A1T_C2N",
#             "A1T_C2P",
#             "A1T_C2Q",
#             "A1T_C2R",
#             "A1T_C2S",
#             "A1T_C2V",
#             "A1T_C2W",
#             "A1T_C2Y",
#             "A1T_D3A",
#             "A1T_D3C",
#             "A1T_D3D",
#             "A1T_D3E",
#             "A1T_D3F",
#             "A1T_D3G",
#             "A1T_D3H",
#             "A1T_D3I",
#             "A1T_D3K",
#             "A1T_D3L",
#             "A1T_D3M",
#             "A1T_D3N",
#             "A1T_D3P",
#             "A1T_D3Q",
#             "A1T_D3R",
#             "A1T_D3S",
#             "A1T_D3V",
#             "A1T_D3W",
#             "A1T_D3Y",
#         ]
#     )

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))
#     assert result == expected, f"Expected mutations {expected}, but got {result}"


# def test_multiple_starting_mutations(wt_seq):
#     """Test the function with multiple starting mutations."""
#     wt_seq = wt_seq  # "A", "C", "D"
#     starting_seq_ids = ["A1T", "C2G"]

#     # Expected mutations:
#     # Starting mutations: "A1T", "C2G"
#     # Combinations: "A1T", "C2G", "A1T_C2G"
#     # Plus mutations at other positions
#     expected = set(
#         [
#             "A1T",
#             "C2G",
#             "A1T_C2G",
#             "A1T_C2G_D3A",
#             "A1T_C2G_D3C",
#             "A1T_C2G_D3D",
#             "A1T_C2G_D3E",
#             "A1T_C2G_D3F",
#             "A1T_C2G_D3G",
#             "A1T_C2G_D3H",
#             "A1T_C2G_D3I",
#             "A1T_C2G_D3K",
#             "A1T_C2G_D3L",
#             "A1T_C2G_D3M",
#             "A1T_C2G_D3N",
#             "A1T_C2G_D3P",
#             "A1T_C2G_D3Q",
#             "A1T_C2G_D3R",
#             "A1T_C2G_D3S",
#             "A1T_C2G_D3V",
#             "A1T_C2G_D3W",
#             "A1T_C2G_D3Y",
#             "A1T_D3A",
#             "A1T_D3C",
#             "A1T_D3D",
#             "A1T_D3E",
#             "A1T_D3F",
#             "A1T_D3G",
#             "A1T_D3H",
#             "A1T_D3I",
#             "A1T_D3K",
#             "A1T_D3L",
#             "A1T_D3M",
#             "A1T_D3N",
#             "A1T_D3P",
#             "A1T_D3Q",
#             "A1T_D3R",
#             "A1T_D3S",
#             "A1T_D3V",
#             "A1T_D3W",
#             "A1T_D3Y",
#             "C2G_D3A",
#             "C2G_D3C",
#             "C2G_D3D",
#             "C2G_D3E",
#             "C2G_D3F",
#             "C2G_D3G",
#             "C2G_D3H",
#             "C2G_D3I",
#             "C2G_D3K",
#             "C2G_D3L",
#             "C2G_D3M",
#             "C2G_D3N",
#             "C2G_D3P",
#             "C2G_D3Q",
#             "C2G_D3R",
#             "C2G_D3S",
#             "C2G_D3V",
#             "C2G_D3W",
#             "C2G_D3Y",
#         ]
#     )

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))
#     assert result == expected, f"Expected mutations {expected}, but got {result}"


# def test_invalid_mutation_format(wt_seq):
#     """Test the function with invalid mutation formats."""
#     #     starting_seq_ids = ["A1T", "InvalidMutation", "C2G_"]

#     with pytest.raises(ValueError):
#         get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids)


# def test_duplicate_mutations(wt_seq):
#     """Test the function with duplicate starting mutations."""
#     #     starting_seq_ids = ["A1T", "A1T"]  # Duplicate mutations

#     # Expected mutations should only include "A1T" once
#     expected = set(
#         [
#             "A1T",
#             "A1T_C2A",
#             "A1T_C2C",
#             "A1T_C2D",
#             "A1T_C2E",
#             "A1T_C2F",
#             "A1T_C2G",
#             "A1T_C2H",
#             "A1T_C2I",
#             "A1T_C2K",
#             "A1T_C2L",
#             "A1T_C2M",
#             "A1T_C2N",
#             "A1T_C2P",
#             "A1T_C2Q",
#             "A1T_C2R",
#             "A1T_C2S",
#             "A1T_C2V",
#             "A1T_C2W",
#             "A1T_C2Y",
#             "A1T_D3A",
#             "A1T_D3C",
#             "A1T_D3D",
#             "A1T_D3E",
#             "A1T_D3F",
#             "A1T_D3G",
#             "A1T_D3H",
#             "A1T_D3I",
#             "A1T_D3K",
#             "A1T_D3L",
#             "A1T_D3M",
#             "A1T_D3N",
#             "A1T_D3P",
#             "A1T_D3Q",
#             "A1T_D3R",
#             "A1T_D3S",
#             "A1T_D3V",
#             "A1T_D3W",
#             "A1T_D3Y",
#         ]
#     )

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))
#     assert result == expected, f"Expected mutations {expected}, but got {result}"


# def test_overlapping_mutations(wt_seq):
#     """Test the function with overlapping mutations at the same locus."""
#     wt_seq = wt_seq  # "A", "C", "D"
#     starting_seq_ids = ["A1T", "A1G"]  # Both mutations at locus 1

#     # Expected mutations:
#     # Starting mutations: "A1T", "A1G"
#     # Possible mutations by replacing "A1T" with other AAs
#     # and "A1G" with other AAs
#     expected = set(
#         [
#             "A1T",
#             "A1G",
#             "C2A",
#             "C2C",
#             "C2D",
#             "C2E",
#             "C2F",
#             "C2G",
#             "C2H",
#             "C2I",
#             "C2K",
#             "C2L",
#             "C2M",
#             "C2N",
#             "C2P",
#             "C2Q",
#             "C2R",
#             "C2S",
#             "C2V",
#             "C2W",
#             "C2Y",
#             "D3A",
#             "D3C",
#             "D3D",
#             "D3E",
#             "D3F",
#             "D3G",
#             "D3H",
#             "D3I",
#             "D3K",
#             "D3L",
#             "D3M",
#             "D3N",
#             "D3P",
#             "D3Q",
#             "D3R",
#             "D3S",
#             "D3V",
#             "D3W",
#             "D3Y",
#             "A1T_C2A",
#             "A1T_C2C",
#             "A1T_C2D",
#             "A1T_C2E",
#             "A1T_C2F",
#             "A1T_C2G",
#             "A1T_C2H",
#             "A1T_C2I",
#             "A1T_C2K",
#             "A1T_C2L",
#             "A1T_C2M",
#             "A1T_C2N",
#             "A1T_C2P",
#             "A1T_C2Q",
#             "A1T_C2R",
#             "A1T_C2S",
#             "A1T_C2V",
#             "A1T_C2W",
#             "A1T_C2Y",
#             "A1G_C2A",
#             "A1G_C2C",
#             "A1G_C2D",
#             "A1G_C2E",
#             "A1G_C2F",
#             "A1G_C2G",
#             "A1G_C2H",
#             "A1G_C2I",
#             "A1G_C2K",
#             "A1G_C2L",
#             "A1G_C2M",
#             "A1G_C2N",
#             "A1G_C2P",
#             "A1G_C2Q",
#             "A1G_C2R",
#             "A1G_C2S",
#             "A1G_C2V",
#             "A1G_C2W",
#             "A1G_C2Y",
#         ]
#     )

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))
#     assert result == expected, f"Expected mutations {expected}, but got {result}"


# def test_sorted_mutations(wt_seq):
#     """Test that mutations in the sequence IDs are sorted by locus."""
#     wt_seq = wt_seq  # "A", "C", "D"
#     starting_seq_ids = ["Y3G_A1T"]  # Out of order mutations

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))

#     # All sequence IDs should have mutations sorted by locus
#     for seq_id in result:
#         mutations = seq_id.split("_")
#         positions = [int(mut[1:-1]) for mut in mutations]
#         assert positions == sorted(
#             positions
#         ), f"Mutations in {seq_id} are not sorted by position."


# def test_no_mutations_needed(wt_seq):
#     """Test the function with a mutation that doesn't change the amino acid."""
#     wt_seq = wt_seq  # "A", "C", "D"
#     starting_seq_ids = ["A1A"]  # Mutation to the same amino acid as WT

#     expected = set(
#         [
#             "A1A",
#             "C2A",
#             "C2C",
#             "C2D",
#             "C2E",
#             "C2F",
#             "C2G",
#             "C2H",
#             "C2I",
#             "C2K",
#             "C2L",
#             "C2M",
#             "C2N",
#             "C2P",
#             "C2Q",
#             "C2R",
#             "C2S",
#             "C2V",
#             "C2W",
#             "C2Y",
#             "D3A",
#             "D3C",
#             "D3D",
#             "D3E",
#             "D3F",
#             "D3G",
#             "D3H",
#             "D3I",
#             "D3K",
#             "D3L",
#             "D3M",
#             "D3N",
#             "D3P",
#             "D3Q",
#             "D3R",
#             "D3S",
#             "D3V",
#             "D3W",
#             "D3Y",
#         ]
#     )

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))
#     assert "A1A" in result, "Expected 'A1A' mutation to be included."
#     # Additionally, verify other mutations are present
#     for mut in [
#         "C2A",
#         "C2C",
#         "C2D",
#         "C2E",
#         "C2F",
#         "C2G",
#         "C2H",
#         "C2I",
#         "C2K",
#         "C2L",
#         "C2M",
#         "C2N",
#         "C2P",
#         "C2Q",
#         "C2R",
#         "C2S",
#         "C2V",
#         "C2W",
#         "C2Y",
#         "D3A",
#         "D3C",
#         "D3D",
#         "D3E",
#         "D3F",
#         "D3G",
#         "D3H",
#         "D3I",
#         "D3K",
#         "D3L",
#         "D3M",
#         "D3N",
#         "D3P",
#         "D3Q",
#         "D3R",
#         "D3S",
#         "D3V",
#         "D3W",
#         "D3Y",
#     ]:
#         assert mut in result, f"Expected mutation {mut} not found in result."


# def test_full_sequence(wt_seq):
#     """Test the function with a starting sequence ID that includes multiple mutations."""
#     wt_seq = "ACD"  # Length 3
#     starting_seq_ids = ["A1T_Y3G"]

#     result = set(get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids))

#     # Check that starting mutation is present
#     assert "A1T_Y3G" in result, "Starting mutation 'A1T_Y3G' not found in result."

#     # Check that mutations are sorted
#     for seq_id in result:
#         mutations = seq_id.split("_")
#         positions = [int(mut[1:-1]) for mut in mutations]
#         assert positions == sorted(
#             positions
#         ), f"Mutations in {seq_id} are not sorted by position."

#     # Check that all mutations are valid
#     for seq_id in result:
#         mutations = seq_id.split("_")
#         for mut in mutations:
#             assert len(mut) == 4, f"Mutation {mut} does not have length 4."
#             orig, pos, new = mut[0], mut[1:-1], mut[-1]
#             assert (
#                 orig in VALID_AMINO_ACIDS
#             ), f"Original amino acid '{orig}' is invalid."
#             assert new in VALID_AMINO_ACIDS, f"New amino acid '{new}' is invalid."
#             pos_int = int(pos)
#             assert (
#                 1 <= pos_int <= len(wt_seq)
#             ), f"Position {pos_int} is out of range for WT sequence."


# @pytest.mark.parametrize(
#     "wt_seq,starting_seq_ids,expected_num",
#     [
#         (
#             "A",
#             ["A1T"],
#             len(
#                 set(
#                     [
#                         "A1T",
#                     ]
#                 ).union(set([f"A1T_C1{aa}" for aa in VALID_AMINO_ACIDS if aa != "A"]))
#             ),
#         ),
#         (
#             "AC",
#             ["A1T"],
#             len(
#                 set(
#                     [
#                         "A1T",
#                         "A1T_C2A",
#                         "A1T_C2C",
#                         "A1T_C2D",
#                         "A1T_C2E",
#                         "A1T_C2F",
#                         "A1T_C2G",
#                         "A1T_C2H",
#                         "A1T_C2I",
#                         "A1T_C2K",
#                         "A1T_C2L",
#                         "A1T_C2M",
#                         "A1T_C2N",
#                         "A1T_C2P",
#                         "A1T_C2Q",
#                         "A1T_C2R",
#                         "A1T_C2S",
#                         "A1T_C2V",
#                         "A1T_C2W",
#                         "A1T_C2Y",
#                     ]
#                 )
#             ),
#         ),
#         ("ACD", ["A1T_Y3G"], 1 + (len(VALID_AMINO_ACIDS) - 1) * 2 + 1),
#     ],
# )
# def test_parametrized_cases(wt_seq, starting_seq_ids, expected_num):
#     """Parameterized tests for different scenarios."""
#     result = get_seq_ids_for_deep_mutational_scan(wt_seq, starting_seq_ids)
#     assert (
#         len(result) == expected_num
#     ), f"Expected {expected_num} mutations, but got {len(result)}"
