import re

VALID_AMINO_ACIDS = "ACDEFGHIKLMNOPQRSTUVWY"


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

    def maybe_get_allele_id_error_message(allele_id):
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

    def assert_valid_seq_id(seq_id):
        if seq_id == "":
            return
        for allele_id in seq_id.split("_"):
            allele_error_msg = maybe_get_allele_id_error_message(allele_id)
            if allele_error_msg:
                raise ValueError(f"Invalid seq_id {seq_id}: {allele_error_msg}")

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
                    allele_is_at_locus(allele, aa_idx + 1)
                    for allele in starting_seq_allele_list
                ]
            ):
                # The case where this locus is already mutated.

                seq_base_allele_list = [
                    allele
                    for allele in starting_seq_allele_list
                    if not allele_is_at_locus(allele, aa_idx + 1)
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
