"""
Utilities for building DNA constructs and posting designs to Teselagen.

This module contains the core business logic for:
1. Analyzing genbank templates to determine mutations
2. Building multi-site mutant constructs
3. Posting designs to Teselagen API
"""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from app.helpers.sequence_util import (
    allele_set_to_seq_id,
    get_locus_from_allele_id,
    maybe_get_allele_id_error_message,
    sort_seq_id_list_no_verification,
)
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


def calculate_mutations_from_genbank(wt_aa_sequence: str, genbank_content: str) -> str:
    """
    Extract the single CDS sequence from a genbank file and calculate mutations relative to WT.

    Parameters
    ----------
    wt_aa_sequence : str
        The wild-type amino acid sequence
    genbank_content : str
        Content of the genbank file

    Returns
    -------
    str
        The seq_id representing mutations (e.g., "WT", "D104G", "D104G_G429R")

    Raises
    ------
    ValueError
        If genbank file issues are found (no CDS, multiple CDS, wrong length, etc.)
    """
    try:
        # Write genbank content to temporary file for BioPython
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as f:
            f.write(genbank_content)
            temp_file = f.name

        try:
            record = SeqIO.read(temp_file, "genbank")
        finally:
            Path(temp_file).unlink()  # Clean up temp file

    except Exception as e:
        raise ValueError(f"Could not read genbank content: {e}")

    # Find CDS features
    cds_features = [f for f in record.features if f.type == "CDS"]
    if len(cds_features) == 0:
        raise ValueError(f"No CDS feature found in genbank")
    if len(cds_features) > 1:
        raise ValueError(f"Multiple CDS features found in genbank, expected exactly one")

    cds_feature = cds_features[0]

    # Extract and translate CDS
    try:
        cds_nt = cds_feature.extract(record.seq)
        cds_aa = str(cds_nt.translate())
    except Exception as e:
        raise ValueError(f"Could not extract/translate CDS: {e}")

    # Remove potential stop codon for comparison
    if cds_aa.endswith("*"):
        cds_aa = cds_aa[:-1]

    wt_clean = wt_aa_sequence.strip().replace("*", "")

    if len(cds_aa) != len(wt_clean):
        raise ValueError(
            f"Length mismatch: CDS AA length {len(cds_aa)} vs WT length {len(wt_clean)}"
        )

    # Find mutations using sequence_util format
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_clean, cds_aa)):
        if wt_aa != mut_aa:
            allele_id = f"{wt_aa}{i+1}{mut_aa}"
            # Validate using existing utility
            error = maybe_get_allele_id_error_message(wt_clean, allele_id)
            if error:
                raise ValueError(f"Invalid mutation detected: {error}")
            mutations.append(allele_id)

    # Use existing utility to create seq_id
    return allele_set_to_seq_id(set(mutations))


def calculate_mutations_from_genbank_file(wt_aa_sequence: str, genbank_file_path: str) -> str:
    """
    Extract the single CDS sequence from a genbank file and calculate mutations relative to WT.

    Parameters
    ----------
    wt_aa_sequence : str
        The wild-type amino acid sequence
    genbank_file_path : str
        Path to the genbank file

    Returns
    -------
    str
        The seq_id representing mutations (e.g., "WT", "D104G", "D104G_G429R")

    Raises
    ------
    ValueError
        If genbank file issues are found (no CDS, multiple CDS, wrong length, etc.)
    """
    try:
        record = SeqIO.read(genbank_file_path, "genbank")
    except Exception as e:
        raise ValueError(f"Could not read genbank file {genbank_file_path}: {e}")

    # Find CDS features
    cds_features = [f for f in record.features if f.type == "CDS"]
    if len(cds_features) == 0:
        raise ValueError(f"No CDS feature found in genbank file {genbank_file_path}")
    if len(cds_features) > 1:
        raise ValueError(
            f"Multiple CDS features found in genbank file {genbank_file_path}, expected exactly one"
        )

    cds_feature = cds_features[0]

    # Extract and translate CDS
    try:
        cds_nt = cds_feature.extract(record.seq)
        cds_aa = str(cds_nt.translate())
    except Exception as e:
        raise ValueError(f"Could not extract/translate CDS from {genbank_file_path}: {e}")

    # Remove potential stop codon for comparison
    if cds_aa.endswith("*"):
        cds_aa = cds_aa[:-1]

    wt_clean = wt_aa_sequence.strip().replace("*", "")

    if len(cds_aa) != len(wt_clean):
        raise ValueError(
            f"Length mismatch in {genbank_file_path}: CDS AA length {len(cds_aa)} vs WT length {len(wt_clean)}"
        )

    # Find mutations using sequence_util format
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_clean, cds_aa)):
        if wt_aa != mut_aa:
            allele_id = f"{wt_aa}{i+1}{mut_aa}"
            # Validate using existing utility
            error = maybe_get_allele_id_error_message(wt_clean, allele_id)
            if error:
                raise ValueError(f"Invalid mutation detected in {genbank_file_path}: {error}")
            mutations.append(allele_id)

    # Use existing utility to create seq_id
    return allele_set_to_seq_id(set(mutations))


def build_genbank_template_map(
    genbank_file_paths: Dict[str, str], wt_aa_sequence: str
) -> Dict[str, str]:
    """
    Build a mapping from seq_id to genbank file name for all valid genbanks.

    Parameters
    ----------
    genbank_file_paths : Dict[str, str]
        Dictionary mapping filename -> genbank file path
    wt_aa_sequence : str
        The wild-type amino acid sequence

    Returns
    -------
    Dict[str, str]
        Dictionary mapping seq_id -> genbank filename
    """
    template_map = {}

    for filename, file_path in genbank_file_paths.items():
        try:
            seq_id = calculate_mutations_from_genbank_file(wt_aa_sequence, file_path)
            template_map[seq_id] = filename
            logging.info(f"Added template: {seq_id} -> {filename}")
        except ValueError as e:
            logging.warning(f"Skipping {filename}: {e}")
            continue

    return template_map


def count_mutations_in_seq_id(seq_id: str) -> int:
    """Count the number of mutations in a seq_id."""
    if seq_id == "WT":
        return 0
    return len(seq_id.split("_"))


def get_seq_id_to_build_tuples(
    seq_ids: List[str], template_map: Dict[str, str], distance: int = 1
) -> Dict[str, Tuple[str, List[str]]]:
    """
    Map each target seq_id to (base_seq_id, new_alleles) for building.

    Parameters
    ----------
    seq_ids : List[str]
        List of target seq_ids to build
    template_map : Dict[str, str]
        Dictionary mapping seq_id -> genbank filename
    distance : int
        Number of mutations to add (distance from base template)

    Returns
    -------
    Dict[str, Tuple[str, List[str]]]
        Dictionary mapping seq_id -> (base_seq_id, new_alleles_list)
    """
    base_seq_ids = list(template_map.keys())

    possible_base_frequency = defaultdict(int)
    for new_seq_id in seq_ids:
        new_seq_id_allele_set = set(new_seq_id.split("_"))
        target_mutation_count = len(new_seq_id_allele_set)

        # Look for bases that are exactly 'distance' mutations away
        for possible_base in base_seq_ids:
            base_mutation_count = count_mutations_in_seq_id(possible_base)

            # Check if this base is the right distance away
            if base_mutation_count + distance == target_mutation_count:
                possible_base_allele_set = (
                    set(possible_base.split("_")) if possible_base != "WT" else set()
                )

                # Check if all base mutations are present in the target
                if possible_base_allele_set.issubset(new_seq_id_allele_set):
                    required_new_alleles = new_seq_id_allele_set - possible_base_allele_set
                    if len(required_new_alleles) == distance:
                        possible_base_frequency[possible_base] += 1

    seq_id_to_build_tuple = {}
    for new_seq_id in seq_ids:
        new_seq_id_allele_set = set(new_seq_id.split("_"))
        target_mutation_count = len(new_seq_id_allele_set)

        # Find the best base (most frequently usable)
        best_bases = []
        for possible_base, frequency in sorted(
            possible_base_frequency.items(), key=lambda x: x[1], reverse=True
        ):
            base_mutation_count = count_mutations_in_seq_id(possible_base)

            if base_mutation_count + distance == target_mutation_count:
                possible_base_allele_set = (
                    set(possible_base.split("_")) if possible_base != "WT" else set()
                )

                if possible_base_allele_set.issubset(new_seq_id_allele_set):
                    required_new_alleles = new_seq_id_allele_set - possible_base_allele_set
                    if len(required_new_alleles) == distance:
                        best_bases.append((possible_base, list(required_new_alleles), frequency))

        if best_bases:
            # Choose the most frequently usable base
            base_seq_id, new_alleles, _ = best_bases[0]
            # Sort new alleles by position using existing utility
            sorted_alleles = sorted(new_alleles, key=lambda x: get_locus_from_allele_id(x))
            seq_id_to_build_tuple[new_seq_id] = (base_seq_id, sorted_alleles)
        else:
            logging.error(
                f"No base sequence found for {new_seq_id} at distance {distance}. Skipping."
            )

    return seq_id_to_build_tuple


class TeselagenDesignPart:
    """A part in a Teselagen construct."""

    def __init__(
        self,
        nucleic_acid_seq: Optional[str] = None,
        gb_tuple: Optional[Tuple[str, int, int]] = None,
    ):
        if (nucleic_acid_seq is None) == (gb_tuple is None):
            raise ValueError("Provide *either* nucleic_acid_seq or gb_tuple, not both.")
        self.nucleic_acid_seq = nucleic_acid_seq
        self.gb_tuple = gb_tuple  # (filename, start, stop)

    @property
    def _key(self) -> Tuple:
        if self.nucleic_acid_seq is not None:
            return ("seq", self.nucleic_acid_seq)
        assert self.gb_tuple is not None
        filename, start, stop = self.gb_tuple
        return ("gb", filename, start, stop)

    def get_length(self, genbank_files: Dict[str, str]) -> int:
        """Get the length of this part."""
        if self.nucleic_acid_seq:
            return len(self.nucleic_acid_seq)
        else:
            assert self.gb_tuple is not None
            filename, start, stop = self.gb_tuple
            return stop - start

    def to_dict(self) -> Dict[str, Any]:
        if self.nucleic_acid_seq:
            return {"type": "peptide", "sequence": self.nucleic_acid_seq}
        assert self.gb_tuple is not None
        filename, start, stop = self.gb_tuple
        return {"type": "reference", "path": filename, "start": start, "stop": stop}


class TeselagenDesignConstruct:
    """Container for parts belonging to one mutant design."""

    def __init__(self, name: str):
        self.name = name
        self.parts: List[TeselagenDesignPart] = []

    def add_part(self, part: TeselagenDesignPart):
        self.parts.append(part)

    def get_total_length(self, genbank_files: Dict[str, str]) -> int:
        """Get total length of all parts in this construct."""
        return sum(part.get_length(genbank_files) for part in self.parts)

    def validate_structure(
        self, expected_mutations: int, expected_total_length: int, genbank_files: Dict[str, str]
    ):
        """Validate the construct structure."""
        expected_parts = 2 * expected_mutations + 1
        if len(self.parts) != expected_parts:
            raise ValueError(
                f"Expected {expected_parts} parts for {expected_mutations} mutations, got {len(self.parts)}"
            )

        if self.get_total_length(genbank_files) != expected_total_length:
            raise ValueError(
                f"Construct length {self.get_total_length(genbank_files)} doesn't match expected length {expected_total_length}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "parts": [p.to_dict() for p in self.parts]}


class TeselagenDesignBuilder:
    """Accumulates Teselagen constructs and can mutate targets, then export."""

    def __init__(self, design_name: str):
        self.constructs: List[TeselagenDesignConstruct] = []
        self.design_name = design_name

    def build_mutant(
        self, base_file_path: str, new_seq_id: str, distance: int, wt_aa_sequence: str
    ) -> TeselagenDesignConstruct:
        """Create a mutant construct and store it internally."""

        if not Path(base_file_path).exists():
            raise ValueError(f"Base genbank file {base_file_path} not found")

        # Get the seq_id of the starting genbank
        base_seq_id = calculate_mutations_from_genbank_file(wt_aa_sequence, base_file_path)

        # Verify distance
        base_mutations = set(base_seq_id.split("_")) if base_seq_id != "WT" else set()
        new_mutations = set(new_seq_id.split("_"))

        if not base_mutations.issubset(new_mutations):
            raise ValueError(
                f"Base mutations {base_mutations} not subset of new mutations {new_mutations}"
            )

        required_new_mutations = new_mutations - base_mutations
        if len(required_new_mutations) != distance:
            raise ValueError(
                f"Expected distance {distance}, but found {len(required_new_mutations)} new mutations"
            )

        # Parse and sort the new mutations by position
        new_alleles = []
        for allele in required_new_mutations:
            error = maybe_get_allele_id_error_message(wt_aa_sequence, allele)
            if error:
                raise ValueError(f"Invalid allele {allele}: {error}")

            pos = get_locus_from_allele_id(allele)
            wt_ltr = allele[0]
            mut_ltr = allele[-1]
            new_alleles.append((pos, wt_ltr, mut_ltr, allele))

        # Sort by position
        new_alleles.sort(key=lambda x: x[0])

        # Read the starting genbank
        record = SeqIO.read(base_file_path, "genbank")

        cds_feature = next(f for f in record.features if f.type == "CDS")
        cds_nt = cds_feature.extract(record.seq)
        cds_aa = cds_nt.translate()

        # Validate all mutations
        for pos, wt_ltr, mut_ltr, allele in new_alleles:
            if cds_aa[pos - 1] != wt_ltr:
                raise ValueError(
                    f"Expected {wt_ltr} at AA pos {pos} in {base_file_path}, found {cds_aa[pos-1]}"
                )

        # Build the construct with alternating segments and mutations
        construct_name = f"{Path(base_file_path).stem}_{new_seq_id}"
        construct = TeselagenDesignConstruct(construct_name)

        cds_start = int(cds_feature.location.start)
        last_end = 0

        for pos, wt_ltr, mut_ltr, allele in new_alleles:
            codon_start = cds_start + (pos - 1) * 3
            codon_end = codon_start + 3

            # Add segment before this mutation (if any)
            if last_end < codon_start:
                construct.add_part(
                    TeselagenDesignPart(gb_tuple=(Path(base_file_path).name, last_end, codon_start))
                )

            # Add the mutant codon
            current_codon = str(cds_nt[(pos - 1) * 3 : pos * 3])
            new_codon = self._choose_codon(current_codon, mut_ltr, cds_nt, cds_aa)
            construct.add_part(TeselagenDesignPart(nucleic_acid_seq=new_codon))

            last_end = codon_end

        # Add final segment (if any)
        if last_end < len(record):
            construct.add_part(
                TeselagenDesignPart(gb_tuple=(Path(base_file_path).name, last_end, len(record)))
            )

        # Validate construct length
        expected_parts = 2 * distance + 1
        if len(construct.parts) != expected_parts:
            raise ValueError(
                f"Expected {expected_parts} parts for {distance} mutations, got {len(construct.parts)}"
            )

        self.constructs.append(construct)
        return construct

    def _choose_codon(self, current_codon: str, aa_letter: str, cds_nt: Seq, cds_aa: Seq) -> str:
        aa_codon_counts = {}
        for aa_idx, aa in enumerate(cds_aa):
            if aa == aa_letter:
                codon = cds_nt[aa_idx * 3 : (aa_idx + 1) * 3]
                aa_codon_counts[codon] = aa_codon_counts.get(codon, 0) + 1
        if len(aa_codon_counts) == 0:
            raise ValueError(f"No codons found for {aa_letter} in existing gene")

        codon_counts_sorted = sorted(aa_codon_counts.items(), key=lambda x: x[1], reverse=True)
        most_common_codon = codon_counts_sorted[0][0]
        if most_common_codon.translate() != aa_letter:
            raise ValueError(
                f"Bug! Most common codon for {aa_letter} translates to {most_common_codon.translate()}"
            )
        return str(most_common_codon)

    def to_teselagen_json(
        self,
        genbank_file_paths: Dict[str, str],
        assembly_method: str = "golden gate",
        allow_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """Produce Teselagen‑compatible JSON including *all* annotated GenBank parts."""

        part_key_to_id: Dict[Tuple, str] = {}
        gb_subsets: Dict[str, Set[Tuple[int, int]]] = {}
        nucleic_parts: Dict[str, str] = {}

        def _sha_id(txt: str) -> str:
            return f"p_{txt}"

        # aggregate parts from constructs
        for cons in self.constructs:
            for part in cons.parts:
                key = part._key
                if key in part_key_to_id:
                    continue
                if key[0] == "gb":
                    _, filename, s, e = key
                    pid = f"{Path(filename).stem}-{s}-{e}"
                    part_key_to_id[key] = pid
                    gb_subsets.setdefault(filename, set()).add((s, e))
                else:  # synthetic sequence
                    _, seq = key
                    pid = _sha_id(seq)
                    part_key_to_id[key] = pid
                    nucleic_parts[pid] = seq

        # sequences array
        sequences_json = []
        for filename, subset_set in gb_subsets.items():
            if filename not in genbank_file_paths:
                raise ValueError(f"Genbank file {filename} referenced but not provided")

            file_path = genbank_file_paths[filename]

            # Parse genbank file
            rec = SeqIO.read(file_path, "genbank")

            seq_name = Path(filename).stem
            parts_json = []

            # add *all* annotated features as parts
            for feat in rec.features:
                if feat.type == "source":
                    continue
                start = int(feat.location.start)
                end = int(feat.location.end) - 1  # Inclusive for Teselagen
                fid = f"{seq_name}_{start}_{end}_{feat.type}"
                fname = feat.qualifiers.get("label", [feat.type])[0]
                parts_json.append(
                    {
                        "start": start,
                        "end": end,
                        "id": fid,
                        "name": fname,
                        "strand": 1 if feat.location.strand != -1 else -1,
                    }
                )

            # add subset slices actually referenced by constructs
            for s, e in sorted(subset_set):
                pid = part_key_to_id[("gb", filename, s, e)]
                parts_json.append(
                    {
                        "start": s,
                        "end": e - 1,
                        "id": pid,
                        "name": pid,
                        "strand": 1,
                    }
                )

            sequences_json.append(
                {
                    "name": seq_name,
                    "sequence": str(rec.seq),
                    "parts": parts_json,
                    "circular": True,
                }
            )

        # synthetic sequences
        for pid, seq in nucleic_parts.items():
            sequences_json.append(
                {
                    "name": pid,
                    "sequence": seq,
                    "parts": [
                        {"start": 0, "end": len(seq) - 1, "id": pid, "name": pid, "strand": 1}
                    ],
                }
            )

        # Build columns (one per construct)
        columns_json = []
        max_parts = (
            max(len(construct.parts) for construct in self.constructs) if self.constructs else 0
        )
        for column_idx in range(max_parts):
            col_parts = []
            for construct in self.constructs:
                if column_idx < len(construct.parts):
                    part = construct.parts[column_idx]
                    pid = part_key_to_id[part._key]
                    col_parts.append({"id": pid})
                else:
                    col_parts.append({"id": ""})
            columns_json.append(
                {
                    "direction": "forward",
                    "icon": "cds",
                    "name": f"Part {column_idx + 1}",
                    "parts": col_parts,
                }
            )

        # Final JSON structure
        design_json = {
            "assembly_method": assembly_method,
            "columns": columns_json,
            "layout_type": "list",
            "name": self.design_name,
            "sequences": sequences_json,
        }

        return {
            "allowDuplicates": allow_duplicates,
            "designJson": design_json,
        }


def post_design_to_teselagen(
    design_json: Dict[str, Any],
    username: str,
    otp: str,
    project_id: str,
    teselagen_base_url: str = "https://jbei.teselagen.com",
) -> Dict[str, str]:
    """
    Post a design to the Teselagen API.

    Parameters
    ----------
    design_json : Dict[str, Any]
        The design JSON to post
    username : str
        Teselagen username
    otp : str
        One-time password
    project_id : str
        Teselagen project ID
    teselagen_base_url : str, optional
        Base URL for Teselagen instance (default: https://jbei.teselagen.com)

    Returns
    -------
    Dict[str, str]
        Response with design ID: {'id': 'design-uuid'}

    Raises
    ------
    ValueError
        If the API call fails
    """
    BASE_URL = f"{teselagen_base_url}/tg-api"

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    # Authenticate and get the token
    try:
        response = session.put(
            url=f"{BASE_URL}/public/auth",
            json={
                "username": username,
                "password": otp,
                "expiresIn": "1d",
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Authentication failed: {e}")

    session.headers.update(
        {"x-tg-api-token": response.json()["token"], "tg-project-id": project_id}
    )
    session.headers.pop("Content-Type", None)

    # Post the design
    try:
        url = f"{BASE_URL}/designs"
        response = session.post(url, json=design_json, timeout=9)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Design post failed: {e}")


def create_dna_build(
    design_id: str,
    genbank_files: Dict[str, str],  # filename -> content
    wt_aa_sequence: str,
    seq_ids: List[str],
    number_of_mutations: int,
    username: Optional[str] = None,
    otp: Optional[str] = None,
    project_id: Optional[str] = None,
    dry_run: bool = True,
    teselagen_base_url: Optional[str] = None,
) -> Tuple[str, Optional[str], Dict[str, Dict[str, Any]]]:
    """
    Create a DNA build design and optionally post to Teselagen.

    Parameters
    ----------
    design_id : str
        Name for the design
    genbank_files : Dict[str, str]
        Dictionary mapping filename -> genbank file content
    wt_aa_sequence : str
        Wild-type amino acid sequence
    seq_ids : List[str]
        List of target sequence IDs to build
    number_of_mutations : int
        Number of mutations to add per construct
    username : str, optional
        Teselagen username (required if not dry_run)
    otp : str, optional
        Teselagen OTP (required if not dry_run)
    project_id : str, optional
        Teselagen project ID (required if not dry_run)
    dry_run : bool
        If True, only validate and build design without posting

    Returns
    -------
    Tuple[str, Optional[str], Dict[str, Dict[str, Any]]]
        - design_name: str
        - teselagen_id: Optional[str]
        - seq_id_to_build_results: Dict mapping seq_id to:
            - success: bool
            - error_msg: Optional[str]
            - template_used: Optional[str] (genbank filename)
            - teselagen_seq_id: Optional[str]

    Raises
    ------
    ValueError
        If validation fails or API call fails
    """
    if not genbank_files:
        raise ValueError("At least one genbank file must be provided")

    if not seq_ids:
        raise ValueError("At least one sequence ID must be provided")

    if not dry_run and (not username or not otp or not project_id):
        raise ValueError("Username, OTP, and project_id required when not in dry_run mode")

    # Create temp directory and write GenBank files to disk
    with tempfile.TemporaryDirectory() as temp_dir:
        genbank_file_paths = {}
        for filename, content in genbank_files.items():
            try:
                # Validate GenBank content by trying to parse it
                with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as f:
                    f.write(content)
                    temp_file = f.name
                try:
                    SeqIO.read(temp_file, "genbank")
                finally:
                    Path(temp_file).unlink()
            except Exception as e:
                raise ValueError(f"Invalid GenBank file {filename}: {e}")

            # Write to temp directory
            file_path = Path(temp_dir) / filename
            with open(file_path, "w") as f:
                f.write(content)
            genbank_file_paths[filename] = str(file_path)

        # Build template map using file paths
        template_map = build_genbank_template_map(genbank_file_paths, wt_aa_sequence)

        if not template_map:
            raise ValueError("No valid genbank templates found")

        logging.info(
            f"Built template map with {len(template_map)} templates: {list(template_map.keys())}"
        )

        # Get build tuples
        build_tuples = get_seq_id_to_build_tuples(seq_ids, template_map, number_of_mutations)

        if not build_tuples:
            raise ValueError(
                f"No buildable constructs found for the given seq_ids at distance {number_of_mutations}"
            )

        # Build constructs
        builder = TeselagenDesignBuilder(design_id)
        seq_id_to_build_results: Dict[str, Dict[str, Any]] = {}

        seq_id_to_teselagen_id = {}

        ii = 0
        for seq_id in sort_seq_id_list_no_verification(seq_ids):
            if seq_id in build_tuples:
                base_seq_id, new_alleles = build_tuples[seq_id]
                base_filename = template_map[base_seq_id]
                base_filepath = genbank_file_paths[base_filename]

                try:
                    builder.build_mutant(base_filepath, seq_id, number_of_mutations, wt_aa_sequence)
                    teselagen_seq_id = f"{design_id}_{ii+1:04}"
                    seq_id_to_teselagen_id[seq_id] = teselagen_seq_id
                    seq_id_to_build_results[seq_id] = {
                        "success": True,
                        "error_msg": None,
                        "template_used": base_filename,
                        "teselagen_seq_id": teselagen_seq_id,
                    }
                    ii += 1
                except Exception as e:
                    seq_id_to_build_results[seq_id] = {
                        "success": False,
                        "error_msg": str(e),
                        "template_used": base_filename,
                        "teselagen_seq_id": None,
                    }
                    logging.error(f"Failed to build {seq_id}: {e}")
            else:
                seq_id_to_build_results[seq_id] = {
                    "success": False,
                    "error_msg": f"No build path found at distance {number_of_mutations}",
                    "template_used": None,
                    "teselagen_seq_id": None,
                }

        # Check if any constructs were successfully built
        successful_builds = [
            seq_id for seq_id, result in seq_id_to_build_results.items() if result["success"]
        ]
        if not successful_builds:
            failed_errors = [
                result["error_msg"]
                for result in seq_id_to_build_results.values()
                if not result["success"]
            ]
            raise ValueError(f"No constructs could be built. Errors: {failed_errors}")

        # Generate design JSON
        design_json = builder.to_teselagen_json(genbank_file_paths)

        # Post to Teselagen if not dry run
        teselagen_id = None
        if not dry_run:
            assert username is not None and otp is not None and project_id is not None
            # Use provided Teselagen URL or default
            base_url = teselagen_base_url or "https://jbei.teselagen.com"
            teselagen_response = post_design_to_teselagen(
                design_json, username, otp, project_id, base_url
            )
            teselagen_id = teselagen_response["id"]

        return design_id, teselagen_id, seq_id_to_build_results
