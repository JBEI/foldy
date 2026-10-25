"""
Data loading utilities for protein engineering prediction tasks.

This module provides functions for loading and processing protein engineering
datasets, including Deep Mutational Scanning (DMS) data from ProteinGym.
"""

import ast
import glob
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from app.helpers.sequence_util import (
    allele_set_to_seq_id,
    get_allele_set,
    get_loci_set,
    is_homolog_seq_id,
    maybe_get_allele_id_error_message,
    seq_id_to_seq,
    sort_seq_id_list,
)

logger = logging.getLogger(__name__)

# Constants for data locations
MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = MODULE_DIR / "data"
DMS_DIR = DATA_DIR / "DMS_ProteinGym_substitutions"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NATURALNESS_DIR = DATA_DIR / "naturalness"
DMS_METADATA_FILE = DATA_DIR / "DMS_substitutions.csv"
FLIP_AAV_DATA_FILE = DATA_DIR / "FLIP-AAV_multimutant_dataset.csv"


def maybe_modify_seq_id(dms_id: str, seq_id: str) -> str:
    """Modify the sequence ID if necessary."""
    if dms_id == "A0A140D2T1_ZIKV_Sourisseau_2019":
        m = re.match(r"^([A-Z])([0-9]+)([A-Z])$", seq_id)
        if not m:
            return seq_id
        return f"{m.group(1)}{int(m.group(2)) + 290}{m.group(3)}"
    return seq_id


def get_dms_metadata() -> pd.DataFrame:
    dms_metadata = pd.read_csv(DMS_METADATA_FILE)
    logger.info(f"Loaded metadata for {len(dms_metadata)} DMS datasets")

    dms_metadata = pd.concat(
        [
            dms_metadata,
            pd.DataFrame(
                {
                    "DMS_id": ["FLIP-AAV"],
                    "DMS_filename": [None],
                    "target_seq": [
                        "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    return dms_metadata


def get_available_proteingym_datasets(
    embedding_model_id: str, naturalness_model_id: str
) -> pd.DataFrame:
    """Get metadata for available datasets with specific embedding and naturalness models.

    Args:
        embedding_model_id: The specific embedding model ID to search for
        naturalness_model_id: The specific naturalness model ID to search for

    Returns:
        DataFrame containing metadata for datasets with the specified models
    """
    # Check if metadata file exists
    if not os.path.exists(DMS_METADATA_FILE):
        logger.error(f"DMS metadata file not found at {DMS_METADATA_FILE}")
        return pd.DataFrame()

    # Load metadata
    dms_metadata = get_dms_metadata()

    # Find datasets that have both the specified embedding and naturalness files
    available_datasets = []

    for known_dms_id in dms_metadata.DMS_id.tolist():
        # Check if both required files exist
        embedding_file = os.path.join(
            EMBEDDINGS_DIR, f"{known_dms_id}_embedding_{embedding_model_id}.csv"
        )
        naturalness_file = os.path.join(
            NATURALNESS_DIR, f"{known_dms_id}_naturalness_{naturalness_model_id}.csv"
        )

        if os.path.exists(embedding_file) and os.path.exists(naturalness_file):
            available_datasets.append(known_dms_id)

    # Filter metadata to only include datasets with both required files
    filtered_metadata = dms_metadata[dms_metadata["DMS_id"].isin(available_datasets)].copy()

    logger.info(
        f"Found {len(filtered_metadata)} datasets with embedding model '{embedding_model_id}' and naturalness model '{naturalness_model_id}'"
    )
    return filtered_metadata


def _parse_embedding_columns_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Parse embedding columns from JSON strings to numpy arrays in-place.

    Modifies embedding columns in-place to avoid memory duplication.

    Args:
        df: DataFrame with potential embedding columns as JSON strings

    Returns:
        The same DataFrame with parsed embeddings (modified in-place)
    """
    for col in df.columns:
        if col == "embedding" or col.startswith("embedding_layer_"):
            if len(df) > 0 and isinstance(df[col].iloc[0], str):
                # Parse in-place to avoid memory duplication
                for idx in df.index:
                    val = df.at[idx, col]
                    if isinstance(val, str):
                        df.at[idx, col] = np.array(json.loads(val))
    return df


def load_embedding_csv_in_chunks(path: str, chunk_rows: int = 20000) -> pd.DataFrame:
    """Read a large embedding CSV without ever holding every embedding string at once.

    ``pd.read_csv`` materializes the whole embedding column as Python strings before anything
    parses them. On a 12 GB embedding CSV that alone exceeds a 16 GB machine (measured peak RSS
    is ~2.2x file size), so the dense multi-mutation pools were unloadable. Reading in chunks
    and parsing each chunk immediately keeps peak at (parsed pool + one chunk).

    Produces exactly what the single-shot path produces: ``np.array(json.loads(...))``, i.e.
    float64, so results are unchanged. The ``seq`` column is skipped -- it holds the full
    protein sequence per row and nothing downstream reads it -- but every other column,
    including ``embedding_layer_*`` variants, is preserved.
    """
    header = pd.read_csv(path, nrows=0)
    columns = [c for c in header.columns if c != "seq"]
    embedding_columns = [c for c in columns if c == "embedding" or c.startswith("embedding_layer_")]
    collected: Dict[str, List[Any]] = {c: [] for c in columns}
    n_chunks = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_rows):
        for column in columns:
            values = chunk[column].to_numpy()
            if column in embedding_columns:
                collected[column].extend(
                    np.array(json.loads(v)) if isinstance(v, str) else v for v in values
                )
            else:
                collected[column].extend(values)
        n_chunks += 1
        del chunk
    frame = pd.DataFrame({columns[0]: collected[columns[0]]})
    for column in columns[1:]:
        frame[column] = collected[column]
    logger.info(
        f"Chunk-loaded {len(frame)} embedding rows from {os.path.basename(path)} "
        f"in {n_chunks} chunks of {chunk_rows}"
    )
    return frame


def try_load_sharded_embedding_file(embeddings_dir: str, prefix: str) -> pd.DataFrame:
    """Load sharded embedding files matching pattern <prefix>-(#)_of_(#).csv.

    Args:
        embeddings_dir: Directory containing the embedding files
        prefix: File prefix (e.g., "SPG1_STRSG_Olson_2014_embedding_15b")

    Returns:
        Concatenated DataFrame from all shards

    Raises:
        FileNotFoundError: If no shard files are found
        ValueError: If shard validation fails (missing shards, duplicates, etc.)
    """
    shard_pattern = os.path.join(embeddings_dir, f"{prefix}-*_of_*.csv")
    shard_files = glob.glob(shard_pattern)

    if not shard_files:
        raise FileNotFoundError(f"No shard files found matching pattern: {shard_pattern}")

    # Parse shard information: {shard_idx: (total_shards, filepath)}
    shard_info = {}
    shard_regex = re.compile(rf"{re.escape(prefix)}-(\d+)_of_(\d+)\.csv$")

    for filepath in shard_files:
        match = shard_regex.search(filepath)
        if not match:
            continue
        shard_idx = int(match.group(1))
        total_shards = int(match.group(2))

        if shard_idx in shard_info:
            raise ValueError(
                f"Duplicate shard index {shard_idx} found for {prefix}: "
                f"{shard_info[shard_idx][1]} and {filepath}"
            )

        shard_info[shard_idx] = (total_shards, filepath)

    # Validate shards
    if not shard_info:
        raise ValueError(f"No valid shard files found matching pattern: {shard_pattern}")

    # Check all shards have the same total count
    total_shard_counts = set(info[0] for info in shard_info.values())
    if len(total_shard_counts) != 1:
        raise ValueError(f"Inconsistent total shard counts found: {total_shard_counts}")

    expected_total_shards = total_shard_counts.pop()

    # Check all shard indices from 1 to N are present
    found_indices = set(shard_info.keys())
    expected_indices = set(range(1, expected_total_shards + 1))

    if found_indices != expected_indices:
        missing = expected_indices - found_indices
        extra = found_indices - expected_indices
        error_parts = []
        if missing:
            error_parts.append(f"missing shards: {sorted(missing)}")
        if extra:
            error_parts.append(f"unexpected shard indices: {sorted(extra)}")
        raise ValueError(f"Shard validation failed for {prefix}: {', '.join(error_parts)}")

    # Check no shard index is out of range
    max_idx = max(found_indices)
    if max_idx > expected_total_shards:
        raise ValueError(f"Shard index {max_idx} exceeds total shard count {expected_total_shards}")

    # Load and concatenate all shards in parallel
    logger.info(f"Loading {expected_total_shards} sharded embedding files for {prefix}")

    def load_and_parse_shard(idx: int) -> pd.DataFrame:
        logging.info(f"Loading shard {idx}/{expected_total_shards} from {shard_info[idx][1]}...")
        filepath = shard_info[idx][1]
        shard_df = pd.read_csv(filepath)
        _parse_embedding_columns_inplace(shard_df)
        logger.info(
            f"Loaded and parsed shard {idx}/{expected_total_shards} with {len(shard_df)} rows"
        )
        return shard_df

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all shard loading tasks
        futures = {
            idx: executor.submit(load_and_parse_shard, idx)
            for idx in range(1, expected_total_shards + 1)
        }
        # Collect results in order
        shard_dfs = [futures[idx].result() for idx in range(1, expected_total_shards + 1)]

    return pd.concat(shard_dfs, ignore_index=True)


# Sentinel embedding_model_id that builds one-hot encodings from the DMS file itself
# instead of reading precomputed protein-language-model embeddings from disk. Used by the
# ALDE baseline (Yang et al. 2025), whose default and wet-lab-validated encoding is one-hot,
# and which therefore needs no PLM artifacts at all.
ONEHOT_EMBEDDING_MODEL_ID = "onehot"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_ALLELE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class OneHotVector:
    """A one-hot encoded variant, stored as set-bit indices and densified on demand.

    Behaves like a 1-D float32 array to anything that calls ``np.asarray`` on it, but
    occupies ~1/40th the memory and pickles ~1/40th the bytes, which is what makes the
    large multi-mutation pools fit in the process pool. ``indices`` holds exactly one
    column per mutated position.
    """

    __slots__ = ("indices", "dim")

    def __init__(self, indices: NDArray[np.int32], dim: int):
        self.indices = indices
        self.dim = dim

    def __array__(self, dtype=None, copy=None):
        dense = np.zeros(self.dim, dtype=np.float32)
        dense[self.indices] = 1.0
        return dense if dtype is None else dense.astype(dtype, copy=False)

    def __len__(self) -> int:
        return self.dim

    @property
    def shape(self):
        return (self.dim,)

    def __reduce__(self):
        return (OneHotVector, (self.indices, self.dim))


def build_onehot_dataset(
    dms_id: str, wt_aa_seq: str, naturalness_model_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-hot embeddings and placeholder naturalness directly from a DMS file.

    The candidate pool is every variant in the ProteinGym DMS file. We take the union of
    all mutated positions across the pool and encode each variant as the concatenation of
    per-position 20-dim one-hot vectors, using the wild-type residue wherever a variant is
    unmutated. This is the natural generalization of ALDE's "concatenate the k mutated
    residues" encoding to a pool whose mutated positions are not a fixed small set.

    Returns:
        Tuple of (embedding_df, naturalness_df), both indexed by seq_id. The naturalness
        frame carries a constant ``log_wt_marginal`` column: ALDE never consults naturalness,
        but downstream code requires the column to exist and to be non-null.
    """
    dms_file_path = os.path.join(DMS_DIR, f"{dms_id}.csv")
    if not os.path.exists(dms_file_path):
        raise FileNotFoundError(f"DMS file not found: {dms_file_path}")

    activity_df = pd.read_csv(dms_file_path)
    seq_ids = activity_df["mutant"].apply(lambda x: allele_set_to_seq_id(set(x.split(":"))))

    naturalness_path = (
        os.path.join(NATURALNESS_DIR, f"{dms_id}_naturalness_{naturalness_model_id}.csv")
        if naturalness_model_id and naturalness_model_id != ONEHOT_EMBEDDING_MODEL_ID
        else None
    )
    if naturalness_path is not None and not os.path.exists(naturalness_path):
        # A config asking for real naturalness must get it or fail. Silently substituting a
        # constant placeholder makes a verification run look fine and mean nothing.
        raise FileNotFoundError(
            f"Naturalness file not found: {naturalness_path} "
            f"(naturalness_model_id={naturalness_model_id!r})"
        )

    raw_naturalness: Optional[pd.DataFrame] = None
    if naturalness_path:
        raw_naturalness = pd.read_csv(naturalness_path)
        raw_naturalness["seq_id"] = raw_naturalness["seq_id"].apply(
            lambda x: maybe_modify_seq_id(dms_id, x)
        )
        raw_naturalness = raw_naturalness[~raw_naturalness["seq_id"].duplicated()].set_index(
            "seq_id", drop=False
        )
        raw_naturalness["log_wt_marginal"] = raw_naturalness["wt_marginal"].apply(
            lambda x: np.log(max(x, 1e-20))
        )

        # NaturalnessImputer scores a multi-mutant as the sum over its constituent single
        # mutants, so every constituent must be present in the returned frames -- but a
        # multi-mutation DMS file routinely omits them as standalone variants (GRB2 is missing
        # 14, PABP 8). The embedding path never hits this because its pool is
        # embeddings & (naturalness | activity), and the naturalness CSV supplies every single.
        # Mirror that here. These carry no activity, so _finish_proteingym_dataset admits them
        # with a null activity and they never enter the candidate world -- they exist purely so
        # the imputer's lookup resolves. Appended after the pool so pool ordering, and therefore
        # the seeded bootstrap, is unchanged.
        pool_seq_ids = set(seq_ids)
        needed_singles: set = set()
        for seq_id in pool_seq_ids:
            needed_singles |= get_allele_set(seq_id)
        extra_singles = sorted((needed_singles - pool_seq_ids) & set(raw_naturalness.index))
        if extra_singles:
            logger.info(
                f"{dms_id}: adding {len(extra_singles)} constituent single mutants absent from "
                "the DMS file so NaturalnessImputer can derive multi-mutant scores"
            )
            seq_ids = pd.concat([seq_ids, pd.Series(extra_singles)], ignore_index=True)

    # Parse every allele once, and collect the union of mutated positions.
    parsed_per_variant: List[List[Tuple[int, str]]] = []
    positions: set = set()
    for seq_id in seq_ids:
        alleles = []
        for allele in get_allele_set(seq_id):
            match = _ALLELE_RE.match(allele)
            if match is None:
                continue
            _, locus, mut_aa = match.groups()
            if mut_aa not in _AA_TO_INDEX:
                continue
            alleles.append((int(locus), mut_aa))
            positions.add(int(locus))
        parsed_per_variant.append(alleles)

    sorted_positions = sorted(positions)
    position_to_slot = {locus: slot for slot, locus in enumerate(sorted_positions)}
    n_positions = len(sorted_positions)
    logger.info(
        f"One-hot encoding {dms_id}: {len(seq_ids)} variants over {n_positions} mutated "
        f"positions -> {n_positions * len(AMINO_ACIDS)} dimensions"
    )

    # Baseline row: wild-type residue at every mutated position (1-indexed loci).
    baseline = np.zeros((n_positions, len(AMINO_ACIDS)), dtype=np.float32)
    for locus, slot in position_to_slot.items():
        wt_aa = wt_aa_seq[locus - 1]
        if wt_aa in _AA_TO_INDEX:
            baseline[slot, _AA_TO_INDEX[wt_aa]] = 1.0

    # Store each variant as the indices of its set bits rather than a dense float row.
    # A one-hot row has exactly one 1 per mutated position, so the sparse form is
    # n_positions int32s instead of n_positions * 20 float32s -- a 40x reduction. For the
    # largest benchmark pools that is the difference between ~30 MB and ~1.2 GB, and the pool
    # is pickled once per queued simulation by the process pool, so the dense form multiplied
    # that cost by the number of simulations and exhausted memory. OneHotVector densifies on
    # demand via __array__, so every existing `np.asarray(emb)` consumer still works.
    baseline_indices = np.array(
        [slot * len(AMINO_ACIDS) + int(baseline[slot].argmax()) for slot in range(n_positions)],
        dtype=np.int32,
    )
    dim = n_positions * len(AMINO_ACIDS)
    embeddings = []
    for alleles in parsed_per_variant:
        idx = baseline_indices.copy()
        for locus, mut_aa in alleles:
            slot = position_to_slot[locus]
            idx[slot] = slot * len(AMINO_ACIDS) + _AA_TO_INDEX[mut_aa]
        embeddings.append(OneHotVector(idx, dim))

    embedding_df = pd.DataFrame({"seq_id": seq_ids.to_numpy()})
    embedding_df["embedding"] = embeddings
    embedding_df = embedding_df[~embedding_df["seq_id"].duplicated()]

    if raw_naturalness is not None:
        # Real naturalness alongside one-hot embeddings. This is the configuration that
        # isolates the effect of the round-1 strategy: the surrogate still sees one-hot
        # inputs, exactly as in the ALDE baseline, but the zero-shot model gets the same
        # naturalness FolDE uses. Only one variable changes between the two runs.
        naturalness_df = raw_naturalness[["seq_id", "log_wt_marginal"]].reindex(
            embedding_df["seq_id"]
        )
        naturalness_df["seq_id"] = embedding_df["seq_id"].to_numpy()
        # Leave uncovered variants NaN on purpose. The naturalness files score single mutants
        # only; NaturalnessImputer.impute() turns a NaN multi-mutant into the sum of its
        # constituent single-mutant log wt-marginals, which is the additive rule behind the
        # published multi-mutation zero-shot line and needs only the singles to derive every
        # double. Median-filling here handed the imputer a real number, so it passed it
        # straight through and every multi-mutant tied at the median -- making doubles
        # unselectable and the one-hot path inconsistent with the embedding path.
        missing = int(naturalness_df["log_wt_marginal"].isna().sum())
        if missing:
            logger.info(
                f"{dms_id}: {missing}/{len(naturalness_df)} pool variants have no direct "
                "naturalness; NaturalnessImputer derives them from their single mutants"
            )
        naturalness_df = naturalness_df.reset_index(drop=True)
    else:
        naturalness_df = pd.DataFrame(
            {
                "seq_id": embedding_df["seq_id"].to_numpy(),
                "log_wt_marginal": np.zeros(len(embedding_df), dtype=np.float64),
            }
        )
    return embedding_df, naturalness_df


def get_proteingym_dataset(
    dms_id: str, embedding_model_id: str, naturalness_model_id: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ProteinGym activity data, embeddings, and naturalness scores.

    Args:
        dms_id: Identifier for the DMS dataset (e.g., "BLAT_ECOLX_Stiffler_2015")
        embedding_model_id: Identifier for the embedding model (e.g., "300m")
        naturalness_model_id: Identifier for the naturalness model (e.g., "esm2")

    Returns:
        Tuple of (wt_aa_seq, naturalness_df, embedding_df, activity_df, category_df)

    Raises:
        FileNotFoundError: If any required files are not found
    """
    dms_metadata = get_dms_metadata()
    dms_metadata = dms_metadata[dms_metadata["DMS_id"] == dms_id]
    if len(dms_metadata) != 1:
        raise ValueError(
            f"Did not find one row for DMS {dms_id} in metadata file at {DMS_METADATA_FILE}"
        )
    wt_aa_seq = dms_metadata["target_seq"].iloc[0]

    # #########################################################
    # ONE-HOT SHORTCUT #########################################
    # No PLM artifacts on disk are needed or consulted in this mode; both the embeddings and
    # a placeholder naturalness frame are derived from the DMS file. See ALDEFewShotModel.
    if embedding_model_id == ONEHOT_EMBEDDING_MODEL_ID:
        embedding_df, incomplete_naturalness_df = build_onehot_dataset(
            dms_id, wt_aa_seq, naturalness_model_id
        )
        embedding_df = embedding_df.set_index("seq_id", drop=False)
        seq_ids_with_embeddings = set(embedding_df.index)
        incomplete_naturalness_df = incomplete_naturalness_df.set_index("seq_id", drop=False)
        seq_ids_with_naturalness = set(incomplete_naturalness_df.index)
        naturalness_df = incomplete_naturalness_df.reindex(embedding_df.index)
        return _finish_proteingym_dataset(
            dms_id=dms_id,
            wt_aa_seq=wt_aa_seq,
            embedding_df=embedding_df,
            naturalness_df=naturalness_df,
            seq_ids_with_embeddings=seq_ids_with_embeddings,
            seq_ids_with_naturalness=seq_ids_with_naturalness,
        )

    # #########################################################
    # LOAD EMBEDDING DATA ######################################
    embedding_file_path = os.path.join(
        EMBEDDINGS_DIR, f"{dms_id}_embedding_{embedding_model_id}.csv"
    )

    if os.path.exists(embedding_file_path):
        # Single file case. Chunk-load above a threshold: a single read_csv of a multi-GB
        # embedding CSV peaks at roughly 2.2x the file size because every embedding is first a
        # Python string. Below the threshold the single-shot read is faster, and both paths
        # produce identical float64 arrays.
        if os.path.getsize(embedding_file_path) > 2_000_000_000:
            embedding_df = load_embedding_csv_in_chunks(embedding_file_path)
        else:
            embedding_df = pd.read_csv(embedding_file_path)
            _parse_embedding_columns_inplace(embedding_df)
    else:
        # Try loading sharded files (parsing happens inside)
        prefix = f"{dms_id}_embedding_{embedding_model_id}"
        try:
            embedding_df = try_load_sharded_embedding_file(str(EMBEDDINGS_DIR), prefix)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Embedding file not found (neither single file nor shards): {embedding_file_path}"
            )

    # Check that the naturalness file exists
    naturalness_file_path = os.path.join(
        NATURALNESS_DIR, f"{dms_id}_naturalness_{naturalness_model_id}.csv"
    )
    if not os.path.exists(naturalness_file_path):
        raise FileNotFoundError(f"Naturalness file not found: {naturalness_file_path}")
    logger.info(f"Loaded embeddings for {dms_id} with {len(embedding_df)} rows")
    embedding_df["seq_id"] = embedding_df["seq_id"].apply(lambda x: maybe_modify_seq_id(dms_id, x))
    embedding_df = embedding_df.set_index("seq_id", drop=False)
    seq_ids_with_embeddings = set(embedding_df.index)

    # #########################################################
    # LOAD NATURALNESS DATA ######################################
    # AUGMENT SINGLE MUTANT NATURALNESS FOR MULTI MUTANTS ########
    incomplete_naturalness_df = pd.read_csv(naturalness_file_path)
    logger.info(
        f"Loaded naturalness scores for {dms_id} with {len(incomplete_naturalness_df)} rows"
    )
    assert (
        "seq_id" in incomplete_naturalness_df.columns
    ), f"Naturalness file missing 'seq_id' column. Available columns: {incomplete_naturalness_df.columns.tolist()}"
    incomplete_naturalness_df["seq_id"] = incomplete_naturalness_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(dms_id, x)
    )
    if any(incomplete_naturalness_df["wt_marginal"] < 0):
        raise ValueError(
            f"wt_marginal for {dms_id} is negative, eg: {incomplete_naturalness_df[incomplete_naturalness_df['wt_marginal'] <= 0]['wt_marginal'].tolist()[:5]}"
        )

    def safe_log_for_wt_naturalness(x):
        return np.log(max(x, 1e-20))

    if (
        "model" in incomplete_naturalness_df.columns
        and incomplete_naturalness_df.model.unique().size > 1
    ):
        # We have an ensemble naturalness file! Stay calm and follow the drill.
        # We turn the wt_marginal column into multiple columns, one for each model.
        # Note that we deliberately lose the wt_marginal column - we want this to crash
        # later on, unless the user specifies the proper column names.
        incomplete_naturalness_df["log_wt_marginal"] = incomplete_naturalness_df[
            "wt_marginal"
        ].apply(safe_log_for_wt_naturalness)
        incomplete_naturalness_df["model"] = incomplete_naturalness_df["model"].apply(
            lambda x: f"log_wt_marginal_{x}"
        )
        incomplete_naturalness_df = incomplete_naturalness_df.pivot(
            index="seq_id", columns="model", values="log_wt_marginal"
        )
    else:
        # Otherwise, under normal circumstances, just set up seq_id as index.
        incomplete_naturalness_df = incomplete_naturalness_df.set_index("seq_id", drop=False)
        incomplete_naturalness_df["log_wt_marginal"] = incomplete_naturalness_df[
            "wt_marginal"
        ].apply(safe_log_for_wt_naturalness)
        incomplete_naturalness_df.drop(columns=["wt_marginal"], inplace=True)

    seq_ids_with_naturalness = set(incomplete_naturalness_df.index)
    naturalness_df = incomplete_naturalness_df.reindex(embedding_df.index)

    return _finish_proteingym_dataset(
        dms_id=dms_id,
        wt_aa_seq=wt_aa_seq,
        embedding_df=embedding_df,
        naturalness_df=naturalness_df,
        seq_ids_with_embeddings=seq_ids_with_embeddings,
        seq_ids_with_naturalness=seq_ids_with_naturalness,
        incomplete_naturalness_df=incomplete_naturalness_df,
    )


def _finish_proteingym_dataset(
    dms_id: str,
    wt_aa_seq: str,
    embedding_df: pd.DataFrame,
    naturalness_df: pd.DataFrame,
    seq_ids_with_embeddings: set,
    seq_ids_with_naturalness: set,
    incomplete_naturalness_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach activity and category data and align everything to a common seq_id index.

    Shared by the PLM-embedding path and the one-hot path of get_proteingym_dataset.
    """
    if incomplete_naturalness_df is None:
        incomplete_naturalness_df = naturalness_df

    # #########################################################
    # LOAD ACTIVITY DATA ######################################
    # We mostly only pass through mutants that have activity data. But sometimes, for
    # those with just naturalness, we pass through a null activity value.
    if dms_id == "FLIP-AAV":
        incomplete_activity_df = pd.read_csv(FLIP_AAV_DATA_FILE)
        incomplete_activity_df = incomplete_activity_df.rename(columns={"homolog_seq_id": "seq_id"})
        incomplete_activity_df = incomplete_activity_df[
            ~incomplete_activity_df.full_aa_sequence.duplicated(keep=False)
        ]

        # Find if any of the seq_ids are in the naturalness df and rename where possible.
        logger.info(f"Converting seq_ids in naturalness df to full sequences")
        tmp_naturalness_df = pd.DataFrame(
            {"seq_id": incomplete_naturalness_df.index}, index=incomplete_naturalness_df.index
        )

        def maybe_convert_seq_id_to_seq(seq_id: str):
            if maybe_get_allele_id_error_message(wt_aa_seq, seq_id) is not None:
                return pd.NA
            return seq_id_to_seq(wt_aa_seq, seq_id)

        tmp_naturalness_df["full_seq"] = tmp_naturalness_df.seq_id.apply(
            maybe_convert_seq_id_to_seq
        )
        tmp_naturalness_df = tmp_naturalness_df[tmp_naturalness_df.full_seq.notna()]
        tmp_naturalness_df = tmp_naturalness_df[~tmp_naturalness_df.full_seq.duplicated()]
        tmp_naturalness_df.set_index("full_seq", drop=True, inplace=True)

        logger.info(f"Reassigning seq_ids from activity df to matching naturalness seq_ids.")
        incomplete_activity_df = incomplete_activity_df.join(
            tmp_naturalness_df.seq_id.rename("naturalness_seq_id"),
            on="full_aa_sequence",
            how="left",
        )
        incomplete_activity_df["seq_id"] = incomplete_activity_df.apply(
            lambda x: x.naturalness_seq_id if pd.notna(x.naturalness_seq_id) else x.seq_id, axis=1
        )

        incomplete_activity_df_seq_id_dupes = incomplete_activity_df.seq_id.duplicated()
        logger.info(
            f"Dropping {incomplete_activity_df_seq_id_dupes.sum()} seq_ids from activity df that are duplicated such as {incomplete_activity_df[incomplete_activity_df_seq_id_dupes].seq_id.tolist()[:5]} and {incomplete_activity_df[incomplete_activity_df_seq_id_dupes].seq_id.tolist()[-5:]}"
        )
        incomplete_activity_df = incomplete_activity_df[~incomplete_activity_df_seq_id_dupes]
    else:
        # Check that the DMS data exists
        dms_file_path = os.path.join(DMS_DIR, f"{dms_id}.csv")
        incomplete_activity_df = pd.read_csv(dms_file_path)
        # Convert 'mutant' column to 'seq_id' by replacing ':' with '_'
        incomplete_activity_df["seq_id"] = incomplete_activity_df["mutant"].apply(
            lambda x: allele_set_to_seq_id(set(x.split(":")))
        )
    logger.info(f"Loaded activity data for {dms_id} with {len(incomplete_activity_df)} rows")
    incomplete_activity_df = incomplete_activity_df.set_index("seq_id", drop=False)
    seq_ids_with_activity = set(incomplete_activity_df.index)
    activity_df = incomplete_activity_df.reindex(embedding_df.index)

    # #########################################################
    # LOAD CATEGORY DATA ######################################
    if dms_id == "FLIP-AAV":
        category_df = pd.read_csv(FLIP_AAV_DATA_FILE)
        category_df = category_df[
            ["homolog_seq_id"] + [c for c in category_df.columns if c.endswith("_split")]
        ]
        category_df = category_df.set_index("homolog_seq_id")
        # Replace all elements of category df with a bool whereever the string equals 'train'
        category_df = (category_df == "train").astype(bool)
        category_df = category_df.reindex(embedding_df.index)
        category_df = category_df.fillna(False)
    elif dms_id == "SPG1_STRSG_Olson_2014":
        valid_activity_seq_ids = activity_df[activity_df.DMS_score.notna()].index
        category_df = pd.DataFrame(
            {
                "one_vs_many_split": [
                    len(get_loci_set(seq_id)) <= 1 for seq_id in valid_activity_seq_ids
                ],
            },
            index=valid_activity_seq_ids,
        )
    elif dms_id == "SPG1_STRSG_Wu_2016":
        valid_activity_seq_ids = activity_df[activity_df.DMS_score.notna()].index
        category_df = pd.DataFrame(
            {
                "one_vs_many_split": [
                    len(get_loci_set(seq_id)) <= 1 for seq_id in valid_activity_seq_ids
                ],
                "two_vs_many_split": [
                    len(get_loci_set(seq_id)) <= 2 for seq_id in valid_activity_seq_ids
                ],
                "three_vs_many_split": [
                    len(get_loci_set(seq_id)) <= 3 for seq_id in valid_activity_seq_ids
                ],
            },
            index=valid_activity_seq_ids,
        )
    else:
        category_df = pd.DataFrame(index=embedding_df.index)
    category_df = category_df.reindex(activity_df.index)
    category_df = category_df.fillna(False)

    # Embeddings are already parsed during loading (via _parse_embedding_columns_inplace)

    # We lose ordering with the set operations but recover it with a sort later.
    logging.info(
        f"seq_ids_with_embeddings & seq_ids_with_naturalness: {len(seq_ids_with_embeddings & seq_ids_with_naturalness)}"
    )
    logging.info(
        f"seq_ids_with_embeddings & seq_ids_with_activity: {len(seq_ids_with_embeddings & seq_ids_with_activity)}"
    )
    logging.info(
        f"seq_ids_with_naturalness & seq_ids_with_activity: {len(seq_ids_with_naturalness & seq_ids_with_activity)}"
    )
    common_seq_ids = list(
        seq_ids_with_embeddings & (seq_ids_with_naturalness | seq_ids_with_activity)
    )
    common_seq_ids = sort_seq_id_list(wt_aa_seq, common_seq_ids)

    logging.info(f"Going forward with {len(common_seq_ids)} common seq ids")
    if activity_df.shape[0] > len(common_seq_ids):
        logging.warning(
            f"Dropping seq ids from activity df such as {activity_df[~activity_df.index.isin(common_seq_ids)].index[:3].tolist()}"
        )

    common_seq_id_index = pd.Index(common_seq_ids)
    return (
        wt_aa_seq,
        naturalness_df.loc[common_seq_id_index],
        embedding_df.loc[common_seq_id_index],
        activity_df.loc[common_seq_id_index],
        category_df.loc[common_seq_id_index],
    )
