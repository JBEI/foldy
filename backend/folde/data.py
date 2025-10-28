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

from app.helpers.sequence_util import (
    allele_set_to_seq_id,
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
    # LOAD EMBEDDING DATA ######################################
    embedding_file_path = os.path.join(
        EMBEDDINGS_DIR, f"{dms_id}_embedding_{embedding_model_id}.csv"
    )

    if os.path.exists(embedding_file_path):
        # Single file case
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
