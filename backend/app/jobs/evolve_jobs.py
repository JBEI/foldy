import io
import json
import logging
import re
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest

from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.jobs_util import (
    LoggingRecorder,
    _live_update_tail,
    _psql_tail,
)
from app.helpers.sequence_util import (
    get_loci_set,
    is_homolog_seq_id,
    process_and_validate_evolve_input_files,
)
from app.models import FewShot, Fold, Invokation
from folde.few_shot_models import get_few_shot_model, is_valid_few_shot_model_name


def get_embedding_df_from_file(
    fold_id: int, fsm: FoldStorageManager, embedding_files: list[str]
) -> pd.DataFrame:
    logging.info(f"Reading {len(embedding_files)} embedding files")

    embedding_dfs = []
    chunk_size = 10000  # Adjust based on memory constraints

    for path in embedding_files:
        # Get the CSV content as a string
        assert fsm.storage_manager is not None, "Storage manager not set up"
        csv_blob = fsm.storage_manager.get_blob(fold_id, path)

        with csv_blob.open("r") as csv_f:
            # Create chunks iterator
            chunks = pd.read_csv(csv_f, chunksize=chunk_size)

            # Process each chunk
            path_dfs = []
            for chunk in chunks:
                path_dfs.append(chunk)

            # Combine chunks for this path
            if path_dfs:
                embedding_dfs.append(pd.concat(path_dfs, ignore_index=True))
    raw_embedding_df = pd.concat(embedding_dfs, ignore_index=True)
    return raw_embedding_df


def get_naturalness_df_from_file(
    fold_id: int, fsm: FoldStorageManager, naturalness_files: list[str]
) -> pd.DataFrame:
    logging.info(f"Reading {len(naturalness_files)} naturalness files")
    naturalness_dfs = []

    for path in naturalness_files:
        assert fsm.storage_manager is not None, "Storage manager not set up"
        csv_blob = fsm.storage_manager.get_blob(fold_id, path)
        with csv_blob.open("r") as csv_f:
            naturalness_dfs.append(pd.read_csv(csv_f))
    raw_naturalness_df = pd.concat(naturalness_dfs, ignore_index=True)

    # Drop weird rows.
    weird_bracket_re = re.compile(r"[A-Z]\d+<.*>")
    weird_character_re = re.compile(r"[A-Z]\d+[\.\-|]")
    weird_aminoacid_re = re.compile(r"[A-Z]\d+[OUX]")

    def is_self_mutation(seq_id: str) -> bool:
        if is_homolog_seq_id(seq_id):
            return False
        if seq_id == "WT":
            return False
        return len(seq_id.split("_")) == 1 and seq_id[0] == seq_id[-1]

    is_weird_row = raw_naturalness_df.seq_id.apply(
        lambda x: bool(
            weird_bracket_re.fullmatch(x)
            or weird_character_re.fullmatch(x)
            or weird_aminoacid_re.fullmatch(x)
            or is_self_mutation(x)
        )
    )
    raw_naturalness_df = raw_naturalness_df[~is_weird_row]

    return raw_naturalness_df


def run_few_shot_prediction(few_shot_id: int):
    """Run the evolvepro workflow."""
    few_shot = FewShot.get_by_id(few_shot_id)
    if not few_shot:
        raise BadRequest(f"FewShot {few_shot_id} not found")
    fold = Fold.get_by_id(few_shot.fold_id)
    if not fold:
        raise BadRequest(f"Fold {few_shot.fold_id} not found")
    invokation = Invokation.get_by_id(few_shot.invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {few_shot.invokation_id} not found")

    with LoggingRecorder(invokation):
        """Helper function to run evolvepro with a logger."""
        # REQUIRED SETUP #######################################################
        fsm = FoldStorageManager()
        fsm.setup()

        # INPUT VALIDATION #####################################################
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        if not few_shot.embedding_files or not few_shot.naturalness_files:
            raise ValueError(
                f"These days, slate build jobs must specify both embedding files (found {few_shot.embedding_files}) and naturalness files (found {few_shot.naturalness_files})"
            )
        if not few_shot.few_shot_params:
            raise ValueError(
                f"These days, few shot params are required, got {few_shot.few_shot_params}"
            )
        if not few_shot.mode or not is_valid_few_shot_model_name(few_shot.mode):
            raise BadRequest(f"Old modes such as {few_shot.mode} are no longer supported.")
        if few_shot.num_mutants is None or few_shot.num_mutants <= 0:
            raise ValueError(
                f"Slate build job must specify a positive number of mutants, got {few_shot.num_mutants}"
            )

        # LOAD INPUTS #########################################################
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) != 1:
            raise ValueError(
                f"Fold has {len(boltz_yaml_helper.get_protein_sequences())} protein sequences, which is not supported for evolvepro yet."
            )

        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]
        few_shot_directory = Path("few_shots") / few_shot.name
        try:
            few_shot_params = json.loads(few_shot.few_shot_params)
        except Exception as e:
            raise BadRequest(f"Failed to parse few shot params: {e}")

        # 2. Read and merge all embedding CSVs
        activity_fpath = few_shot.input_activity_fpath
        if not activity_fpath:
            raise ValueError("No activity file path found for few shot")
        activity_file_contents = fsm.storage_manager.get_binary(few_shot.fold_id, activity_fpath)

        raw_activity_df = pd.read_excel(BytesIO(activity_file_contents))
        raw_embedding_df = get_embedding_df_from_file(
            few_shot.fold_id, fsm, few_shot.embedding_files.split(",")
        )
        raw_naturalness_df = get_naturalness_df_from_file(
            few_shot.fold_id, fsm, few_shot.naturalness_files.split(",")
        )

        logging.info(
            f"Found {raw_embedding_df.shape[0]} embeddings and {raw_naturalness_df.shape[0]} naturalness values"
        )

        # PROCESS INPUTS #######################################################
        activity_df, embedding_df, naturalness_df = process_and_validate_evolve_input_files(
            wt_aa_seq, raw_activity_df, raw_embedding_df, raw_naturalness_df
        )
        assert embedding_df is not None
        assert naturalness_df is not None
        logging.info(
            f"Found {activity_df.shape[0]} activity measurements among {activity_df.index.unique().shape[0]} mutants"
        )

        # # AUGMENT SINGLE MUTANT NATURALNESS FOR MULTI MUTANTS ##################
        # def get_naturalness_of_multi_mutant(seq_id) -> float:
        #     if seq_id == "WT":
        #         return 1.0
        #     try:
        #         return incomplete_naturalness_df.wt_marginal.loc[seq_id.split("_")].prod()
        #     except Exception as e:
        #         raise BadRequest(f"Failure computing naturalness for {seq_id}: {e}")

        # augmented_naturalness_series = pd.Series(
        #     embedding_df.index.map(get_naturalness_of_multi_mutant), index=embedding_df.index
        # )

        # VALIDATE FINAL INPUT SEQUENCES #######################################
        for seq_id in activity_df.index:
            if seq_id not in embedding_df.index:
                raise ValueError(
                    f"Activity seq id {seq_id} is missing either an embedding or naturalness value"
                )

        few_shot_model = get_few_shot_model(
            few_shot.mode,
            random_state=42,
            wt_aa_seq=wt_aa_seq,
            **few_shot_params,
        )

        #########################################################
        # WARM-START / PRETRAINING ##############################
        nat_not_in_emb = ~naturalness_df.index.isin(embedding_df.index)
        if nat_not_in_emb.any():
            raise ValueError(
                f"Found {nat_not_in_emb.sum()} naturalness values that are not in the embedding_df including {list(naturalness_df.index[nat_not_in_emb])[:3]}"
            )
        pretrain_embedding_series = embedding_df.embedding.loc[naturalness_df.index]
        few_shot_model.pretrain(
            naturalness_df,
            pretrain_embedding_series,
        )

        #########################################################
        # FIT TO ACTIVITY DATA ##################################
        entire_embedding_series = embedding_df.embedding
        entire_naturalness_df = naturalness_df.reindex(entire_embedding_series.index)

        training_activity_series = activity_df.activity
        few_shot_model.fit(
            entire_naturalness_df,
            entire_embedding_series,
            training_activity_series,
        )

        #########################################################
        # BUILD SLATE ###########################################
        top_seq_ids, predicted_activity_ensemble = few_shot_model.get_top_n(
            few_shot.num_mutants,
            entire_naturalness_df,
            entire_embedding_series,
        )

        predicted_activity_df = pd.DataFrame(
            {
                f"model_{ii}": predicted_activity_ensemble[ii]
                for ii in range(len(predicted_activity_ensemble))
            },
            index=predicted_activity_ensemble[0].index,
        )

        logging.info(f"Top seq ids: {top_seq_ids}")

        # def get_selected_idx_or_none(seq_id):
        #     try:
        #         return top_seq_ids.index(seq_id)
        #     except ValueError as e:
        #         return None
        predicted_activity_df["selected"] = predicted_activity_df.index.isin(top_seq_ids)

        # predicted_activity_df[~predicted_activity_df.selected_idx.isna()]

        try:
            loci_to_measured_mutants = defaultdict(list)
            for measured_seq_id in activity_df.index.unique():
                loci = get_loci_set(measured_seq_id)
                for locus in loci:
                    loci_to_measured_mutants[locus].append(measured_seq_id)

            def get_relevant_measured_mutants(seq_id) -> str:
                return ", ".join(
                    sorted(
                        sum(
                            [
                                loci_to_measured_mutants.get(locus, [])[:3]
                                for locus in get_loci_set(seq_id)
                            ],
                            [],
                        )
                    )
                )

            predicted_activity_df["relevant_measured_mutants"] = predicted_activity_df.index.map(
                get_relevant_measured_mutants
            )
        except Exception as e:
            logging.error(f"Error computing relevant measured mutants: {e}")

        # Save debug info to file
        try:
            fsm.storage_manager.write_file(
                few_shot.fold_id,
                str(few_shot_directory / "debug_info.json"),
                json.dumps(few_shot_model.get_debug_info()),
            )
        except Exception as e:
            logging.error(f"Error saving debug info to file: {e}")
        output_path = str(few_shot_directory / "predicted_activity.csv")
        fsm.storage_manager.write_file(
            few_shot.fold_id,
            output_path,
            predicted_activity_df.to_csv(),
        )

        # Update the slate build record with the output file path
        few_shot.output_fpath = output_path
        few_shot.save()
