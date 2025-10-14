import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from app.helpers.jobs_util import get_torch_cuda_is_available_and_add_logs
from app.helpers.sequence_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
)


def get_naturalness(
    wt_aa_seq: str,
    logit_model: str,
    get_depth_two_logits: Optional[bool] = False,
    cif_file_path: Optional[str] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    Compute naturalness scores for a given wild-type amino acid sequence.

    Args:
        wt_aa_seq: Wild-type amino acid sequence
        logit_model: ESM model name to use for logit computation
        get_depth_two_logits: If True, compute logits for all second-order mutants
        cif_file_path: Optional path to CIF file for structure-aware models

    Returns:
        Tuple containing:
            - JSON string of position probabilities (empty for depth-two logits)
            - DataFrame with melted logit data
    """
    # Import ESM client
    logging.info(f"Creating ESM client for {logit_model}")
    import torch
    from app.helpers.esm_client import FoldyESMClient

    # Log cache directories
    torch_cache_dir = torch.hub.get_dir()
    logging.info(f"Torch cache directory: {torch_cache_dir}")

    # Try to get Hugging Face cache dir if available
    try:
        from huggingface_hub import get_cache_dir

        hf_cache_dir = get_cache_dir()
        logging.info(f"Hugging Face cache directory: {hf_cache_dir}")
    except Exception as e:
        logging.info(f"Hugging Face cache directory command failed: {e}")

    logging.info(
        f"Starting to load model {logit_model} - this may take several minutes on first run"
    )
    get_torch_cuda_is_available_and_add_logs(logging.info)

    # Create client using factory method
    client = FoldyESMClient.get_client(logit_model)
    logging.info(f"Model {logit_model} loaded successfully")

    # Get logits using the client
    logging.info("Computing logits")
    if get_depth_two_logits:
        base_seq_ids = get_seq_ids_for_deep_mutational_scan(wt_aa_seq, ["WT"], ["WT"])
        logging.info(
            f"Going to depth 2 mutants; getting logits for {len(base_seq_ids)} base mutants"
        )

        melted_df_list: List[pd.DataFrame] = []
        for ii, base_seq_id in enumerate(base_seq_ids):
            base_seq = seq_id_to_seq(wt_aa_seq, base_seq_id)
            melted_df = client.get_logits(base_seq, cif_file_path)
            melted_df["base_seq_id"] = base_seq_id
            melted_df_list.append(melted_df)

            if ii % 100 == 0:
                logging.info(f"Finished {ii}/{len(base_seq_ids)} base mutants")
        melted_df = pd.concat(melted_df_list)
        return "", melted_df
    else:
        melted_df = client.get_logits(wt_aa_seq, cif_file_path)

        # Process the melted dataframe to add WT marginal scores
        logging.info("Processing logits and preparing to save")

        def seq_id_to_locus(seq_id: str) -> int:
            """Extract locus position from sequence ID."""
            match = re.match(r".(\d+).*", seq_id)
            if match is None:
                raise ValueError(f"Invalid sequence ID format: {seq_id}")
            return int(match.group(1))

        melted_df["locus"] = melted_df.seq_id.apply(seq_id_to_locus)

        # Add the "WT marginal" score column
        wt_naturalness = (
            melted_df[melted_df.seq_id.apply(lambda x: x[0] == x[-1])]
            .rename(columns={"probability": "wt_probability"})
            .drop(columns=["seq_id"])
        )

        melted_df = pd.merge(
            melted_df, wt_naturalness, left_on="locus", right_on="locus", how="left"
        )
        melted_df["wt_marginal"] = melted_df.apply(
            lambda x: x.probability / x.wt_probability, axis=1
        )

        # Create position_probs format for JSON
        position_probs: List[Dict[str, Any]] = []
        for pos in range(1, len(wt_aa_seq) + 1):
            pos_probs = melted_df[melted_df.locus == pos]
            wt_aa = wt_aa_seq[pos - 1]
            probs = [
                float(pos_probs[pos_probs.seq_id.str.endswith(aa)].probability.iloc[0])
                for aa in pos_probs.seq_id.str[-1].unique()
            ]
            position_probs.append({"locus": pos, "wt_aa": wt_aa, "probabilities": probs})

        logits_json = json.dumps(position_probs)

        return logits_json, melted_df
