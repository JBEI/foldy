from typing import Callable
import pandas as pd
import re
import json
from app.helpers.jobs_util import get_torch_cuda_is_available_and_add_logs
from typing import Optional


def get_naturalness(
    wt_aa_seq: str,
    logit_model: str,
    add_log: Callable[[str], None],
    pdb_file_path: Optional[str] = None,
):
    # Import ESM client
    add_log(f"Creating ESM client for {logit_model}")
    from app.helpers.esm_client import FoldyESMClient
    import torch

    # Log cache directories
    torch_cache_dir = torch.hub.get_dir()
    add_log(f"Torch cache directory: {torch_cache_dir}")

    # Try to get Hugging Face cache dir if available
    try:
        from huggingface_hub import get_cache_dir

        hf_cache_dir = get_cache_dir()
        add_log(f"Hugging Face cache directory: {hf_cache_dir}")
    except Exception as e:
        add_log(f"Hugging Face cache directory command failed: {e}")

    add_log(
        f"Starting to load model {logit_model} - this may take several minutes on first run"
    )
    get_torch_cuda_is_available_and_add_logs(add_log)

    # Create client using factory method
    client = FoldyESMClient.get_client(logit_model)
    add_log(f"Model {logit_model} loaded successfully")

    # Get logits using the client
    add_log("Computing logits")
    melted_df = client.get_logits(wt_aa_seq, pdb_file_path)

    # Process the melted dataframe to add WT marginal scores
    add_log("Processing logits and preparing to save")

    def seq_id_to_locus(seq_id):
        return int(re.match(r".(\d+).*", seq_id).group(1))

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
    position_probs = []
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
