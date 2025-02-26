# my_esm_lib/training.py

import os
import re
import random
import numpy as np
import pandas as pd
import torch
import evaluate
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import logging
import types

import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import TrainerCallback
from app.helpers.sequence_util import seq_id_to_seq

# from .modeling_esm import load_esm_model
# from .ranking_trainer import RankingTrainer


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def create_dataset(tokenizer, sequences, labels, seq_ids):
    """
    Create a dataset from sequences, labels, and sequence IDs.
    """
    from datasets import Dataset

    # Convert sequences to list if it's a pandas Series
    if hasattr(sequences, "tolist"):
        sequences = sequences.tolist()

    # Tokenize all sequences
    tokenized = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    # Create dataset dictionary
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

    # Add labels
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    dataset_dict["labels"] = labels

    # Add sequence IDs
    if hasattr(seq_ids, "tolist"):
        seq_ids = seq_ids.tolist()
    dataset_dict["seq_ids"] = seq_ids

    return Dataset.from_dict(dataset_dict)


class LoggingCallback(TrainerCallback):
    """Custom callback to log training progress metrics to Python's logging system."""

    def __init__(self, logging_interval=10):
        self.logging_interval = logging_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.logging_interval == 0:
            # Filter out logs that contain metrics
            # metrics_logs = {
            #     k: v for k, v in logs.items() if k not in ["epoch", "total_flos"]
            # }
            # if metrics_logs:
            log_str = f"Step {state.global_step}: " + ", ".join(
                [f"{k}={v:.4f}" for k, v in logs.items()]
            )
            logging.info(log_str)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log_str = f"Evaluation at step {state.global_step}: " + ", ".join(
                [f"{k}={v:.4f}" for k, v in metrics.items()]
            )
            logging.info(log_str)


def full_ranking_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Custom ranking loss that does pairwise BCE across the batch.

    preds: (batch_size,) or (batch_size, num_classes) logit predictions
    targets: (batch_size,) or (batch_size, 1) real-valued or ordinal targets
    """
    # For classification, take the positive class logit
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        raise ValueError(
            f"Only one score should be provided per sequence, got {preds.shape[1]} scores"
        )

    # Ensure everything is shape (batch_size,)
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate pairwise differences between all predictions
    pairwise_diffs = preds[:, None] - preds[None, :]

    # Determine if each target is greater than others in pairwise manner
    target_comparisons = (targets[:, None] > targets[None, :]).float()

    # Compute binary cross-entropy loss for pairwise comparisons
    ranking_loss = F.binary_cross_entropy_with_logits(
        pairwise_diffs, target_comparisons, reduction="none"
    )

    # Create a mask to exclude diagonal (self-comparisons)
    batch_size = preds.size(0)
    diag_mask = 1 - torch.eye(batch_size, device=preds.device)

    # Apply mask and average
    masked_loss = 0.5 * ranking_loss * diag_mask
    return masked_loss.mean((-1, -2))


def train_per_protein(
    checkpoint: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: torch.device,
    num_labels: int = 1,
    use_ranking_loss: bool = False,
    train_batch_size: int = 4,
    grad_accum_steps: int = 2,
    val_batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    seed: int = 42,
    deepspeed_config=None,
    mixed_precision: bool = True,
    train_full: bool = False,
):
    """
    Main training function for ESM with mutation score calculation.
    """

    # Set random seeds
    set_all_seeds(seed)

    # 1. Load the model
    logging.info(f"Loading model from {checkpoint}")
    model, tokenizer = load_esm_model(
        checkpoint,
        num_labels=num_labels,
        half_precision=mixed_precision,
        train_full=train_full,
        deepspeed=bool(deepspeed_config),
    )

    # Move model to the specified device
    model.to(device)

    if epochs == 0:
        return tokenizer, model, []

    # 2. Basic data cleaning: e.g., replace weird AAs with X
    for df in [train_df, valid_df]:
        df["sequence"] = df["sequence"].str.replace(r"[OBUZJ]", "X", regex=True)

    # 3. Check if seq_id column exists
    if use_ranking_loss:
        if "seq_id" not in train_df.columns or "seq_id" not in valid_df.columns:
            raise ValueError(
                "seq_id column is required in both train_df and valid_df for mutation score calculation"
            )

    # 4. Create HF Datasets
    required_columns = ["sequence", "label", "seq_id"]
    for df, df_name in [(train_df, "train_df"), (valid_df, "valid_df")]:
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"{col} column is required in {df_name}, only found {df.columns}"
                )

    train_dataset = create_dataset(
        tokenizer, train_df["sequence"], train_df["label"], train_df["seq_id"]
    )
    valid_dataset = create_dataset(
        tokenizer, valid_df["sequence"], valid_df["label"], valid_df["seq_id"]
    )

    # 5. Create custom data collator that preserves seq_ids
    class CustomDataCollator(DataCollatorWithPadding):
        """Custom data collator that keeps seq_ids as is without trying to tensorize them."""

        def __call__(self, features):
            # Extract seq_ids before handling the rest
            seq_ids = (
                [f.pop("seq_ids") for f in features]
                if "seq_ids" in features[0]
                else None
            )

            # Process the remaining features normally (convert to tensors, pad, etc.)
            batch = super().__call__(features)

            # Add seq_ids back to the batch without tensorizing
            if seq_ids:
                batch["seq_ids"] = seq_ids

            return batch

    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # 6. Trainer arguments
    logging.info(f"Creating training arguments")
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",  # or "epoch"/"steps" if you prefer
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=deepspeed_config,
        fp16=mixed_precision,
        remove_unused_columns=False,
    )

    def compute_spearmanr(eval_pred):
        predictions, labels = eval_pred
        # If predictions are scores from our custom trainer
        if isinstance(predictions, dict) and "scores" in predictions:
            predictions = predictions["scores"]
        rho, _ = spearmanr(predictions, labels)
        return {"spearmanr": rho}

    # 7. Decide which Trainer
    if use_ranking_loss:
        trainer_cls = ESMMutationScoreTrainer
    else:
        trainer_cls = Trainer

    # Create custom logging callback
    logging_callback = LoggingCallback(logging_interval=1)

    logging.info(f"Creating trainer")
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_spearmanr,
        callbacks=[logging_callback],
    )

    logging.info(f"Training model")
    trainer.train()
    logging.info(f"Training complete")

    # Return model, tokenizer, plus the trainer log
    return tokenizer, model, trainer.state.log_history


def load_esm_model(
    checkpoint,
    num_labels=1,
    half_precision=False,
    train_full=False,
    deepspeed=False,
):
    """
    Load an ESM model for masked language modeling to calculate mutation scores.
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Use AutoModelForMaskedLM to get the language modeling head
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    if half_precision:
        model = model.half()

    # Configure which parts of the model to train
    if not train_full:
        # Freeze most parameters except for specific layers you want to train
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the layers you want to train
        # For example, to train only the final layers:
        for i, layer in enumerate(model.esm.encoder.layer):
            if i >= len(model.esm.encoder.layer) - 2:  # Unfreeze last 2 layers
                for param in layer.parameters():
                    param.requires_grad = True

        # Always unfreeze the LM head
        for param in model.lm_head.parameters():
            param.requires_grad = True

    return model, tokenizer


class ESMMutationScoreTrainer(Trainer):
    """
    Custom trainer that calculates mutation scores from ESM masked language modeling
    and uses them with a ranking loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the wild-type sequence for reference
        self.wt_sequence = kwargs.pop("wt_sequence", None)


def parse_mutations(seq_id):
    """
    Parse mutations from a sequence ID (e.g., "A132W_S155G").

    Returns:
        List of tuples (wt_aa, position, mut_aa)
    """
    import re

    if not seq_id:
        return []

    mutations = []
    for mut in seq_id.split("_"):
        match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
        if match:
            wt_aa, pos_str, mut_aa = match.groups()
            pos = int(pos_str) - 1  # convert to 0-indexed
            mutations.append((wt_aa, pos, mut_aa))
    return mutations


class ESMMutationScoreTrainer(Trainer):
    # Override _prepare_inputs to preserve extra keys (e.g. "seq_ids")
    def _prepare_inputs(self, inputs):
        return inputs

    def __init__(self, *args, **kwargs):
        # Remove and store wt_sequence if provided
        self.wt_sequence = kwargs.pop("wt_sequence", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure that "seq_ids" and "labels" are present.
        if "seq_ids" not in inputs:
            raise ValueError(
                f"seq_ids is required for ESMMutationScoreTrainer. Inputs had {inputs.keys()}"
            )
        if "labels" not in inputs:
            raise ValueError(
                f"labels is required for ESMMutationScoreTrainer. Inputs had {inputs.keys()}"
            )

        # Remove extra keys so they are not passed to the model's forward
        seq_ids = inputs.pop("seq_ids")
        labels = inputs.pop("labels")

        # Single forward pass for the entire batch
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Calculate scores for each sequence
        scores_list = []
        for i, seq_id in enumerate(seq_ids):
            sequence_score = torch.tensor(0.0, device=logits.device)
            # Parse mutations from seq_id (e.g., "A132W_S155G")
            if seq_id and seq_id != "WT":
                mutations = parse_mutations(seq_id)
                if mutations:
                    for wt_aa, pos, mut_aa in mutations:
                        # Account for special tokens (e.g., <cls>)
                        token_pos = pos + 1  # +1 for <cls> token

                        # Get probabilities for this position
                        probs = torch.softmax(logits[i, token_pos], dim=0)

                        # Get token IDs for wild-type and mutant amino acids
                        wt_token_id = self.tokenizer.convert_tokens_to_ids(wt_aa)
                        mut_token_id = self.tokenizer.convert_tokens_to_ids(mut_aa)

                        # Calculate score for this mutation
                        wt_prob = probs[wt_token_id]
                        mut_prob = probs[mut_token_id]
                        score = torch.log(mut_prob + 1e-10) - torch.log(wt_prob + 1e-10)
                        sequence_score += score

            scores_list.append(sequence_score)

        scores = torch.stack(scores_list)

        # Convert labels to tensor if needed and ensure same device
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=scores.device)
        if labels.device != scores.device:
            labels = labels.to(scores.device)

        loss = full_ranking_bce(scores, labels)
        outputs = {"loss": loss, "scores": scores}

        return (loss, outputs) if return_outputs else loss


def score_sequences(model, tokenizer, wt_aa_seq, seq_ids, batch_size=32):
    """Score a list of sequences using the trained model.

    Args:
        model: Trained ESM model
        tokenizer: ESM tokenizer
        wt_aa_seq: Wild-type amino acid sequence
        seq_ids: List of sequence IDs to score
        batch_size: Batch size for processing

    Returns:
        DataFrame with columns:
        - seq_id: Sequence identifier
        - sequence: Full amino acid sequence
        - wt_marginal_score: Sum of wild-type amino acid log probabilities
    """
    import torch
    import pandas as pd
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    results = []

    # Process in batches
    for i in range(0, len(seq_ids), batch_size):
        logging.info(
            f"Scoring batch {i // batch_size + 1} of {len(seq_ids) // batch_size}"
        )
        batch_seq_ids = seq_ids[i : i + batch_size]

        # Convert seq_ids to sequences
        sequences = [seq_id_to_seq(wt_aa_seq, seq_id) for seq_id in batch_seq_ids]

        # Tokenize
        inputs = tokenizer(
            sequences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Calculate scores for each sequence
        for j, (seq_id, sequence) in enumerate(zip(batch_seq_ids, sequences)):
            sequence_score = torch.tensor(0.0, device=logits.device)
            for wt_aa, pos, mut_aa in parse_mutations(seq_id):
                # Account for special tokens (e.g., <cls>)
                token_pos = pos + 1  # +1 for <cls> token

                # Get probabilities for this position
                probs = torch.softmax(logits[j, token_pos], dim=0)

                # Get token IDs for wild-type and mutant amino acids
                wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
                mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)

                # Calculate score for this mutation
                wt_prob = probs[wt_token_id]
                mut_prob = probs[mut_token_id]
                score = torch.log(mut_prob + 1e-10) - torch.log(wt_prob + 1e-10)
                sequence_score += score

            results.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "wt_marginal_score": sequence_score.item(),
                }
            )

    return pd.DataFrame(results)
