# my_esm_lib/training.py

import logging
import os
import random
import re
import types

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
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


def create_dataset_entropy(tokenizer, input_df):
    """
    Create a dataset from a dataframe with columns "sequence", "label", and "seq_id".
    """
    from datasets import Dataset

    if any([col not in input_df.columns for col in ["sequence", "label", "seq_id"]]):
        raise ValueError(
            f"Input dataframe must have columns 'sequence', 'label', 'seq_id', got {input_df.columns}"
        )

    # Tokenize all sequences
    tokenized = tokenizer(
        input_df["sequence"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    # Create dataset dictionary
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": input_df["label"].tolist(),
        "seq_id": input_df["seq_id"].tolist(),
    }

    return Dataset.from_dict(dataset_dict)


def create_dataset_direct_preference(tokenizer, input_df):
    """Takes a tokenizer and a dataframe with columns "sequence", "seq_id_w", "seq_id_l", and puts them into a dataset.

    Conceptually each row corresponds to a WT sequence, the seq_id and
    label for a "winning" mutant, and the seq_id and label for a "losing" mutant."""
    from datasets import Dataset

    if any([col not in input_df.columns for col in ["sequence", "seq_id_w", "seq_id_l"]]):
        raise ValueError(
            f"Input dataframe must have columns 'sequence', 'seq_id_w', 'seq_id_l', got {input_df.columns}"
        )

    # Tokenize all sequences
    tokenized = tokenizer(
        input_df["sequence"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    # Create dataset dictionary
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "seq_id_w": input_df["seq_id_w"].tolist(),
        "seq_id_l": input_df["seq_id_l"].tolist(),
    }

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
                [f"{k}={v:.4f}" for k, v in (logs.items() if logs else [])]
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

    # Ensure everything is shape (batch_size,) and has gradients enabled
    preds = preds.view(-1)
    targets = targets.view(-1).detach()  # Detach targets as we don't need gradients for them

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
    return masked_loss.mean()


def train_per_protein(
    checkpoint: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: torch.device,
    loss: str = "dpo",  # can be "dpo" or "entropy"
    train_batch_size: int = 4,
    grad_accum_steps: int = 1,
    val_batch_size: int = 4,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    seed: int = 42,
    deepspeed_config=None,
    mixed_precision: bool = True,
    train_full: bool = False,
    output_dir: str = "./checkpoints",
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
        half_precision=mixed_precision,
        train_full=train_full,
        deepspeed=bool(deepspeed_config),
    )

    # Move model to the specified device
    model.to(device)

    if epochs == 0:
        return tokenizer, model, []

    # # 2. Basic data cleaning: e.g., replace weird AAs with X
    # for df in [train_df, valid_df]:
    #     df["sequence"] = df["sequence"].str.replace(r"[OBUZJ]", "X", regex=True)

    # 4. Trainer arguments
    logging.info(f"Creating training arguments")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=10,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=deepspeed_config,
        fp16=mixed_precision,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        warmup_steps=150,
        metric_for_best_model="ranking_accuracy" if loss == "dpo" else "spearmanr",
        load_best_model_at_end=True,
    )

    compute_evaluation_function = None
    if loss == "dpo":
        compute_evaluation_function = compute_ranking_accuracy
    elif loss == "entropy":
        compute_evaluation_function = compute_spearmanr

    # 5. Create HF Datasets
    if loss == "entropy":
        train_dataset = create_dataset_entropy(tokenizer, train_df)
        valid_dataset = create_dataset_entropy(tokenizer, valid_df)
    elif loss == "dpo":
        train_dataset = create_dataset_direct_preference(tokenizer, train_df)
        valid_dataset = create_dataset_direct_preference(tokenizer, valid_df)
    else:
        raise ValueError(f"Invalid loss function: {loss}")

    # 6. Create custom data collator that preserves seq_ids
    class EntropyDataCollator(DataCollatorWithPadding):
        """Custom data collator that keeps seq_ids as is without trying to tensorize them."""

        def __call__(self, features):
            # Extract seq_ids before handling the rest
            seq_ids = [f.pop("seq_id") for f in features] if "seq_id" in features[0] else None

            # Process the remaining features normally (convert to tensors, pad, etc.)
            batch = super().__call__(features)

            # Add seq_ids back to the batch without tensorizing
            if seq_ids:
                batch["seq_id"] = seq_ids

            return batch

    class DPODataCollator(DataCollatorWithPadding):
        """Custom data collator that keeps seq_ids as is without trying to tensorize them."""

        def __call__(self, features):
            # Extract seq_ids before handling the rest
            seq_id_ws = [f.pop("seq_id_w") for f in features] if "seq_id_w" in features[0] else None

            seq_id_ls = [f.pop("seq_id_l") for f in features] if "seq_id_l" in features[0] else None
            # Process the remaining features normally (convert to tensors, pad, etc.)
            batch = super().__call__(features)

            # Add seq_ids back to the batch without tensorizing
            if seq_id_ws:
                batch["seq_id_w"] = seq_id_ws
            if seq_id_ls:
                batch["seq_id_l"] = seq_id_ls

            return batch

    if loss == "dpo":
        data_collator = DPODataCollator(tokenizer=tokenizer)
    elif loss == "entropy":
        data_collator = EntropyDataCollator(tokenizer=tokenizer)
    else:
        raise ValueError(f"Invalid loss function: {loss}")

    # 7. Decide which Trainer
    if loss == "dpo":
        trainer_cls = ESMDirectPreferenceTrainer
    elif loss == "entropy":
        trainer_cls = ESMEntropyTrainer
    else:
        raise ValueError(f"Invalid loss function: {loss}")

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
        compute_metrics=compute_evaluation_function,
        callbacks=[logging_callback],
    )

    logging.info(f"Training model")
    trainer.train()
    logging.info(f"Training complete")

    # Return model, tokenizer, plus the trainer log
    return tokenizer, model, trainer.state.log_history


def load_esm_model(
    checkpoint,
    half_precision=False,
    train_full=False,
    deepspeed=False,
):
    """
    Load an ESM model for masked language modeling to calculate mutation scores.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

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


def parse_mutations(seq_id):
    """
    Parse mutations from a sequence ID (e.g., "A132W_S155G").

    Returns:
        List of tuples (wt_aa, locus (1-indexed), mut_aa)
    """
    import re

    if not seq_id:
        return []

    mutations = []
    for mut in seq_id.split("_"):
        match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
        if match:
            wt_aa, pos_str, mut_aa = match.groups()
            pos = int(pos_str)  # Leave 1-indexed
            mutations.append((wt_aa, pos, mut_aa))
    return mutations


def calculate_log_wt_marginal_from_logits(single_protein_logits, seq_id, tokenizer):
    """Calculate the WT marginal score from a protein logits for a seq id."""
    sequence_score = torch.tensor(0.0, device=single_protein_logits.device, requires_grad=True)
    if not seq_id or seq_id == "WT":
        return sequence_score

    mutations = parse_mutations(seq_id)
    if not mutations:
        return sequence_score

    for wt_aa, locus, mut_aa in mutations:
        # pos is already 1-indexed which handles <cls> token

        # Get probabilities for this position
        log_probs = F.log_softmax(single_protein_logits[locus], dim=0)

        # Get token IDs for wild-type and mutant amino acids
        wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)

        # Calculate score for this mutation
        log_wt_prob = log_probs[wt_token_id]
        log_mut_prob = log_probs[mut_token_id]
        score = log_mut_prob - log_wt_prob

        # TODO: For multi mutants, can we add in log space? Or should we add in linear space?
        sequence_score = sequence_score + score

    return sequence_score


class ESMDirectPreferenceTrainer(Trainer):
    # Override _prepare_inputs to preserve extra keys (e.g. "seq_ids")
    def _prepare_inputs(self, inputs):
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to handle our custom inputs and return predictions"""
        # Get seq_ids but keep them in inputs as compute_loss needs them
        seq_id_w = inputs.get("seq_id_w", None)
        seq_id_l = inputs.get("seq_id_l", None)

        # Calculate loss and get outputs
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Suppose we have the same batch size = N
        dummy_labels = torch.zeros((len(seq_id_w)), dtype=torch.long).to(loss.device)

        # During evaluation, we want to return the scores
        return (
            loss.detach(),
            {
                "log_score_w": torch.stack(outputs["log_score_w"]).detach(),  # type: ignore[reportArgumentType]
                "log_score_l": torch.stack(outputs["log_score_l"]).detach(),  # type: ignore[reportArgumentType]
            },
            # Labels aren't needed for ranking accuracy, but "None" labels won't have compute_accuracy called.
            dummy_labels,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        necessary_keys = ["seq_id_w", "seq_id_l"]
        if not all(key in inputs for key in necessary_keys):
            raise ValueError(
                f"Missing keys in inputs, expected {necessary_keys}, only found {inputs.keys()}"
            )

        # Remove extra keys so they are not passed to the model's forward
        seq_id_w_list = inputs.pop("seq_id_w")
        seq_id_l_list = inputs.pop("seq_id_l")

        # Single forward pass for the entire batch
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Extract the wt marginal scores.
        log_score_w_list = []
        log_score_l_list = []
        assert logits.shape[0] == len(
            seq_id_w_list
        ), f"Batch size numbers are inconsistent... {logits.shape[0]} {len(seq_id_w_list)}"
        for within_batch_idx in range(logits.shape[0]):
            seq_id_w = seq_id_w_list[within_batch_idx]
            seq_id_l = seq_id_l_list[within_batch_idx]

            log_score_w_list.append(
                calculate_log_wt_marginal_from_logits(
                    logits[within_batch_idx], seq_id_w, self.tokenizer
                )
            )
            log_score_l_list.append(
                calculate_log_wt_marginal_from_logits(
                    logits[within_batch_idx], seq_id_l, self.tokenizer
                )
            )

        # Calculate the DPO loss with ESM3's temperature scaling
        beta = 0.05  # Temperature parameter from ESM3 paper
        log_ratios = [
            (log_score_w - log_score_l) * beta
            for log_score_w, log_score_l in zip(log_score_w_list, log_score_l_list)
        ]
        losses = [F.softplus(-log_ratio) for log_ratio in log_ratios]
        loss = torch.stack(losses).mean()

        outputs = {
            "loss": loss,
            "log_score_w": log_score_w_list,
            "log_score_l": log_score_l_list,
        }

        return (loss, outputs) if return_outputs else loss


class ESMEntropyTrainer(Trainer):
    # Override _prepare_inputs to preserve extra keys (e.g. "seq_ids")
    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure that "seq_ids" and "labels" are present.
        if "seq_id" not in inputs:
            raise ValueError(
                f"seq_id is required for ESMMutationScoreTrainer. Inputs had {inputs.keys()}"
            )
        if "labels" not in inputs:
            raise ValueError(
                f"labels is required for ESMMutationScoreTrainer. Inputs had {inputs.keys()}"
            )

        # Remove extra keys so they are not passed to the model's forward
        seq_ids = inputs.pop("seq_id")
        labels = inputs.pop("labels")

        # Single forward pass for the entire batch
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Calculate scores for each sequence
        scores_list = []
        for i, seq_id in enumerate(seq_ids):
            sequence_score = torch.zeros(
                1, device=logits.device, requires_grad=True
            )  # Initialize with requires_grad=True
            # Parse mutations from seq_id (e.g., "A132W_S155G")
            if seq_id and seq_id != "WT":
                mutations = parse_mutations(seq_id)
                if mutations:
                    for wt_aa, locus, mut_aa in mutations:
                        # Locus is already 1-indexed which handles <cls> token

                        # Get probabilities for this position
                        logits_at_pos = logits[i, locus]  # Keep raw logits
                        log_probs = torch.log_softmax(logits_at_pos, dim=0)

                        # Get token IDs for wild-type and mutant amino acids
                        wt_token_id = self.tokenizer.convert_tokens_to_ids(wt_aa)
                        mut_token_id = self.tokenizer.convert_tokens_to_ids(mut_aa)

                        # Calculate score for this mutation
                        log_wt_prob = log_probs[wt_token_id]
                        log_mut_prob = log_probs[mut_token_id]
                        score = log_mut_prob - log_wt_prob
                        sequence_score = sequence_score + score

            scores_list.append(sequence_score)

        scores = torch.stack(scores_list)

        # Convert labels to tensor if needed and ensure same device
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float, device=scores.device)
        if labels.device != scores.device:
            labels = labels.to(scores.device)

        loss = full_ranking_bce(scores, labels)
        outputs = {"loss": loss, "scores": scores}

        return (loss, outputs) if return_outputs else loss


def compute_spearmanr(eval_pred):
    predictions, labels = eval_pred
    # If predictions are scores from our custom trainer
    if isinstance(predictions, dict) and "scores" in predictions:
        predictions = predictions["scores"]
    rho, _ = spearmanr(predictions, labels)
    return {"spearmanr": rho}


def compute_ranking_accuracy(eval_pred):
    """Compute the accuracy of preference rankings (seq_id_w > seq_id_l)"""
    # HF passes in a namedtuple: (predictions, label_ids, something_else)
    predictions = eval_pred.predictions
    # label_ids = eval_pred.label_ids     # We can ignore if we want
    if isinstance(predictions, dict):
        winning_scores = predictions["log_score_w"]
        losing_scores = predictions["log_score_l"]
    else:
        winning_scores, losing_scores = predictions

    # If they are PyTorch tensors, convert them to NumPy
    if torch.is_tensor(winning_scores):
        winning_scores = winning_scores.cpu().numpy()
    if torch.is_tensor(losing_scores):
        losing_scores = losing_scores.cpu().numpy()

    # Now do the boolean comparison in NumPy
    correct_rankings = np.mean((winning_scores > losing_scores).astype(np.float32))

    return {"ranking_accuracy": float(correct_rankings)}


def score_sequences(model, tokenizer, wt_aa_seq, seq_ids):
    """Score a list of sequences using the trained model.

    Args:
        model: Trained ESM model
        tokenizer: ESM tokenizer
        wt_aa_seq: Wild-type amino acid sequence
        seq_ids

    Returns:
        DataFrame with columns:
        - seq_id: Sequence identifier
        - sequence: Full amino acid sequence
        - log_wt_marginal: Sum of wild-type amino acid log probabilities
    """
    import pandas as pd
    import torch
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer([wt_aa_seq], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(logits.shape)

    results = []

    # Process in batches
    for seq_id in seq_ids:
        sequence = seq_id_to_seq(wt_aa_seq, seq_id)
        sequence_score = 0
        for wt_aa, locus, mut_aa in parse_mutations(seq_id):
            # Locus is already 1-indexed which handles <cls> token

            # Get probabilities for this position
            log_probs = F.log_softmax(logits[0, locus], dim=0)

            # Get token IDs for wild-type and mutant amino acids
            wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)

            # Calculate score for this mutation
            log_wt_prob = log_probs[wt_token_id]
            log_mut_prob = log_probs[mut_token_id]
            score = log_mut_prob - log_wt_prob
            sequence_score = sequence_score + score

        results.append(
            {
                "seq_id": seq_id,
                "sequence": sequence,
                "log_wt_marginal": float(sequence_score),
            }
        )

    return pd.DataFrame(results)
