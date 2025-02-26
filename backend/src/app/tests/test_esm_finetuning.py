import pytest
from unittest.mock import Mock, patch
import torch
import pandas as pd
import numpy as np
from app.helpers.finetuning.training import train_per_protein


def test_full_ranking_bce_basic():
    """
    Test the ranking loss on a small, controlled input.
    """
    # preds: shape (3,), targets: shape (3,)
    preds = torch.tensor([0.0, 0.5, 1.0])
    targets = torch.tensor([0.0, 0.0, 1.0])

    loss = full_ranking_bce(preds, targets)

    # Loss should be a single scalar
    assert loss.dim() == 0, "Loss should be scalar."
    assert loss >= 0, "Loss should be non-negative."

    # We don't necessarily know the exact value off-hand,
    # but we can check it's not NaN or something unexpected.
    assert not torch.isnan(loss), "Loss should not be NaN."


def test_full_ranking_bce_batch_shapes():
    """
    Test the ranking loss with a larger batch to check shape logic.
    """
    batch_size = 5
    preds = torch.randn(batch_size)  # e.g., [0.2, -0.1, 0.5, ...]
    targets = torch.rand(batch_size)  # e.g., [0.4, 0.1, 0.9, ...]

    loss = full_ranking_bce(preds, targets)

    assert loss.dim() == 0, "Loss should be scalar even for a batch."
    assert loss >= 0, "Loss should be non-negative."


@pytest.mark.parametrize("train_full", [False, True])
def test_load_esm_model_basic(train_full):
    """
    Smoke test to verify we can load the base ESM model
    and optionally freeze or unfreeze layers.
    """
    checkpoint = "facebook/esm2_t6_8M_UR50D"  # small model
    num_labels = 2  # classification with 2 classes

    model, tokenizer = load_esm_model(
        checkpoint=checkpoint,
        num_labels=num_labels,
        half_precision=False,  # for local testing, avoid FP16 to keep it simple
        train_full=train_full,
        deepspeed=False,
    )

    # Check that the model has the correct number of labels
    # If you used AutoModelForSequenceClassification, there's typically
    # model.classifier.out_proj.weight with shape [num_labels, hidden_dim].
    # But check your architecture for the actual attribute name.
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]
    assert len(classifier_params) > 0, "We should have a classifier head in the model."

    # Check requires_grad
    # If train_full=False, we expect mostly frozen base params, except LoRA + classifier
    if not train_full:
        # At least the classifier or LoRA should be trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0, "LoRA or classifier params should be unfrozen."
    else:
        # If train_full=True, we expect the base encoder to be trainable
        base = [
            n for n, p in model.named_parameters() if "esm" in n and p.requires_grad
        ]
        assert (
            len(base) > 0
        ), "Base ESM parameters should be trainable when train_full=True."


def test_esm_forward_pass():
    """
    Test the model can perform a single forward pass with dummy data.
    """
    checkpoint = "facebook/esm2_t6_8M_UR50D"
    model, tokenizer = load_esm_model(
        checkpoint=checkpoint,
        num_labels=2,
        half_precision=False,
        train_full=False,
        deepspeed=False,
    )

    # Dummy batch of 2 sequences, tokenized
    sequences = ["ACD", "MKT"]
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits  # shape [batch_size, num_labels]
    assert logits.shape == (2, 2), f"Expected logits shape (2,2), got {logits.shape}."


@pytest.mark.parametrize("use_ranking_loss", [False, True])
def test_train_per_protein_smoke(use_ranking_loss):
    """
    A 'smoke test' that runs the training loop with a tiny dataset
    for 1 epoch, just to check no major errors occur.
    """
    # Minimal checkpoint (small ESM):
    checkpoint = "facebook/esm2_t6_8M_UR50D"

    # Tiny dataset
    # We'll do a trivial "classification" with label 0 or 1
    data = {"sequence": ["ACDE", "ACDF", "ACDG", "XXXX"], "label": [0, 1, 0, 1]}
    df = pd.DataFrame(data)

    # Split into train/valid
    train_df = df.iloc[:2].reset_index(drop=True)
    valid_df = df.iloc[2:].reset_index(drop=True)

    # We only do 1 epoch, tiny batch, etc.
    tokenizer, model, history = train_per_protein(
        checkpoint=checkpoint,
        train_df=train_df,
        valid_df=valid_df,
        num_labels=2,  # binary classification
        use_ranking_loss=use_ranking_loss,
        train_batch_size=1,
        grad_accum_steps=1,
        val_batch_size=1,
        epochs=1,
        learning_rate=1e-4,
        seed=42,
        deepspeed_config=None,  # no DS for testing
        mixed_precision=False,  # keep it simple for testing
        train_full=False,
        gpu_id=0,
    )

    # Basic assertions
    assert isinstance(history, list), "History should be a list of log entries."
    assert len(history) > 0, "We expect some logging from 1 epoch of training."

    # Check final entry has 'epoch'==1
    final_entry = history[-1]
    assert final_entry.get("epoch", None) == 1, "We trained for 1 epoch only."

    # Confirm the model was updated
    # e.g., we can check the classifier's weight changed from initial
    # Not strictly necessary, but good to know training happened.
    # For more robust test, you'd store old weights and compare.
    # We'll skip that detail here.
