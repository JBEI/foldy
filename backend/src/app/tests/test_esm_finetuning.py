import pytest
from unittest.mock import Mock, patch
import torch
import pandas as pd
import numpy as np
from app.helpers.finetuning.training import (
    train_per_protein,
    load_esm_model,
    full_ranking_bce,
    calculate_log_wt_marginal_from_logits,
    score_sequences,
)
import types


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


def test_esm_forward_pass():
    """
    Test the model can perform a single forward pass with dummy data.
    """
    checkpoint = "facebook/esm2_t6_8M_UR50D"
    model, tokenizer = load_esm_model(
        checkpoint=checkpoint,
        half_precision=False,
        train_full=False,
        deepspeed=False,
    )

    # Dummy batch of 2 sequences, tokenized
    sequences = ["ACD", "MKT"]
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits  # shape [batch_size]
    assert logits.shape == (2, 5, 33), f"Expected logits shape (2), got {logits.shape}."


@pytest.mark.parametrize("loss", ["dpo", "entropy"])
def test_train_per_protein_smoke(loss):
    """
    A 'smoke test' that runs the training loop with a tiny dataset
    for 1 epoch, just to check no major errors occur.
    """
    # Minimal checkpoint (small ESM):
    checkpoint = "facebook/esm2_t6_8M_UR50D"

    # Tiny dataset
    if loss == "dpo":
        data = {
            "sequence": ["ACDE", "ACDF", "ACDG", "XXXX"],
            "seq_id_w": ["WT", "MUT1", "MUT2", "WT"],
            "seq_id_l": ["WT", "MUT1", "MUT2", "WT"],
        }
    else:
        data = {
            "sequence": ["ACDE", "ACDF", "ACDG", "XXXX"],
            "seq_id": ["WT", "E4F", "E4G", "A1X_C2X_D3X_E4X"],
            "label": [0, 1, 0, 1],
        }
    df = pd.DataFrame(data)

    # Split into train/valid
    train_df = df.iloc[:2].reset_index(drop=True)
    valid_df = df.iloc[2:].reset_index(drop=True)

    # We only do 1 epoch, tiny batch, etc.
    tokenizer, model, history = train_per_protein(
        checkpoint=checkpoint,
        train_df=train_df,
        valid_df=valid_df,
        device="cpu",
        loss=loss,
        train_batch_size=1,
        grad_accum_steps=1,
        val_batch_size=1,
        epochs=1,
        learning_rate=1e-4,
        seed=42,
        deepspeed_config=None,  # no DS for testing
        mixed_precision=False,  # keep it simple for testing
        train_full=False,
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


def test_calculate_log_wt_marginal_from_logits():
    """
    Test that calculate_log_wt_marginal_from_logits handles gradients correctly.
    """

    # Mock tokenizer
    class MockTokenizer:
        def convert_tokens_to_ids(self, token):
            # Simple mapping for test
            return {"A": 0, "W": 1}[token]

    tokenizer = MockTokenizer()

    # Create dummy logits tensor that requires gradients
    # Shape: [seq_len, vocab_size]
    logits = torch.zeros((5, 2), requires_grad=True)

    # Test with a double mutation
    seq_id = "A1W_A2W"  # Two A->W mutations

    # This should not raise a RuntimeError about in-place operations
    score = calculate_log_wt_marginal_from_logits(logits, seq_id, tokenizer)

    # Verify we can backpropagate through the score
    score.backward()

    assert score.requires_grad, "Score should require gradients"


def test_score_sequences_basic():
    """Test basic functionality of score_sequences with a simple mutation."""

    # Mock model and tokenizer
    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, **kwargs):
            # Return mock logits that will give predictable scores
            # Shape: [batch_size=1, seq_len=5, vocab_size=33]
            logits = torch.zeros(1, 5, 33)
            return types.SimpleNamespace(logits=logits)

    class MockTokenizer:
        def __call__(
            self, sequences, padding=True, truncation=True, return_tensors="pt"
        ):
            return {"input_ids": torch.zeros(1, 5), "attention_mask": torch.ones(1, 5)}

        def convert_tokens_to_ids(self, token):
            # Simple mapping for test
            return {"A": 0, "W": 1}[token]

    model = MockModel()
    tokenizer = MockTokenizer()
    wt_aa_seq = "AAAAA"  # 5 alanines
    seq_ids = ["WT", "A1W"]  # Wild-type and one mutation

    df = score_sequences(model, tokenizer, wt_aa_seq, seq_ids)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"seq_id", "sequence", "log_wt_marginal"}
    assert len(df) == 2  # Should have 2 rows for WT and A1W

    # Check model was put in eval mode
    assert model.eval_called


def test_score_sequences_multiple_mutations():
    """Test scoring with multiple mutations in a single sequence."""

    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            pass

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, **kwargs):
            # Create logits where W is always more favorable than A
            logits = torch.zeros(1, 5, 33)
            logits[:, :, 1] = 1.0  # Make W (index 1) more favorable
            return types.SimpleNamespace(logits=logits)

    class MockTokenizer:
        def __call__(
            self, sequences, padding=True, truncation=True, return_tensors="pt"
        ):
            return {"input_ids": torch.zeros(1, 5), "attention_mask": torch.ones(1, 5)}

        def convert_tokens_to_ids(self, token):
            return {"A": 0, "W": 1}[token]

    model = MockModel()
    tokenizer = MockTokenizer()
    wt_aa_seq = "AAAAA"
    seq_ids = ["WT", "A1W_A2W"]  # Wild-type and double mutation

    df = score_sequences(model, tokenizer, wt_aa_seq, seq_ids)

    # Check we have the expected number of rows
    assert len(df) == 2

    # The double mutation should have a higher score than WT
    wt_score = df[df["seq_id"] == "WT"]["log_wt_marginal"].iloc[0]
    mut_score = df[df["seq_id"] == "A1W_A2W"]["log_wt_marginal"].iloc[0]
    assert mut_score > wt_score


def test_score_sequences_empty_seq_ids():
    """Test behavior with empty sequence ID list."""

    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            pass

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=torch.zeros(1, 5, 33))

    class MockTokenizer:
        def __call__(
            self, sequences, padding=True, truncation=True, return_tensors="pt"
        ):
            return {"input_ids": torch.zeros(1, 5), "attention_mask": torch.ones(1, 5)}

        def convert_tokens_to_ids(self, token):
            return {"A": 0, "W": 1}[token]

    model = MockModel()
    tokenizer = MockTokenizer()
    wt_aa_seq = "AAAAA"
    seq_ids = []

    df = score_sequences(model, tokenizer, wt_aa_seq, seq_ids)

    # Should return empty DataFrame
    assert isinstance(df, pd.DataFrame)
    # assert set(df.columns) == {"seq_id", "sequence", "log_wt_marginal"}
    assert len(df) == 0
