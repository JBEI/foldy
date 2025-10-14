from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from app.helpers.esm_util import get_naturalness


@pytest.fixture
def mock_esm_setup():
    """Setup common mocks for ESM-related tests"""
    with (
        patch("torch.device") as mock_device,
        patch("esm.models.esmc.ESMC") as mock_ESMC,
        patch("esm.models.esm3.ESM3") as mock_ESM3,
    ):

        # Mock device setup
        mock_device.return_value = "cpu"

        # Create mock client
        mock_client = Mock()
        mock_ESMC.from_pretrained.return_value.to.return_value = mock_client

        mock_esm3_client = Mock()
        mock_ESM3.from_pretrained.return_value.to.return_value = mock_esm3_client

        # Create mock logits output
        # Shape: [batch=1, sequence_length=7, vocab_size=33]
        mock_logits = torch.randn(1, 7, 33)  # 5 amino acids + 2 special tokens
        mock_logits_output = Mock()
        mock_logits_output.logits.sequence = mock_logits.clone()

        # Setup client encode and logits methods
        mock_client.encode.return_value = Mock()
        mock_client.logits.return_value = mock_logits_output

        # Setup client encode and logits methods
        mock_esm3_client.encode.return_value = Mock()
        mock_esm3_client.logits.return_value = mock_logits_output

        yield {
            "device": mock_device,
            "ESMC": mock_ESMC,
            "ESM3": mock_ESM3,
            "client": mock_client,
            "logits": mock_logits,
        }


def test_get_naturalness_basic(mock_esm_setup):
    # Test sequence
    wt_aa_seq = "ABCDE"  # 5 amino acids

    # Call function
    logits_json, melted_df = get_naturalness(wt_aa_seq, "esmc_mock_model")

    # Verify basic calls
    mock_esm_setup["ESMC"].from_pretrained.assert_called_once_with("esmc_mock_model")

    # Verify shape of output DataFrame
    assert isinstance(melted_df, pd.DataFrame)
    assert set(melted_df.columns) == {
        "seq_id",
        "probability",
        "locus",
        "wt_probability",
        "wt_marginal",
    }

    # Each position should have entries for all possible amino acids
    expected_positions = len(wt_aa_seq)
    expected_rows = expected_positions * 33  # 33 amino acids in vocab per position
    assert len(melted_df) == expected_rows


def test_get_naturalness_wt_marginal_calculation(mock_esm_setup):
    wt_aa_seq = "ABC"

    # Create predictable logits for easier testing
    # Shape: [batch=1, sequence_length=5, vocab_size=33]
    mock_logits = torch.zeros(1, 5, 33)  # 3 amino acids + 2 special tokens
    # Set some known probabilities after softmax
    mock_logits[0, 1:4, 0] = 0  # probability for first amino acid in vocab
    mock_logits_output = Mock()
    mock_logits_output.logits.sequence = mock_logits
    mock_esm_setup["client"].logits.return_value = mock_logits_output

    # Call function
    _, melted_df = get_naturalness(wt_aa_seq, "esmc_mock_model")

    # Verify wt_marginal calculations
    # Filter for a specific position
    pos1_data = melted_df[melted_df.locus == 1]
    assert all(pos1_data.wt_marginal.notna())  # All wt_marginal values should be calculated


def test_get_naturalness_error_handling(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "TOOLONG"  # 7 amino acids, but mock returns logits for 5

    with pytest.raises(IndexError, match="index 7 is out of bounds"):
        get_naturalness(wt_aa_seq, "esmc_mock_model")


def test_add_pdb_file_path_fails_for_esmc(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "ABCDE"
    pdb_file_path = "app/tests/testdata/rubisco-boltz.pdb"

    with pytest.raises(ValueError, match="does not support PDB"):
        get_naturalness(wt_aa_seq, "esmc_mock_model", cif_file_path=pdb_file_path)


def test_add_pdb_file_path_works_for_esm3(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "ABCDE"
    pdb_file_path = "app/tests/testdata/rubisco-boltz.pdb"

    logits_json, melted_df = get_naturalness(
        wt_aa_seq, "esm3_mock_model", cif_file_path=pdb_file_path
    )
    assert logits_json is not None
    assert melted_df is not None
