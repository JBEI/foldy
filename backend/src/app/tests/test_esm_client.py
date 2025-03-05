import pytest
from unittest.mock import Mock, patch
import torch
import pandas as pd
import numpy as np
from app.helpers.esm_client import (
    FoldyESMClient,
    FoldyESMCClient,
    FoldyESM1and2Client,
    FoldyESM3Client,
)

# Test sequences
TEST_SEQUENCE = "MGSSHHHHHHSSGLVPRGSHM"
TEST_PDB_PATH = "app/tests/testdata/rubisco-boltz.pdb"
ESM3_VOCAB_SIZE = 64


@pytest.fixture
def mock_torch_device():
    with patch("torch.device") as mock_device, patch(
        "torch.cuda.is_available", return_value=False
    ):
        mock_device.return_value = "cpu"
        yield mock_device


@pytest.fixture
def mock_esmc_client():
    with patch("esm.models.esmc.ESMC") as MockESMC:

        # Create mock client
        mock_client = Mock()
        MockESMC.from_pretrained.return_value = mock_client
        mock_client.to.return_value = mock_client

        # Mock encode method
        mock_client.encode.return_value = Mock()

        # Mock logits method with embeddings
        mock_embeddings = torch.randn(
            1, len(TEST_SEQUENCE), 1280
        )  # Example embedding size
        mock_logits_output = Mock(embeddings=mock_embeddings)
        mock_client.logits.return_value = mock_logits_output

        yield mock_client


@pytest.fixture
def mock_esm3_client():
    with patch("esm.models.esm3.ESM3") as MockESM3:

        # Create mock client
        mock_client = Mock()
        MockESM3.from_pretrained.return_value = mock_client
        mock_client.to.return_value = mock_client

        # Mock encode method
        mock_client.encode.return_value = Mock()

        # Mock logits method with embeddings
        mock_embeddings = torch.randn(
            1, len(TEST_SEQUENCE), 1280
        )  # Example embedding size
        mock_logits_output = Mock(embeddings=mock_embeddings)
        mock_client.logits.return_value = mock_logits_output

        yield mock_client


@pytest.fixture
def mock_esm2_hub():
    with patch("torch.hub.load") as mock_hub:
        # Create mock model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_converter = Mock()

        # Setup returns
        mock_hub.return_value = (mock_model, mock_alphabet)
        mock_alphabet.get_batch_converter.return_value = mock_converter
        mock_converter.return_value = (
            None,
            None,
            torch.zeros((1, len(TEST_SEQUENCE) + 2)),
        )

        # Mock model forward pass
        mock_representations = {33: torch.randn(1, len(TEST_SEQUENCE) + 2, 1280)}
        mock_logits = torch.randn(1, len(TEST_SEQUENCE) + 2, 20)  # 20 amino acids
        mock_model.return_value = {
            "representations": mock_representations,
            "logits": mock_logits,
        }

        # Setup alphabet tokens
        mock_alphabet.all_toks = list("ACDEFGHIKLMNPQRSTVWY")
        mock_alphabet.standard_toks = list("ACDEFGHIKLMNPQRSTVWY")

        yield mock_model, mock_alphabet


def test_get_client_invalid():
    with pytest.raises(ValueError):
        FoldyESMClient.get_client("invalid_model")


def test_esmc_embed(mock_torch_device, mock_esmc_client):
    client = FoldyESMClient.get_client("esmc_t36_3B_UR50D")
    embedding = client.embed(TEST_SEQUENCE)

    assert isinstance(embedding, list)
    assert len(embedding) == 1280  # Expected embedding dimension


def test_esmc_embed_with_pdb_fails(mock_torch_device, mock_esmc_client):
    client = FoldyESMClient.get_client("esmc_t36_3B_UR50D")
    with pytest.raises(ValueError, match="ESM-C does not support PDB-based embeddings"):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esm3_embed_with_pdb_succeeds(mock_torch_device, mock_esm3_client):
    client = FoldyESMClient.get_client("esm3_t36_3B_UR50D")
    embedding = client.embed(TEST_SEQUENCE, TEST_PDB_PATH)

    assert isinstance(embedding, list)
    assert len(embedding) == 1280


def test_esm2_embed_with_pdb_fails(mock_torch_device, mock_esm2_hub):
    client = FoldyESMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError, match="do not support PDB-based embeddings"):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esmc_get_logits(mock_torch_device, mock_esmc_client):
    # Mock the logits output
    sequence_logits = torch.randn(
        1, len(TEST_SEQUENCE) + 2, ESM3_VOCAB_SIZE
    )  # batch, seq_len + special tokens, vocab_size
    mock_esmc_client.logits.return_value = Mock(logits=Mock(sequence=sequence_logits))

    client = FoldyESMClient.get_client("esmc_t36_3B_UR50D")
    df = client.get_logits(TEST_SEQUENCE)

    assert isinstance(df, pd.DataFrame)
    assert "seq_id" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0


def test_esm2_embed(mock_torch_device, mock_esm2_hub):
    client = FoldyESMClient.get_client("esm2_t33_650M_UR50D")
    embedding = client.embed(TEST_SEQUENCE)

    assert isinstance(embedding, list)
    assert len(embedding) == 1280


def test_esm2_embed_with_pdb(mock_torch_device, mock_esm2_hub):
    client = FoldyESMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esm2_get_logits(mock_torch_device, mock_esm2_hub):
    client = FoldyESMClient.get_client("esm2_t33_650M_UR50D")
    df = client.get_logits(TEST_SEQUENCE)

    assert isinstance(df, pd.DataFrame)
    assert "seq_id" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0


def test_esm2_get_logits_with_pdb(mock_torch_device, mock_esm2_hub):
    client = FoldyESMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError):
        client.get_logits(TEST_SEQUENCE, TEST_PDB_PATH)


# Helper function to verify DataFrame structure
def verify_logits_df_structure(df: pd.DataFrame, sequence: str):
    assert all(col in df.columns for col in ["seq_id", "probability"])
    assert len(df) > 0
    # Verify seq_id format (e.g., "M1A" for mutation of M at position 1 to A)
    assert all(len(seq_id) >= 3 for seq_id in df["seq_id"])
    assert all(0 <= prob <= 1 for prob in df["probability"])
