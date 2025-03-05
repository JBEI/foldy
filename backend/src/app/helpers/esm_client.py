from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional
import json


class FoldyESMClient(ABC):
    """Interface for ESM model clients."""

    @classmethod
    def get_client(cls, model_name: str) -> "FoldyESMClient":
        """Factory method to create appropriate ESM client based on model name."""
        if model_name.startswith("esmc"):
            return FoldyESMCClient(model_name)
        elif model_name.startswith("esm3"):
            return FoldyESM3Client(model_name)
        elif model_name.startswith("esm1") or model_name.startswith("esm2"):
            return FoldyESM1and2Client(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    @abstractmethod
    def embed(self, sequence: str, pdb_file_path: Optional[str] = None) -> List[float]:
        """Get embedding for a sequence."""
        pass

    @abstractmethod
    def get_logits(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Get logits for a sequence."""
        pass


class FoldyESMCClient(FoldyESMClient):
    def __init__(self, model_name: str):
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = ESMC.from_pretrained(model_name).to(device)
        self.device = device

    def get_esm_protein_tensor(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ):  # -> torch.Tensor:
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if pdb_file_path:
            raise ValueError("ESM-C does not support PDB-based embeddings")
        protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def embed(self, sequence: str, pdb_file_path: Optional[str] = None) -> List[float]:
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        protein_tensor = self.get_esm_protein_tensor(sequence, pdb_file_path)
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=False, return_embeddings=True)
        )
        # Average across residue dimension
        embedding = logits_output.embeddings.mean(dim=1).squeeze(0)
        return embedding.tolist()

    def get_logits(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex
        from esm.utils.constants import esm3 as esm3_constants

        protein_tensor = self.get_esm_protein_tensor(sequence, pdb_file_path)
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)
        )

        sequence_probs = torch.softmax(logits_output.logits.sequence, dim=-1)
        melted_rows = []

        for pos in range(1, len(sequence) + 1):  # 1-based positions
            wt_aa = sequence[pos - 1]
            probs = sequence_probs[0, pos, :].tolist()

            for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                prob = probs[vocab_idx]
                seq_id = f"{wt_aa}{pos}{vocab_char}"
                melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)


class FoldyESM3Client(FoldyESMClient):
    def __init__(self, model_name: str):
        import torch
        from esm.models.esm3 import ESM3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = ESM3.from_pretrained(model_name).to(device)
        self.device = device

    def get_esm_protein_tensor(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ):  # -> torch.Tensor:
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if pdb_file_path:
            protein_complex = ProteinComplex.from_pdb(path=pdb_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            protein = ESMProtein(sequence=sequence)

        protein_tensor = self.client.encode(protein)

    # Implementation similar to ESMCClient
    embed = FoldyESMCClient.embed
    get_logits = FoldyESMCClient.get_logits


class FoldyESM1and2Client(FoldyESMClient):
    def __init__(self, model_name: str):
        import torch

        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", model_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Set to evaluation mode

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def embed(self, sequence: str, pdb_file_path: Optional[str] = None) -> List[float]:
        import torch

        if pdb_file_path:
            raise ValueError("ESM1 and 2 do not support PDB-based embeddings")

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        token_embeddings = results["representations"][33]

        # Remove cls and eos tokens, then average
        protein_embedding = token_embeddings[0, 1:-1].mean(0)
        return protein_embedding.cpu().tolist()

    def get_logits(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        import torch

        if pdb_file_path:
            raise ValueError("ESM1 and 2 do not support PDB-based logits")

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tokens, repr_layers=[33])["logits"]

        sequence_probs = torch.softmax(logits, dim=-1)
        melted_rows = []

        for pos in range(1, len(sequence) + 1):  # 1-based positions
            wt_aa = sequence[pos - 1]
            probs = sequence_probs[0, pos, :].cpu().tolist()

            for vocab_idx, vocab_char in enumerate(self.alphabet.all_toks):
                if (
                    vocab_char in self.alphabet.standard_toks
                ):  # Only include standard amino acids
                    prob = probs[vocab_idx]
                    seq_id = f"{wt_aa}{pos}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)
