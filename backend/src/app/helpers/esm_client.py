from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, cast
import json

# Type definitions for complex inputs
SequenceType = str
ComplexType = List[Tuple[str, str]]
SequenceOrComplexType = Union[SequenceType, ComplexType]

# Import for type checking only
if TYPE_CHECKING:
    import torch
    from esm.sdk.api import ESMProtein


class FoldyESMClient(ABC):
    """
    Interface for ESM model clients that provide embedding and logit functionality.
    
    This abstract base class defines the interface that all ESM client 
    implementations must follow.
    """

    @classmethod
    def get_client(cls, model_name: str) -> "FoldyESMClient":
        """
        Factory method to create appropriate ESM client based on model name.
        
        Args:
            model_name: Name of the ESM model to use
            
        Returns:
            An instance of the appropriate FoldyESMClient subclass
            
        Raises:
            ValueError: If model_name does not match any known model type
        """
        if model_name.startswith("esmc"):
            return FoldyESMCClient(model_name)
        elif model_name.startswith("esm3"):
            return FoldyESM3Client(model_name)
        elif model_name.startswith("esm1") or model_name.startswith("esm2"):
            return FoldyESM1and2Client(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    @abstractmethod
    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> List[float]:
        """
        Get embedding for a protein sequence or complex.
        
        Args:
            sequence_or_complex: Either a protein sequence string or a list of 
                                (chain_id, sequence) tuples for complexes
            pdb_file_path: Optional path to a PDB file for structure-aware models
            
        Returns:
            A list of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.
        
        Args:
            sequence_or_complex: Either a protein sequence string or a list of 
                                (chain_id, sequence) tuples for complexes
            pdb_file_path: Optional path to a PDB file for structure-aware models
            
        Returns:
            A pandas DataFrame with sequence logits in melted format with 
            columns 'seq_id' and 'probability'
        """
        pass


class FoldyESMCClient(FoldyESMClient):
    """
    ESM-C model client implementation for the foldy platform.
    
    Handles ESM-C specific operations including protein tensor creation, 
    embedding extraction, and logit computation.
    """
    
    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-C client with the specified model.
        
        Args:
            model_name: Name of the ESM-C model to load
        """
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = ESMC.from_pretrained(model_name).to(device)
        self.device = device

    def _get_esm_protein_tensor_for_sequence(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.
        
        Args:
            sequence: Protein sequence string
            pdb_file_path: Not supported for ESM-C
            
        Returns:
            Tensor representation of the protein
            
        Raises:
            ValueError: If pdb_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        if pdb_file_path:
            raise ValueError("ESM-C does not support PDB-based embeddings")
        protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, pdb_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.
        
        Args:
            complex_input: List of (chain_id, sequence) tuples
            pdb_file_path: Not supported for ESM-C
            
        Returns:
            Tensor representation of the protein complex
            
        Raises:
            ValueError: If pdb_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex
        from esm.utils.structure.protein_chain import ProteinChain

        if pdb_file_path:
            raise ValueError("ESM-C does not support PDB-based embeddings")

        chains = [
            ProteinChain(chain_id=chain_id, sequence=sequence)
            for chain_id, sequence in complex_input
        ]
        protein_complex = ProteinComplex.from_chains(chains)
        protein = ESMProtein.from_protein_complex(protein_complex)

        return self.client.encode(protein)

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> List[float]:
        """
        Get embedding for a protein sequence or complex.
        
        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            pdb_file_path: Optional path to a PDB file (not supported for ESM-C)
            
        Returns:
            A list of floats representing the embedding vector
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if isinstance(sequence_or_complex, list):
            protein_tensor = self._get_esm_protein_tensor_for_complex(
                sequence_or_complex, pdb_file_path
            )
        else:
            protein_tensor = self._get_esm_protein_tensor_for_sequence(
                sequence_or_complex, pdb_file_path
            )
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=False, return_embeddings=True)
        )
        # Average across residue dimension
        embedding = logits_output.embeddings.mean(dim=1).squeeze(0)
        return embedding.tolist()

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.
        
        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            pdb_file_path: Optional path to a PDB file (not supported for ESM-C)
            
        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex
        from esm.utils.constants import esm3 as esm3_constants

        if isinstance(sequence_or_complex, list):
            protein_tensor = self._get_esm_protein_tensor_for_complex(
                sequence_or_complex, pdb_file_path
            )
        else:
            protein_tensor = self._get_esm_protein_tensor_for_sequence(
                sequence_or_complex, pdb_file_path
            )
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)
        )

        sequence_probs = torch.softmax(logits_output.logits.sequence, dim=-1)
        melted_rows: List[Dict[str, Any]] = []

        if isinstance(sequence_or_complex, str):
            sequence = sequence_or_complex
            for pos in range(1, len(sequence) + 1):  # 1-based positions
                wt_aa = sequence[pos - 1]
                probs = sequence_probs[0, pos, :].tolist()

                for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                    prob = probs[vocab_idx]
                    seq_id = f"{wt_aa}{pos}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})
        else:
            # We don't know how this is formatted to start, so we just dump out the data.
            for idx in range(sequence_probs.shape[1]):
                probs = sequence_probs[0, idx, :].tolist()

                for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                    prob = probs[vocab_idx]
                    seq_id = f"{idx}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)


class FoldyESM3Client(FoldyESMClient):
    """
    ESM-3 model client implementation for the foldy platform.
    
    Handles ESM-3 specific operations including protein tensor creation.
    Uses the same embedding and logit computation as ESM-C.
    """
    
    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-3 client with the specified model.
        
        Args:
            model_name: Name of the ESM-3 model to load
        """
        import torch
        from esm.models.esm3 import ESM3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = ESM3.from_pretrained(model_name).to(device)
        self.device = device

    def _get_esm_protein_tensor_for_sequence(
        self, sequence: str, pdb_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.
        
        Args:
            sequence: Protein sequence string
            pdb_file_path: Optional path to a PDB file for structure-aware modeling
            
        Returns:
            Tensor representation of the protein
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if pdb_file_path:
            protein_complex = ProteinComplex.from_pdb(path=pdb_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, pdb_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.
        
        Args:
            complex_input: List of (chain_id, sequence) tuples
            pdb_file_path: Optional path to a PDB file for structure-aware modeling
            
        Returns:
            Tensor representation of the protein complex
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex
        from esm.utils.structure.protein_chain import ProteinChain

        if pdb_file_path:
            protein_complex = ProteinComplex.from_pdb(path=pdb_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            chains = [
                ProteinChain(chain_id=chain_id, sequence=sequence)
                for chain_id, sequence in complex_input
            ]
            protein_complex = ProteinComplex.from_chains(chains)
            protein = ESMProtein.from_protein_complex(protein_complex)

        return self.client.encode(protein)

    # Implementation similar to ESMCClient - reuse these methods
    embed = FoldyESMCClient.embed
    get_logits = FoldyESMCClient.get_logits


class FoldyESM1and2Client(FoldyESMClient):
    """
    ESM-1 and ESM-2 model client implementation for the foldy platform.
    
    Handles the older ESM-1 and ESM-2 models which have a different API
    compared to ESM-3 and ESM-C.
    """
    
    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-1/2 client with the specified model.
        
        Args:
            model_name: Name of the ESM-1 or ESM-2 model to load
        """
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

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> List[float]:
        """
        Get embedding for a protein sequence.
        
        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            pdb_file_path: Not supported for ESM-1/2
            
        Returns:
            A list of floats representing the embedding vector
            
        Raises:
            ValueError: If a complex or PDB file is provided (not supported)
        """
        import torch

        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        sequence = sequence_or_complex
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
        self,
        sequence_or_complex: SequenceOrComplexType,
        pdb_file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence.
        
        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            pdb_file_path: Not supported for ESM-1/2
            
        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'
            
        Raises:
            ValueError: If a complex or PDB file is provided (not supported)
        """
        import torch

        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        sequence = sequence_or_complex

        if pdb_file_path:
            raise ValueError("ESM1 and 2 do not support PDB-based logits")

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tokens, repr_layers=[33])["logits"]

        sequence_probs = torch.softmax(logits, dim=-1)
        melted_rows: List[Dict[str, Any]] = []

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
