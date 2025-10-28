import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

# Type definitions for complex inputs
SequenceType = str
ComplexType = List[Tuple[str, str]]
SequenceOrComplexType = Union[SequenceType, ComplexType]


# Import for type checking only
if TYPE_CHECKING:
    import torch
    from esm.sdk.api import ESMProtein


def recursive_cleanup(obj):
    """Recursively detach and delete tensors in a complex nested object."""
    import torch

    if isinstance(obj, torch.Tensor):
        # Detach and move to CPU before deletion
        if obj.is_cuda:
            obj = obj.detach().cpu()
        return None  # Return None to indicate this should be deleted

    elif hasattr(obj, "__dict__"):
        # For objects with attributes
        for attr_name, attr_value in list(obj.__dict__.items()):
            result = recursive_cleanup(attr_value)
            if result is None:
                delattr(obj, attr_name)
            else:
                setattr(obj, attr_name, result)

    elif isinstance(obj, dict):
        # For dictionaries
        for key in list(obj.keys()):
            result = recursive_cleanup(obj[key])
            if result is None:
                del obj[key]
            else:
                obj[key] = result

    elif isinstance(obj, (list, tuple)):
        # For lists/tuples
        new_obj = type(obj)(
            recursive_cleanup(item) for item in obj if recursive_cleanup(item) is not None
        )
        return new_obj if new_obj else None

    return obj  # Return object with cleaned up components


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
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file for structure-aware models
            extra_layers: List of layer indices to return embeddings for
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.
        """
        pass

    @abstractmethod
    def get_logits(
        self, sequence_or_complex: SequenceOrComplexType, cif_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file for structure-aware models

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

    def _pool_by_domains(
        self, hidden_states: "torch.Tensor", domain_boundaries: List[int]
    ) -> "torch.Tensor":
        """
        Pool hidden states by domains and concatenate the results.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
            domain_boundaries: List of domain boundary positions (0-indexed)

        Returns:
            Concatenated tensor of domain embeddings
        """
        import torch

        if not domain_boundaries:
            return hidden_states.mean(dim=-2)

        # Convert to sorted list and add 0 at the start and seq_len at the end
        boundaries = [0] + sorted(domain_boundaries) + [hidden_states.shape[1]]
        domain_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            domain_embedding = hidden_states[:, start_idx:end_idx, :].mean(dim=1)
            domain_embeddings.append(domain_embedding)

        return torch.cat(domain_embeddings, dim=-1)

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
        model = ESMC.from_pretrained(model_name).to(device)

        # --- Add this block: ensure dtype compatibility ---
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            if major < 8:  # Turing (RTX 20xx) or older → no bfloat16 support
                model = model.to(torch.float16)
        # ---------------------------------------------------

        print(f"[ESMC] Loaded {model_name} on {device} with dtype {next(model.parameters()).dtype}")
        self.client = model
        self.device = device

    def _get_esm_protein_tensor_for_sequence(
        self, sequence: str, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.

        Args:
            sequence: Protein sequence string
            cif_file_path: Not supported for ESM-C

        Returns:
            Tensor representation of the protein

        Raises:
            ValueError: If cif_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        if cif_file_path:
            raise ValueError("ESM-C does not support CIF or PDB-based embeddings")
        protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.

        Args:
            complex_input: List of (chain_id, sequence) tuples
            cif_file_path: Not supported for ESM-C

        Returns:
            Tensor representation of the protein complex

        Raises:
            ValueError: If cif_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_chain import ProteinChain
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            raise ValueError("ESM-C does not support CIF or PDB-based embeddings")

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
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file (not supported for ESM-C)
            extra_layers: List of layer indices to return embeddings for
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.
        """
        import gc

        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embeddings = []
        protein_tensor = None
        logits_output = None

        try:
            if isinstance(sequence_or_complex, list):
                protein_tensor = self._get_esm_protein_tensor_for_complex(
                    sequence_or_complex, cif_file_path
                )
            else:
                protein_tensor = self._get_esm_protein_tensor_for_sequence(
                    sequence_or_complex, cif_file_path
                )

            with torch.no_grad():
                logits_output = self.client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=False,
                        return_embeddings=True,
                        return_hidden_states=True if len(extra_layers) > 0 else False,
                    ),
                )
            if extra_layers:
                hidden_states = logits_output.hidden_states
                for extra_layer_idx in extra_layers:
                    layer_embedding = self._pool_by_domains(
                        hidden_states[extra_layer_idx], domain_boundaries
                    ).squeeze()
                    embeddings.append(layer_embedding.cpu().tolist())
                del hidden_states

            final_embedding = self._pool_by_domains(
                logits_output.embeddings.detach().cpu(), domain_boundaries
            ).squeeze(0)
            embeddings.append(final_embedding.tolist())

        finally:
            # First recursively clean complex objects
            if "logits_output" in locals() and logits_output is not None:
                recursive_cleanup(logits_output)

            if "protein_tensor" in locals() and protein_tensor is not None:
                recursive_cleanup(protein_tensor)
            # Expanded cleanup
            for local_var in [
                "logits_output",
                "protein_tensor",
                "all_embeddings",
                "final_embedding",
            ]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

        return embeddings

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file (not supported for ESM-C)

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.constants import esm3 as esm3_constants
        from esm.utils.structure.protein_complex import ProteinComplex

        if isinstance(sequence_or_complex, list):
            protein_tensor = self._get_esm_protein_tensor_for_complex(
                sequence_or_complex, cif_file_path
            )
        else:
            protein_tensor = self._get_esm_protein_tensor_for_sequence(
                sequence_or_complex, cif_file_path
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
        self, sequence: str, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.

        Args:
            sequence: Protein sequence string
            cif_file_path: Optional path to a CIF file for structure-aware modeling

        Returns:
            Tensor representation of the protein
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            protein_complex = ProteinComplex.from_cif(path=cif_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.

        Args:
            complex_input: List of (chain_id, sequence) tuples
            cif_file_path: Optional path to a CIF file for structure-aware modeling

        Returns:
            Tensor representation of the protein complex
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_chain import ProteinChain
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            protein_complex = ProteinComplex.from_cif(path=cif_file_path)
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

    def _pool_by_domains(
        self, hidden_states: "torch.Tensor", domain_boundaries: List[int]
    ) -> "torch.Tensor":
        """
        Pool hidden states by domains and concatenate the results.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
            domain_boundaries: List of domain boundary positions (0-indexed)

        Returns:
            Concatenated tensor of domain embeddings
        """
        import torch

        if not domain_boundaries:
            return hidden_states.mean(0)

        # Convert to sorted list and add 0 at the start and seq_len at the end
        boundaries = [0] + sorted(domain_boundaries) + [hidden_states.shape[0]]
        domain_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            domain_embedding = hidden_states[start_idx:end_idx, :].mean(0)
            domain_embeddings.append(domain_embedding)

        return torch.cat(domain_embeddings, dim=-1)

    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-1/2 client with the specified model.

        Args:
            model_name: Name of the ESM-1 or ESM-2 model to load
        """
        import torch

        logging.info(
            f"Loading ESM-1/2 model: {model_name} (note: we have esm in sys.modules: {'esm' in sys.modules})"
        )
        if "esm" in sys.modules:
            logging.error(
                f"WE ARE REMOVING ESM FROM sys.modules... GOD HELP US. To run ESM2, we have to uninstall the esm package (which is only for ESMC/3). This is effectively uninstalling ESMC/ESM3 from the system. If they get used later, they will mysteriously fail."
            )
            sys.modules.pop("esm")

        self.model_name = model_name

        # Load model from torch hub
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", model_name)  # type: ignore[reportGeneralTypeIssues]
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Set to evaluation mode

        # For 15B model, use accelerate to split across GPU + CPU only on smaller GPUs
        if model_name == "esm2_t48_15B_UR50D" and torch.cuda.is_available():
            # Check available GPU memory
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if total_memory_gb < 50:  # Only use CPU offloading on GPUs with <50GB
                try:
                    import accelerate
                    from accelerate import dispatch_model, infer_auto_device_map

                    # Calculate max memory (adjust for CUDA overhead ~2GB)
                    max_memory = {0: "42GiB", "cpu": "100GiB"}  # Adjust based on your system

                    device_map = infer_auto_device_map(
                        self.model,
                        max_memory=max_memory,
                        no_split_module_classes=[
                            "ESM1bLayerNorm",
                            "TransformerLayer",
                        ],  # Keep layers intact
                    )

                    self.model = dispatch_model(self.model, device_map=device_map)
                    self.device = torch.device("cuda")  # Primary device

                    logging.info(
                        f"Loaded {model_name} with device_map across GPU + CPU (GPU memory: {total_memory_gb:.1f}GB)"
                    )
                    logging.info(f"Device map: {device_map}")

                except ImportError:
                    logging.warning("accelerate not available, falling back to CPU for 15B model")
                    self.device = torch.device("cpu")
            else:
                # Large GPU (≥50GB) - load entirely on GPU for best performance
                self.model = self.model.cuda()
                self.device = torch.device("cuda")
                logging.info(
                    f"Loaded {model_name} entirely on GPU (GPU memory: {total_memory_gb:.1f}GB)"
                )
        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for ESM-1/2
            extra_layers: Not supported for ESM-1/2
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.

        Raises:
            ValueError: If a complex, CIF/PDB file, or extra_layers are provided (not supported)
        """
        import gc

        import torch

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sequence = sequence_or_complex
        if cif_file_path:
            raise ValueError("ESM1 and 2 do not support CIF or PDB-based embeddings")
        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        if len(extra_layers) > 0:
            raise ValueError("ESM1 and 2 do not support extra layers")

        batch_tokens = None
        token_embeddings = None
        protein_embedding = None

        try:
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            MODEL_TO_NUM_LAYERS = {
                "esm2_t48_15B_UR50D": 48,
                "esm2_t36_3B_UR50D": 36,
                "esm2_t33_650M_UR50D": 33,
                "esm2_t30_150M_UR50D": 30,
                "esm2_t12_35M_UR50D": 12,
                "esm2_t6_8M_UR50D": 6,
                "esm1v_t33_650M_UR90S_1": 33,
                "esm1v_t33_650M_UR90S_2": 33,
                "esm1v_t33_650M_UR90S_3": 33,
                "esm1v_t33_650M_UR90S_4": 33,
                "esm1v_t33_650M_UR90S_5": 33,
                "esm_msa1b_t12_100M_UR50S": 12,
                "esm_msa1_t12_100M_UR50S": 12,
                "esm1b_t33_650M_UR50S": 33,
                "esm1_t34_670M_UR50S": 34,
                "esm1_t34_670M_UR50D": 34,
                "esm1_t34_670M_UR100": 34,
                "esm1_t12_85M_UR50S": 12,
                "esm1_t6_43M_UR50S": 6,
            }

            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[MODEL_TO_NUM_LAYERS.get(self.model_name)]
                )
                layer_num = MODEL_TO_NUM_LAYERS.get(self.model_name)
                token_embeddings = results["representations"][layer_num]

                # Remove cls and eos tokens, then pool by domains within no_grad block
                token_embeddings_no_special = token_embeddings[0, 1:-1]  # Remove CLS and EOS tokens
                protein_embedding = (
                    self._pool_by_domains(token_embeddings_no_special, domain_boundaries)
                    .detach()
                    .cpu()
                )
                embeddings = [protein_embedding.tolist()]

        finally:
            # Expanded cleanup
            for local_var in ["batch_tokens", "token_embeddings", "protein_embedding", "results"]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

            # Force garbage collection
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure GPU operations are complete
                torch.cuda.ipc_collect()  # Critical for multi-process environments

        return embeddings

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for ESM-1/2

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'

        Raises:
            ValueError: If a complex or CIF/PDB file is provided (not supported)
        """
        import torch

        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        sequence = sequence_or_complex

        if cif_file_path:
            raise ValueError("ESM1 and 2 do not support CIF or PDB-based logits")

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
                if vocab_char in self.alphabet.standard_toks:  # Only include standard amino acids
                    prob = probs[vocab_idx]
                    seq_id = f"{wt_aa}{pos}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)
