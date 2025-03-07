import yaml
from typing import List, Tuple, Optional, Any, Dict, Union


class BoltzYamlHelper:
    """
    A helper class to parse and retrieve data from a Boltz YAML input file.

    Usage:
        helper = BoltzYamlHelper("...")
        protein_seqs = helper.get_protein_sequences()
        # protein_seqs -> List of (chain_id, sequence) pairs.
    """

    def __init__(self, yaml_str: str):
        """
        Initialize the helper by loading the YAML data from the provided file. Raises ValueError if misformatted.

        Args:
            yaml_path: Path to the YAML file.
        """
        print(f"LOADING YAML STR {yaml_str}")
        self.yaml_str = yaml_str
        self.data = yaml.safe_load(yaml_str)

        if type(self.data) != dict:
            raise ValueError("Boltz YAML config is not a dictionary!")
        if "version" not in self.data:
            raise ValueError("Boltz YAML config does not have a 'version' field!")
        if "sequences" not in self.data:
            raise ValueError("Boltz YAML config does not have a 'sequences' field!")

        if not isinstance(self.data["sequences"], list):
            raise ValueError(
                f"Boltz YAML config issue: The 'sequences' field must be a list, got {type(self.data['sequences'])}"
            )

        # Validate each sequence entry is a dict with exactly one key
        for seq in self.data["sequences"]:
            if not isinstance(seq, dict):
                raise ValueError(
                    f"Boltz YAML config issue: Each sequence must be a dictionary, got {type(seq)}"
                )
            if len(seq.keys()) != 1:
                raise ValueError(
                    f"Boltz YAML config issue: Each sequence must have exactly one entity type key, got {seq.keys()}"
                )
            entity_type = list(seq.keys())[0]
            if entity_type not in ["protein", "dna", "rna", "ligand"]:
                raise ValueError(
                    f"Boltz YAML config issue: Invalid entity type in sequence: {entity_type}"
                )

        # Optionally, you can store top-level fields for easy reference:
        self.version = self.data.get("version", None)
        self.sequences = self.data.get("sequences", [])
        self.constraints = self.data.get("constraints", [])
        print(f"YAMLSTR: {yaml_str}")
        print(f"DATA: {self.data}")
        print(f"SEQUENCES: {self.sequences}", flush=True)

    def get_protein_sequences(self) -> List[Tuple[str, str]]:
        """
        Retrieve a list of tuples (chain_id, amino_acid_sequence) for all proteins
        defined in the YAML.

        Returns:
            A list of (chain_id, sequence) pairs. If multiple chain IDs share the
            same sequence, they will each appear as a separate tuple.
        """
        results = []
        for entry in self.sequences:
            # Each entry is a dict with one key (entity_type)
            if "protein" not in entry:
                continue

            protein_data = entry["protein"]
            # The 'id' field can be either a single chain ID or a list of chain IDs.
            chain_ids = protein_data.get("id")
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            elif isinstance(chain_ids, list):
                pass  # already a list
            else:
                # if not provided or invalid, skip
                continue

            # Retrieve the sequence field
            seq = protein_data.get("sequence", None)
            if seq is None:
                # No protein sequence found, skip
                continue

            # Add (chain_id, sequence) for each chain ID
            for cid in chain_ids:
                results.append((cid, seq))

        return results

    def get_modifications(self) -> List[Dict[str, Any]]:
        """
        Retrieve the modifications for each polymer that has a 'modifications' section.

        Returns:
            A list of modification dictionaries, where each modification dictionary
            is augmented with the associated chain_ids for context.
        """
        modifications_list = []
        for entry in self.sequences:
            entity_data = list(entry.values())[
                0
            ]  # Get the entity data regardless of type
            # entity_data might contain 'modifications'
            mods = entity_data.get("modifications", [])
            if mods:
                chain_ids = entity_data.get("id")
                if isinstance(chain_ids, str):
                    chain_ids = [chain_ids]
                for mod in mods:
                    # Copy to avoid mutating the original
                    mod_copy = dict(mod)
                    mod_copy["chain_ids"] = chain_ids
                    modifications_list.append(mod_copy)
        return modifications_list

    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        Retrieve all constraints from the YAML (both 'bond' and 'pocket' if present).

        Returns:
            A list of constraint dictionaries.
        """
        return self.constraints

    def get_ligands(self) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Retrieve a list of ligands (either by 'smiles' or 'ccd').

        Returns:
            A list of dictionaries with fields like:
            {
                "chain_ids": [...],
                "smiles": "...",
                "ccd": "...",
            }
        """
        ligands = []
        for entry in self.sequences:
            if "ligand" not in entry:
                continue

            ligand_data = entry["ligand"]
            chain_ids = ligand_data.get("id")
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            # Build a small dict capturing relevant ligand info:
            ligand_info = {
                "chain_ids": chain_ids,
                "smiles": ligand_data.get("smiles"),
                "ccd": ligand_data.get("ccd"),
            }
            ligands.append(ligand_info)
        return ligands

    def get_version(self) -> Optional[int]:
        """
        Return the 'version' field in the YAML, if it exists.

        Returns:
            The integer version or None if not found.
        """
        return self.version


# Example usage (uncomment to run it locally):
# if __name__ == "__main__":
#     helper = BoltzYamlHelper("path/to/input.yaml")
#     protein_sequences = helper.get_protein_sequences()
#     print(protein_sequences)
#     # e.g. [('A', 'MVTPE...'), ('B', 'MVTPE...')]
