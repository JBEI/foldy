import pytest

from app.helpers.boltz_yaml_helper import BoltzYamlHelper

# Sample YAML strings for testing
BASIC_PROTEIN_YAML = """
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPE
"""

MULTI_CHAIN_YAML = """
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPE
"""

COMPLEX_YAML = """
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPE
      modifications:
        - position: 1
          ccd: MSE
  - ligand:
      id: X
      smiles: CC(=O)O
constraints:
  - bond:
      atom1: [A, 1, CA]
      atom2: [B, 1, CB]
  - pocket:
      binder: A
      contacts: [[B, 1], [B, 2]]
"""


def test_get_protein_sequences_single_chain():
    helper = BoltzYamlHelper(BASIC_PROTEIN_YAML)
    sequences = helper.get_protein_sequences()
    assert sequences == [("A", "MVTPE")]


def test_get_protein_sequences_multi_chain():
    helper = BoltzYamlHelper(MULTI_CHAIN_YAML)
    sequences = helper.get_protein_sequences()
    assert sequences == [("A", "MVTPE"), ("B", "MVTPE")]


def test_get_modifications():
    helper = BoltzYamlHelper(COMPLEX_YAML)
    mods = helper.get_modifications()
    expected = [{"chain_ids": ["A", "B"], "position": 1, "ccd": "MSE"}]
    assert mods == expected


def test_get_ligands():
    helper = BoltzYamlHelper(COMPLEX_YAML)
    ligands = helper.get_ligands()
    expected = [{"chain_ids": ["X"], "smiles": "CC(=O)O", "ccd": None}]
    assert ligands == expected


def test_get_constraints():
    helper = BoltzYamlHelper(COMPLEX_YAML)
    constraints = helper.get_constraints()
    expected = [
        {"bond": {"atom1": ["A", 1, "CA"], "atom2": ["B", 1, "CB"]}},
        {"pocket": {"binder": "A", "contacts": [["B", 1], ["B", 2]]}},
    ]
    assert constraints == expected


# def test_get_version():
#     helper = BoltzYamlHelper(BASIC_PROTEIN_YAML)
#     assert helper.get_version() == 1  # Method doesn't exist


def test_invalid_yaml():
    with pytest.raises(Exception):  # yaml.YAMLError would be more specific
        BoltzYamlHelper("invalid: :")


def test_validates_invalid_sequences_field():
    with pytest.raises(Exception):
        BoltzYamlHelper(
            """version: 1
sequences:
  protein:
    id: [A, B]
    sequence: MVTPE
"""
        )
