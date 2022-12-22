"""Utilities for doing docking
"""
from re import fullmatch
import subprocess
import os

from vina import Vina
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nglview as nv
from Bio.PDB import PDBParser 
from Bio.PDB import Polypeptide
from pathlib import Path


def prepare_receptor_pdbqt(
    receptor_pdb_filename,
    out_path=None,
    prepare_receptor_binary_name="prepare_receptor"
):
    """Convert pdb receptor into pdbqt
    """
    
    if out_path != None:
        receptor_pdbqt_filename = out_path
        print("out_path")
    else:
        receptor_pdbqt_filename = receptor_pdb_filename + "qt"
        print("No out_path")
    
    subprocess.run([prepare_receptor_binary_name, "-r", receptor_pdb_filename, "-o", receptor_pdbqt_filename, "-A", "bonds_hydrogens"], 
                   check = True, )


def dock(
    ligand_fpath: Path,
    receptor_pdb_fpath: Path,
    receptor_pdbqt_fpath: Path,
    poses_fpath: Path,
    bb_residue: str = None,
    bb_radius_ang: float = None,
):
    """Dock a ligand into a protein, returning gibbs free energy."""
    
    v = Vina(sf_name='vina')
    v.set_receptor(str(receptor_pdbqt_fpath))
    v.set_ligand_from_file(str(ligand_fpath))
    
    if not bb_residue and not bb_radius_ang:
        center, box_size = get_boundingbox(str(receptor_pdb_fpath), 5)
    elif bb_residue and bb_radius_ang:
        center, box_size = get_boundingbox_around_residue(
            str(receptor_pdb_fpath),
            bb_residue,
            bb_radius_ang
        )
    else:
        raise KeyError(f'If providing a bounding box residue, must provide residue name (like W23) and radius in angstroms.')

    print(f'Using bounding box {(center, box_size)}')
    # for e1 in (center, box_size):
    #     for e2 in e1:
    #         print(type(e2))
    
    v.compute_vina_maps(center=center, box_size=box_size)
    
    # Dock the ligand
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses(str(poses_fpath), n_poses=20, overwrite=True)
    return v.energies(1)[0][0]


def get_boundingbox(pdb_fname, buffer = 0):
    """Find center and box_size of pdb.
    
    Args:
        pdb_fname: file name of pdb being parsed
        buffer: number of angstroms larger than minimum box
    
    Returns: (center[x,y,z] , box[x,y,z])
    """
    
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_fname)
    mins = [0,0,0]
    maxs = [0,0,0]
    for atom in structure.get_atoms():
        coord = atom.get_coord()
        for i in range(3):
            maxs[i] = max(coord[i], maxs[i])
            mins[i] = min(coord[i], mins[i])
        
    center, box = [], []
    for i in range(3):
        center.append(float((mins[i] + maxs[i]) / 2))
    for i in range(3):
        box.append(float((maxs[i] - mins[i]) + buffer))
    
    return (center, box)


def get_boundingbox_around_residue(
    receptor_pdb_fpath,
    residue_id,
    radius_ang):
    parser = PDBParser()
    structure = parser.get_structure("protein", receptor_pdb_fpath)
    
    res_id_match = fullmatch(r'([A-Z]+)([0-9]+)', residue_id)
    if not res_id_match:
        raise KeyError(f'Residue ID must be of the form W26, got {residue_id}')
    res_type = res_id_match.group(1)
    res_number = int(res_id_match.group(2)) - 1
    
    res = list(structure.get_residues())[res_number]
    
    if Polypeptide.three_to_one(res.resname) is not res_type:
        raise KeyError(f'Residue {res_number} in the PDB is type {res.resname}, expected {res_type}')

    atms = list(res.get_atoms())
    center_np32 = sum([a.get_coord() for a in atms]) / len(atms)
    center = [float(e) for e in center_np32]
    box = [2 * radius_ang, 2 * radius_ang, 2 * radius_ang]

    return (center, box)


def get_energies(ligand_list, 
                 protein_type, 
                 protein_fpath,
                 pdbqt_fpath):
    """Return list of energies corresponding to ligand_list
    """
    
    energieslist = []

    poses_dir = "Ligands_poses/" + protein_type + "/"
    
    for lig_name in tqdm(ligand_list):
        energieslist.append(
            dock( 
                Path('Ligands') / lig_name, 
                Path(protein_fpath),
                Path(pdbqt_fpath),
                Path(poses_dir) / lig_name))
    return energieslist


def get_lig_list():
    """Return list of ligands corresponding to jbei_work/Ligands/
    """
    ligand_fnames = os.listdir('Ligands')
    ligand_fnames = [i for i in ligand_fnames if i.endswith('.pdbqt')]
    return ligand_fnames


def make_barplot(ligand_fnames, energieslist):
    """Make Plot with ligands and energies
    """
    
    data = pd.DataFrame(
            {
                "fname": ligand_fnames, 
                "energy": energieslist
            }
        )

    data["ligand_name"] = data.fname.str.slice(stop = -6)
    ax = sns.barplot(
                data=data,
                x="ligand_name",
                y="energy"
            ).set(title='Regular Protein')
    plt.xticks(rotation = 90);
    plt.ylim((-7,0));
    plt.show()

    
def get_wetlab_kinetics():
    """Returns a pandas DataFrame with wetlab kinetics measurements."""
    cans = 'limonene apinene aterpinene pcymene methylcyclohexane'.split(' ')
    cannots = '''ycaprolactone
decalactone
tetradecalactone
hexalactone
heptalactone
betapinene
apineneoxide
eucalyptol
borneol
carveol
carvone
carvacrol
2methyl2cyclohexen1one
2methylcyclohexanone
Bisabolene'''.split('\n')
    return pd.DataFrame({
        'substrate': cans + cannots,
        'measurement': [True] * len(cans) + [False] * len(cannots)
    })
    

def make_barplot_wetlab(ligand_fnames, energieslist):
    """Make Plot with ligands and energies
    """
    
    data = pd.DataFrame(
            {
                "fname": [WLresults[i][0] for i in range(0, WLresults.length)], 
                "energy": [WLresults[i][1] for i in range(0, WLresults.length)],
                
            }
        )

    data["ligand_name"] = data.fname.str.slice(stop = -6)
    ax = sns.barplot(
                data=data,
                x="ligand_name",
                y="energy"
            ).set(title='Regular Protein')
    plt.xticks(rotation = 90);
    plt.ylim((-7,0));
    plt.show()


def get_literature_kinetics():
    """Returns a pandas DataFrame with literature kinetics values."""
    return pd.DataFrame({
        'substrate': 'hexane heptane octane nonane decane undecane 2methyloctane 25dimethylhexane limonene pcymene ethylcyclohexane'.split(' '),
        'activity': [0.7, 21.2, 60.8, 49.7, 13.8, 0.4, 35, np.NAN, 31.2, 38.9, np.NAN],
        'kd': [3.7, 0.48, 0.17, 0.022, 0.010, 0.020, 0.020, 1.5, 4.7, 5.8, 4.2]
    })


def calc_energy(receptor_fpath, ligandpose_fpath):
    """Returns energy of specific ligand pose."""
    v = Vina(sf_name='vina', verbosity = 2)
    v.set_receptor(str(receptor_fpath))
    v.set_ligand_from_file(str(ligandpose_fpath))
    
    receptor_pdb_path = "Protein_models/"+receptor_fpath[21:]
    receptor_pdb_path = receptor_pdb_path[:-2]
    
    center, box_size = get_boundingbox(str(receptor_pdb_path), 5)
    v.compute_vina_maps(center=center, box_size=box_size)
    
    energy = v.score()
    return energy[0]
    

def energies_list_for_protein(receptor_fpath):
    """Returns energy of all ligand poses for specific protein conformation."""
    energies = []
    
    # Pulls the structure name from the path string, eg:
    #   extracts "rank_01" from Protin_models/pdbqt/rank_01_model_1_seed_22_unrelaxed.pdbqt
    r_fname = receptor_fpath[21:28]

    ligands_fnames = os.listdir('Ligands_poses/' + r_fname)
    ligands_fnames = [l for l in ligands_fnames if l.endswith('.pdbqt')]
    ligands_fnames = sorted(ligands_fnames)
    for l in ligands_fnames:
        energies.append(calc_energy(receptor_fpath, 'Ligands_poses/'+r_fname+'/'+l))
    print(receptor_fpath)
    return energies, ligands_fnames
