"""A simple script to do simple docking for simple people."""

import argparse
from pathlib import Path

from openbabel import pybel

from util import dock, prepare_receptor_pdbqt


def cli():
    parser = argparse.ArgumentParser(
        description=("A CLI for docking."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adfrsuite_path", type=str, default='/adfrsuite')
    parser.add_argument(
        "--bounding_box_residue",
        help='Residue around which to bound the search, in a fomat like W45.',
        type=str,
        default=None
    )
    parser.add_argument(
        "--bounding_box_radius_angstrom",
        help='Radius to bound the search around the residue in angstroms.',
        type=float,
        default=None
    )
    parser.add_argument("receptor_pdb_fpath", type=str)
    parser.add_argument("ligand_smiles_str", type=str)
    parser.add_argument("out_dir_path", type=str)
    return parser


def main(args):
    out_dir = Path(args['out_dir_path'])

    ligand_fpath = out_dir / 'ligand.pdbqt'
    receptor_pdb_fpath = Path(args['receptor_pdb_fpath'])
    receptor_pdbqt_fpath = out_dir / 'receptor.pdbqt'
    poses_pdbqt_fpath = out_dir / 'poses.pdbqt'
    poses_pdb_fpath = out_dir / 'poses.pdb'
    poses_sdf_fpath = out_dir / 'poses.sdf'

    ligand_mol = pybel.readstring('smi', args['ligand_smiles_str'])
    ligand_mol.make3D()
    ligand_mol.write('pdbqt', str(ligand_fpath), overwrite=True)

    prepare_receptor_pdbqt(
        receptor_pdb_fpath,
        out_path=receptor_pdbqt_fpath,
        prepare_receptor_binary_name=f'{args["adfrsuite_path"]}/bin/prepare_receptor'
    )

    print(f"BBox: {args['bounding_box_residue']}, {args['bounding_box_radius_angstrom']}A")

    energy = dock(
        ligand_fpath,
        receptor_pdb_fpath,
        receptor_pdbqt_fpath,
        poses_pdbqt_fpath,
        args['bounding_box_residue'],
        args['bounding_box_radius_angstrom'],
    )
    print(f'Gibbs free energy of best pose: {energy}kJ')


    mols = pybel.readfile(filename=str(poses_pdbqt_fpath), format='pdbqt')
    poses_pdb = pybel.Outputfile(format="pdb", filename=str(poses_pdb_fpath), overwrite=True)
    poses_sdf = pybel.Outputfile(format="sdf", filename=str(poses_sdf_fpath), overwrite=True)
    for ii, mol in enumerate(mols):
        # pose_pdb_fname = out_dir / f'ranked_{ii}.pdb'
        # mol.write(format='pdb', filename=str(pose_pdb_fname))
        poses_pdb.write(mol)
        poses_sdf.write(mol)
    poses_pdb.close()
    poses_sdf.close()

    with open(out_dir / 'energy.txt', 'w') as f:
        f.write(str(energy))


if __name__ == "__main__":
    args: dict = vars(cli().parse_args())
    main(args)
