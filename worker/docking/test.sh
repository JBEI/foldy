#!/bin/bash

#python /code/dock.py /code/protein.pdb "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1N2)C=C)C)C)CCC(=O)[O-])C(=C3C)CCC(=O)O)C=C." /my_output
# docker run -it -v "$(pwd)":/code:ro -v "$(pwd)/tmp":/my_output vina \
#   python /code/dock.py /code/tmp/bezE.pdb "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1N2)C=C)C)C)CCC(=O)[O-])C(=C3C)CCC(=O)O)C=C" /my_output

# docker run vina python /code/dock.py /code/tmp/bezE.pdb "CCCCC" /my_output

docker build -t vina .
docker run -v "$(pwd)/testdata:/data:ro" \
  vina \
  python \
  /code/dock.py \
  /data/smallprot.pdb \
  "CCCCC" \
  /tmp \
  --bounding_box_residue=M1 --bounding_box_radius_angstrom=2 \
