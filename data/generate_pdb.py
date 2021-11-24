import Bio
import json
from Bio.PDB import *
from Bio import SeqIO


def get_coord(residue, atom):
    coord_atom = residue[atom].get_vector()
    coord_atom = [coord_atom.__getitem__(0), coord_atom.__getitem__(1), coord_atom.__getitem__(2)]
    return coord_atom


def get_pdb_coord(pdb_path, target_atoms=["N", "CA", "C", "O"]):

    p = PDBParser()
    pdb_dict = {}
    for record in SeqIO.parse(pdb_path, "pdb-atom"):
        #print(record.seq)
        pdb_dict['seq'] = str(record.seq)

    pdb_dict['coords']  = {code:[] for code in target_atoms}

    structure = p.get_structure('name', pdb_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                for code in target_atoms:
                    pdb_dict['coords'][code].append(get_coord(residue, code))
    pdb_dict['num_chains'] = 1
    pdb_dict['name'] = "CB6_VH"
    #print(pdb_dict)
    return pdb_dict
dataset=[]
pdb_dict=get_pdb_coord("/Users/janie/Desktop/pre-training/alphafold2_results/results/antibody/CB6/CB6_VH.pdb")
dataset.append(pdb_dict)

outfile = 'chain_set.jsonl'
with open(outfile, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')