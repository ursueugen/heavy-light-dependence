from typing import List
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import Bio
import Bio.PDB
import Bio.SVDSuperimposer

def get_dataset_summary():
    df = pd.read_csv("data/SAbDab-Fv-unbound/20210927_0887165_summary.tsv", sep='\t')
    df.drop_duplicates(subset = ["pdb"], keep = 'first', inplace = True)
    df.set_index("pdb", inplace = True)
    df = df[["Hchain", "Lchain", "engineered"]].copy()
    return df

def extract_variable_regions_residues(
    structure: Bio.PDB.Structure.Structure, 
    hchain_id: str, 
    lchain_id: str):
    '''
    Extracts residues of the variable regions. It's used to extract
     a constant length for heavy and light chains, as to allow direct
     structural alignments (e.g. based on Calpha carbons).
    '''

    H_CHAIN_LEN = 113 + 12
    L_CHAIN_LEN = 107 + 11

    if len(structure) > 1:
        logging.warning(f"Structure {structure.id} has >1 model")
    model = structure[0]
    hchain = model[hchain_id]
    lchain = model[lchain_id]

    hchain_res = list(hchain.get_residues())[:H_CHAIN_LEN]
    lchain_res = list(lchain.get_residues())[:L_CHAIN_LEN]

    return hchain_res, lchain_res

def is_atom_ca(atom: Bio.PDB.Atom.Atom) -> bool:
    return atom.get_id() == "CA"

def get_atom_coord(atom: Bio.PDB.Atom.Atom) -> np.array:
    """Returns 1D 3-component array of coordinates from an Atom."""
    return np.array(atom.get_coord())

def extract_calpha_coord(residues: List[Bio.PDB.Residue.Residue]) -> np.array:
    coord_array = None
    for r in residues:
        ca_found = False
        atoms = r.get_atoms()        
        for atom in atoms:
            if is_atom_ca(atom):
                if ca_found:
                    raise RuntimeError("Found >1 Calpha carbons for a residue.")
                coord = get_atom_coord(atom)
                ca_found = True
        
        if coord_array is not None:
            coord_array = np.vstack([coord_array, coord])
        else:
            coord_array = coord
    return coord_array

def compute_rmsd(
    coord_1: np.array, 
    coord_2: np.array, 
    superimposer = None) -> float:
    """
    Minimizes and returns RMSD between 2 sets of coordinates.
    """
    
    if superimposer is None:
        superimposer = Bio.SVDSuperimposer.SVDSuperimposer()
    superimposer.set(coord_1, coord_2)
    superimposer.run()
    return superimposer.get_rms()


@dataclass(frozen=True)
class Pair:
    """
    A pair represents a 2-combination, in which order
     doesn't matter.
    """
    elem1: str
    elem2: str

    def __eq__(self, other):
        if ((self.elem1, self.elem2) == (other.elem1, other.elem2)
            or (self.elem1, self.elem2) == (other.elem2, other.elem1)):
            return True
        else:
            return False
    
    def __hash__(self):
        return hash( self.as_sorted_tuple() )
    
    def as_sorted_tuple(self) -> tuple:
        return tuple(sorted([self.elem1, self.elem2]))

def sample_pairs(ids: set, num_pairs: int = 100) -> set:
    """Sample unique pairs (2-combinations) from the set of ids."""
    
    def sample_pair(choice: set) -> Pair:
        return Pair(
        *np.random.choice(list(choice), size = 2, replace = False)
    )
    
    samples = set()

    while len(samples) < num_pairs:
        pair = sample_pair(ids)
        if pair not in samples:
            samples.add(pair)
    
    samples = set(map(lambda p: p.as_sorted_tuple(), samples))
    
    return samples

def compute_rmsd_dataset(
    df, 
    num_samples=1000, 
    pdb_dir_path=Path('data/SAbDab-Fv-unbound/chothia/')
    ) -> pd.DataFrame:
    '''
    Creates a RMSD dataset by sampling 2-combinations and
     computing RMSD between heavy and light chains.
    '''
    ids = set(df.index)
    samples = sample_pairs(ids, num_pairs = num_samples)
    # print(samples)

    records = []
    for pair in samples:
        try:
            name1 = pair[0]
            name2 = pair[1]
            path1 = pdb_dir_path / f"{name1}.pdb"
            path2 = pdb_dir_path / f"{name2}.pdb"
            hchain_name_1 = df.loc[name1]['Hchain']
            lchain_name_1 = df.loc[name1]['Lchain']
            hchain_name_2 = df.loc[name2]['Hchain']
            lchain_name_2 = df.loc[name2]['Lchain']

            parser = Bio.PDB.PDBParser(PERMISSIVE=0)  # strict parser
            structure1 = parser.get_structure(name1, path1)
            structure2 = parser.get_structure(name2, path2)

            h_res_1, l_res_1 = extract_variable_regions_residues(structure1, hchain_name_1, lchain_name_1)
            h_res_2, l_res_2 = extract_variable_regions_residues(structure2, hchain_name_2, lchain_name_2)

            coord_h1 = extract_calpha_coord(h_res_1)
            coord_l1 = extract_calpha_coord(l_res_1)
            coord_h2 = extract_calpha_coord(h_res_2)
            coord_l2 = extract_calpha_coord(l_res_2)

            sup = Bio.SVDSuperimposer.SVDSuperimposer()
            rmsd_h12 = compute_rmsd(coord_h1, coord_h2, sup)
            rmsd_l12 = compute_rmsd(coord_l1, coord_l2, sup)
            rmsd_v12 = compute_rmsd(
                np.vstack([coord_h2, coord_l2]),
                np.vstack([coord_h1, coord_l1]),
            )

            record = {
                "pdb1": name1,
                "pdb2": name2,
                "rmsd_h12": rmsd_h12,
                "rmsd_l12": rmsd_l12,
                "engineered_1": df.loc[name1]['engineered'],
                "engineered_2": df.loc[name2]['engineered'],
            }
            records.append(record)
        except:
            pass

    res = pd.DataFrame.from_records(records)
    return res

def annotate_engineered_status(df):
    pair_eng_classes = []
    for _, s in df.iterrows():
        eng_1 = bool(s['engineered_1'])
        eng_2 = bool(s['engineered_2'])
        
        if eng_1 and eng_2:
            label = "both"
        elif eng_1 or eng_2:
            label = "one"
        elif not (eng_1 or eng_2):
            label = "none"

        pair_eng_classes.append(label)

    df['engineered_label'] = pair_eng_classes