from typing import List
import logging
from dataclasses import dataclass

import numpy as np

import Bio
import Bio.PDB
import Bio.SVDSuperimposer


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