from typing import List
import numpy as np
import logging
import unittest

import Bio
import Bio.PDB
import Bio.SVDSuperimposer

import utils


class TestRMSD(unittest.TestCase):
    
    def test_rmsd_computation(self):
        parser = Bio.PDB.PDBParser(PERMISSIVE=0)  # strict parser
        structure1 = parser.get_structure("test1", "data/SAbDab-Fv-unbound/chothia/1ad0.pdb")
        structure2 = parser.get_structure("test2", "data/SAbDab-Fv-unbound/chothia/1fgv.pdb")

        h_res_1, l_res_1 = utils.extract_variable_regions_residues(structure1, "B", "A")
        h_res_2, l_res_2 = utils.extract_variable_regions_residues(structure2, "H", "L")

        coord_h1 = utils.extract_calpha_coord(h_res_1)
        coord_l1 = utils.extract_calpha_coord(l_res_1)
        coord_h2 = utils.extract_calpha_coord(h_res_2)
        coord_l2 = utils.extract_calpha_coord(l_res_2)

        sup = Bio.SVDSuperimposer.SVDSuperimposer()
        rmsd_h12 = utils.compute_rmsd(coord_h1, coord_h2, sup)
        rmsd_l12 = utils.compute_rmsd(coord_l1, coord_l2, sup)
        rmsd_v12 = utils.compute_rmsd(
            np.vstack([coord_h2, coord_l2]),
            np.vstack([coord_h1, coord_l1]),
        )

        self.assertIsNotNone(rmsd_h12)
        self.assertIsNotNone(rmsd_l12)

    def test_rmsd_dataset_computation(self):
        df = utils.get_dataset_summary()
        self.assertIsNotNone(utils.compute_rmsd_dataset(df, num_samples=3))


if __name__ == "__main__":
    unittest.main()