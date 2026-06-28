import os
import math
import numpy as np
from pathlib import Path

REDOX_POT_DIR = Path("your_own_path/Redox-Pot")
MODEL_DIR = Path("your_own_path/Batt-P30K/Models")

# predict the ea/ip and output to a file
def predict_energy_by_pinet2(pinet2_model_path, xyz_path):
    """
        Predict HOMO/LUMO, EA/IP by PiNet2 models added by zhan-yun zhang.

        Args:
        ----
            pinet2_model_path (str)         : path of the pinet2 models.
            xyz_path (str)                  : path of the xyz files.
    """

    import os
    import numpy as np
    from pinn import get_calc
    from ase.io import read

    # read all atoms
    file_to_atoms = {}
    for file in sorted(os.listdir(xyz_path)):
        if not file.endswith(".xyz"):
            continue
        file_to_atoms[file] = read(xyz_path + file, format='xyz')

    mol_to_pred = {}
    for model_dir in sorted(os.listdir(pinet2_model_path)):
        print("-------", model_dir)
        calc = get_calc(pinet2_model_path+model_dir, properties=['energy'])
        for file, atoms in file_to_atoms.items():
            e_pred = calc.get_potential_energy(atoms)
            if file not in mol_to_pred.keys():
                mol_to_pred[file] = [e_pred]
            else:
                mol_to_pred[file].append(e_pred)
        del calc

    mol_to_mean = {mol: sum(values) / len(values) for mol, values in mol_to_pred.items()}

    return mol_to_mean

# predict the EA,IP,redox potential by Pinet2
def predict_redox_potential():

    XYZ_DIR = str(REDOX_POT_DIR) + "/XYZ/"

    # predict EA
    mol_to_ea = predict_energy_by_pinet2(str(MODEL_DIR) + "/EA/", str(XYZ_DIR))
    mol_to_ip = predict_energy_by_pinet2(str(MODEL_DIR) + "/IP/", str(XYZ_DIR))

    #
    file_name = []
    for file in sorted(os.listdir(XYZ_DIR)):
        if not file.endswith(".xyz"):
            continue
        file_name.append(file)
    
    predicted_ip = [mol_to_ip[key] for key in file_name]
    predicted_ea = [mol_to_ea[key] for key in file_name]

    mol_ox_free_energy = [0.95 * dIP - 0.21 for dIP in predicted_ip] # eV
    mol_red_free_energy = [-0.83 * dEA - 1.11 for dEA in predicted_ea]  # eV

    mol_ox_potential = [value - 1.44 for value in mol_ox_free_energy] # v.s. Li+/Li
    mol_red_potential = [-1.0 * value - 1.44 for value in mol_red_free_energy] # v.s. Li+/Li

    # print 
    print("File;OxPot;RedPot")
    for mol_idx in range(0, len(file_name)):
        print("%s;%f;%f"%(file_name[mol_idx], mol_ox_potential[mol_idx], mol_red_potential[mol_idx]))

#
if __name__ == "__main__":

    # run by slurm
    predict_redox_potential()
    