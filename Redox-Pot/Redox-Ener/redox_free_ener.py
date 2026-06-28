import os
import math
import numpy as np
from pathlib import Path

REDOX_ENER_DIR = Path("your_own_path/Redox-Pot/Redox-Ener/")

# directory of the intermediate files
INPUT_DIR = REDOX_ENER_DIR / "Input"
if not os.path.exists(INPUT_DIR):
    os.mkdir(INPUT_DIR)

# class of molecules in the MPcules data set
# https://next-gen.materialsproject.org/molecules
class MPculesMol():
    def __init__(self, mol_id):
        self.id = mol_id  # molecule_id
        self.inchi = ""  # inchi
        self.smiles = ""  # SMILES
        self.solvent = []  # unique_solvents
        self.level = []  # unique_levels_of_theory
        self.charge = ""  # charge
        self.spin_multiplicity = ""  # spin_multi
        self.num_atoms = ""  # natoms
        self.num_elements = ""  # nelements
        self.composition = ""  # formula_alphabetical
        self.elements = []  # elements
        self.solvent_to_ie = {}  # ionization_energy, keys from solvent
        self.solvent_to_ea = {}  # electron_affinity, keys from solvent
        self.solvent_to_redox_level = {}  # redox_levels_of_theory
        self.solvent_to_oxidation_pot = {}  # oxidation_potential, keys from solvent
        self.solvent_to_reduction_pot = {}  # reduction_potential, keys from solvent

    @staticmethod
    def join_to_title(delimiter):
        assert delimiter != "&" and delimiter != ":"
        joined_line = "MolId" + delimiter
        joined_line = joined_line + "Inchi" + delimiter
        joined_line = joined_line + "UniqueSolvents" + delimiter
        joined_line = joined_line + "UniqueLevel" + delimiter
        joined_line = joined_line + "Charge" + delimiter
        joined_line = joined_line + "SpinMulti" + delimiter
        joined_line = joined_line + "NumAtoms" + delimiter
        joined_line = joined_line + "NumElems" + delimiter
        joined_line = joined_line + "Composition" + delimiter
        joined_line = joined_line + "listElements" + delimiter
        joined_line = joined_line + "dictIEs" + delimiter
        joined_line = joined_line + "dictEAs" + delimiter
        joined_line = joined_line + "dictRedoxLevels" + delimiter
        joined_line = joined_line + "dictOxPots" + delimiter
        joined_line = joined_line + "dictRedPots"
        return joined_line

    def join_to_line(self, delimiter):

        assert delimiter != "&" and delimiter != ":"

        joined_line = self.id + delimiter
        joined_line = joined_line + self.inchi + delimiter

        all_solvent = ""
        for index in range(0, len(self.solvent)):
            all_solvent += self.solvent[index] + "&"
        all_solvent = all_solvent.rstrip("&")
        joined_line = joined_line + all_solvent + delimiter

        all_level = ""
        for index in range(0, len(self.level)):
            all_level += self.level[index] + "&"
        all_level = all_level.rstrip("&")
        joined_line = joined_line + all_level + delimiter

        joined_line = joined_line + self.charge + delimiter
        joined_line = joined_line + self.spin_multiplicity + delimiter
        joined_line = joined_line + self.num_atoms + delimiter
        joined_line = joined_line + self.num_elements + delimiter
        joined_line = joined_line + self.composition + delimiter

        all_elements = ""
        for index in range(0, len(self.elements)):
            all_elements += self.elements[index] + "&"
        all_elements = all_elements.rstrip("&")
        joined_line = joined_line + all_elements + delimiter

        all_solvent_to_ie = ""
        for key in self.solvent_to_ie.keys():
            all_solvent_to_ie += key + ":" + str(self.solvent_to_ie[key]) + "&"
        all_solvent_to_ie = all_solvent_to_ie.rstrip("&")
        joined_line = joined_line + all_solvent_to_ie + delimiter

        all_solvent_to_ea = ""
        for key in self.solvent_to_ea.keys():
            all_solvent_to_ea += key + ":" + str(self.solvent_to_ea[key]) + "&"
        all_solvent_to_ea = all_solvent_to_ea.rstrip("&")
        joined_line = joined_line + all_solvent_to_ea + delimiter

        all_solvent_to_redox_level = ""
        for key in self.solvent_to_redox_level.keys():
            all_solvent_to_redox_level += key + ":" + str(self.solvent_to_redox_level[key]) + "&"
        all_solvent_to_redox_level = all_solvent_to_redox_level.rstrip("&")
        joined_line = joined_line + all_solvent_to_redox_level + delimiter

        all_solvent_to_oxpot = ""
        for key in self.solvent_to_oxidation_pot.keys():
            all_solvent_to_oxpot += key + ":" + str(self.solvent_to_oxidation_pot[key]) + "&"
        all_solvent_to_oxpot = all_solvent_to_oxpot.rstrip("&")
        joined_line = joined_line + all_solvent_to_oxpot + delimiter

        all_solvent_to_redpot = ""
        for key in self.solvent_to_reduction_pot.keys():
            all_solvent_to_redpot += key + ":" + str(self.solvent_to_reduction_pot[key]) + "&"
        all_solvent_to_redpot = all_solvent_to_redpot.rstrip("&")
        joined_line = joined_line + all_solvent_to_redpot + delimiter

        return joined_line

# obtain properties from dataset
def get_mpcules_mol_prop():

    mpcules_mols = []
    with open(REDOX_ENER_DIR / "RX-392.csv") as f:
        lines = f.readlines()
        titles = lines[0].strip().split("$")
        id_col = titles.index("MolId")
        inchi_col = titles.index("Inchi")
        solvent_col = titles.index("UniqueSolvents")
        level_col = titles.index("UniqueLevel")
        charge_col = titles.index("Charge")
        spin_multi_col = titles.index("SpinMulti")
        num_atoms_col = titles.index("NumAtoms")
        num_elems_col = titles.index("NumElems")
        composition_col = titles.index("Composition")
        elements_col = titles.index("listElements")
        ie_col = titles.index("dictIEs")
        ea_col = titles.index("dictEAs")
        redox_level_col = titles.index("dictRedoxLevels")
        ox_pot_col = titles.index("dictOxPots")
        red_pot_col = titles.index("dictRedPots")

        for index in range(1, len(lines)):
            joined_line = lines[index].strip(" ")
            joined_line = joined_line.strip("\n")
            items = joined_line.split("$")
            assert len(items) > 8

            mol = MPculesMol(items[id_col])
            mol.inchi = items[inchi_col]
            mol.solvent = items[solvent_col].split("&")
            mol.level = items[level_col].split("&")
            mol.charge = items[charge_col]
            mol.num_atoms = items[num_atoms_col]
            mol.num_elements = items[num_elems_col]
            mol.composition = items[composition_col]
            mol.elements = items[elements_col].split("&")

            dict_items = items[ie_col].split("&")
            for index in range(0, len(dict_items)):
                key_and_value = dict_items[index].split(":")
                if key_and_value[0].find("DIELECTRIC=18") > -1:
                    mol.solvent_to_ie[key_and_value[0]] = key_and_value[1]

            dict_items = items[ea_col].split("&")
            for index in range(0, len(dict_items)):
                key_and_value = dict_items[index].split(":")
                if key_and_value[0].find("DIELECTRIC=18") > -1:
                    mol.solvent_to_ea[key_and_value[0]] = key_and_value[1]

            dict_items = items[ox_pot_col].split("&")
            for index in range(0, len(dict_items)):
                key_and_value = dict_items[index].split(":")
                if key_and_value[0].find("DIELECTRIC=18") > -1:
                    mol.solvent_to_oxidation_pot[key_and_value[0]] = key_and_value[1]

            dict_items = items[red_pot_col].split("&")
            for index in range(0, len(dict_items)):
                key_and_value = dict_items[index].split(":")
                if key_and_value[0].find("DIELECTRIC=18") > -1:
                    mol.solvent_to_reduction_pot[key_and_value[0]] = key_and_value[1]

            mpcules_mols.append(mol)

    # output
    with open(INPUT_DIR / "IE-Ox.csv", "w") as f:
        f.write("MolId;Ionization potential (eV);Oxidation free energy (eV)\n")
        for index in range(0, len(mpcules_mols)):
            ip = [float(mpcules_mols[index].solvent_to_ie[key]) for key in mpcules_mols[index].solvent_to_ie.keys()]
            # oxidation free energy = (original oxidation potential vs H+/H + 4.44)
            ox_free_ener = [float(mpcules_mols[index].solvent_to_oxidation_pot[key]) + 4.44 for key in mpcules_mols[index].solvent_to_oxidation_pot.keys()]
            assert len(ip) == 1 and len(ox_free_ener) == 1
            f.write("%s;%f;%f\n" % (mpcules_mols[index].id, ip[0], ox_free_ener[0]))
        f.close()

    with open(INPUT_DIR / "EA-Red.csv", "w") as g:
        g.write("MolId;Electron affinity (eV);Reduction free energy (eV)\n")
        for index in range(0, len(mpcules_mols)):
            # print("EA in the paper is published as the negative value of classical EA!!!!")
            # converted original defination (EA = {A-} - {A0}) to common defination: {A0} - {A-}
            # has been verified by my EAIP calculation model_name (published values have negative relationship to my calculated common EA)
            ea = [-1.0 * float(mpcules_mols[index].solvent_to_ea[key]) for key in mpcules_mols[index].solvent_to_ea.keys()]
            # 1) reduction free energy = {A-} - {A0} = (original reduction potential vs H+/H ) + 4.44,
            # where original reduction potential confilts to common definations.
            # 2) but in the downloaded data, they keep using the common defination {A-} - {A0},
            # because there is a negative relationship between original EA = {A-} - {A0} and original reduction free enegy {{A-} - {A0}}
            # which is in-consistent with the equtions.
            red_free_ener = [-1.0 * (float(mpcules_mols[index].solvent_to_reduction_pot[key]) + 4.44) for key in mpcules_mols[index].solvent_to_reduction_pot.keys()]
            assert len(ea) == 1 and len(red_free_ener) == 1
            g.write("%s;%f;%f\n" % (mpcules_mols[index].id, ea[0], red_free_ener[0]))
        g.close()

#
def linear_fitting_to_free_eners():

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model_name = "LR-EAIP-RedoxFreeEner"
    model_dir = REDOX_ENER_DIR / model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    files_for_fitting = [INPUT_DIR/"IE-Ox.csv", INPUT_DIR/"EA-Red.csv"]

    x_col = 1
    y_col = 2
    all_x_data = []
    all_y_data = []
    all_x_label = []
    all_y_label = []
    for index in range(0, len(files_for_fitting)):

        raw_data = np.genfromtxt(files_for_fitting[index], np.dtype(str), comments="$", delimiter=";")

        # read label and remove label line
        x_label = np.squeeze(raw_data[0:1, x_col:x_col+1])
        all_x_label.append(x_label)

        y_label = np.squeeze(raw_data[0:1, y_col:y_col + 1])
        all_y_label.append(y_label)

        raw_data = np.delete(raw_data, 0, axis=0)

        # read X and Y data
        x_data = np.squeeze(raw_data[:,x_col:x_col+1]).astype(np.float64)
        all_x_data.append(x_data)

        y_data = np.squeeze(raw_data[:,y_col:y_col+1]).astype(np.float64)
        all_y_data.append(y_data)

    # modify font
    plt.rc('font', family='sans-serif', size=8)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rc('text', usetex=True)

    # modify the direction of axes
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')

    # figsize is the Width, height in inches of whole plt
    fig, axs = plt.subplots(1, len(all_y_data), figsize=(3.3*len(all_y_data), 3.3), dpi=300)

    # plot
    for index in range(0, len(all_y_data)):

        # 2D Kernel density estimation, https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation
        xy = np.vstack([all_x_data[index], all_y_data[index]])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = all_x_data[index][idx], all_y_data[index][idx], z[idx]
        # z has the same dimension with x and y
        scatter = axs[index].scatter(x, y, c=z, marker='o', linewidths=0.2, alpha=1.0, s=45)
        cbar = plt.colorbar(scatter)

        slope, intercept = np.polyfit(all_x_data[index],all_y_data[index], 1)
        x_base = np.linspace(np.min(all_x_data[index]), np.max(all_x_data[index]), 500)
        y_base = x_base * slope + intercept
        axs[index].plot(x_base, y_base, marker='o', markersize=0, markeredgewidth=0,
                 linestyle='--', color='red', linewidth=1)

        # Make predictions using the testing set
        pred_y = all_x_data[index] * slope + intercept
        mae = mean_absolute_error(all_y_data[index], pred_y)
        mse = mean_squared_error(all_y_data[index], pred_y)
        rmse = math.sqrt(mse)

        text_pos1 = (0.6, 0.9)
        text_pos2 = (0.65, 0.8)
        if index == 1:
            text_pos1 = (0.85, 0.9)
            text_pos2 = (0.92, 0.8)

        if intercept > 0.0:
            axs[index].text(text_pos1[0], text_pos1[1], "$y = %.2f x + %.2f$"%(slope,intercept), transform=axs[index].transAxes,
                             fontsize=8, fontweight='bold', va='top',ha='right')
        else:
            axs[index].text(text_pos1[0], text_pos1[1], "$y = %.2f x %.2f$"%(slope,intercept), transform=axs[index].transAxes,
                             fontsize=8, fontweight='bold', va='top',ha='right')
        axs[index].text(text_pos2[0], text_pos2[1], "MAE$=%.2f$, RMSE$=%.2f$"%(mae, rmse), transform=axs[index].transAxes,
                         fontsize=8, fontweight='bold', va='top',ha='right')

        # set label
        axs[index].set_xlabel(all_x_label[index])
        axs[index].set_ylabel(all_y_label[index])

    fig.tight_layout(pad=0.5)

    output_file = model_dir / "EAIP_Redox.jpg"
    plt.savefig(fname=output_file, dpi=300)
    plt.close()

#
if __name__ == "__main__":

    get_mpcules_mol_prop()

    linear_fitting_to_free_eners()
    