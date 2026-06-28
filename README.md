# Batt-SLM
This repository provides the battery solvent-like molecules (Batt-SLM) dataset, redox potential prediction models and chelation propensity index (CPI) associated with the preprint "Z.-Y. Zhang, R. Mercado., T. T. Le, C. Zhang, Unlocking the Chemical Space for Rechargeable Batteries with Generative Solvent Design System (2026)" (https://doi.org/10.26434/chemrxiv.15001594/v1).

# Project structure
In the following project structure, _*_ denotes the random seed, _<...>_ indicates an arbitrary sequence of characters.
```text
.
├── Batt-SLM/                               # Directory: the battery solvent-like molecules
│   ├── Batt-SLM.smi                        # The Batt-SLM for training the generators
│   └── filter_smiles.py                    # The python code to filter molecules from given SMILES
├── Batt-P30K/                               
│   ├── Batt-P30K.h5                        # The Batt-P30K dataset 
│   ├── io/                                 # The dataloader for Batt-P30K.h5 in the PiNN package 
│   │   │                                   (https://github.com/Teoroo-CMC/PiNN/tree/master)
│   │   ├── __init__.py                     # The python code to initialize the dataloader
│   │   └── hdf5_gsds.py                    # The python code to load the Batt-P30K dataset
│   ├── Models/                          
│   │   └── {HOMO,LUMO,IP,EA,Dipole}/       # The PiNet2 models  
│   │       └── PiNet2-<...>-B10-3E6-*/     # <...> is a key in {HOMO,LUMO,IP,EA,Dipole}
│   │           ├── eval/events.<...>       # The validation event file
│   │           ├── checkpoint              # The file storing the paths of actual checkpoint files
│   │           ├── params.yml              # The hyper-parameter file
│   │           ├── graph.pbtxt             # The text-format file of TensorFlow computation graph
│   │           ├── events.<...>            # The training event files
│   │           └── model.ckpt-<...>        # The actual Tensorflow checkpoint files
│   └── build_pinet2.py                     # The python code to build PiNet2 models
├── Redox-Pot/                              # Directory: the ML models for redox potential prediction
│   ├── Redox-Ener/                         # The linear fitting of EA/IP vs redox free energies
│   │   ├── RX-392.csv                      # The RX-392 dataset
│   │   ├── Input/
│   │   │   ├── IE-Ox.csv                   # The IP and oxidation free energy in RX-392 dataset
│   │   │   └── EA-Red.csv                  # The EA and reduction free energy in RX-392 dataset
│   │   ├── LR-EAIP-RedoxFreeEner/                
│   │   │   └── EAIP_Redox.jpg              # The linear fitting results
│   │   └── redox_free_ener.py              # The python code for linear fitting
│   └── redox_potential.py                  # The python code for predicting redox potential         
├── CPI/                                    # Diectory: the chelation propensity index (CPI) model
│   ├── SolvFunc-87.csv                     # The functions of collected solvents in battery
│   ├── Input/
│   │   ├── Features.csv                    # The features of all solvents in Fig. 5 of main text
│   │   └── Features-NoF.csv                # The features of non-F solvents to build the CPI
│   ├── LogR-CPI/
│   │   ├── result.txt                      # The logistic regression results for CPI
│   │   └── Features-NoF-predictions.csv    # The prediction results on mols without F atoms
│   ├── RandomTest/
│   │   ├── final.txt                       # The final evaluation results of CPI
│   │   ├── result-*.txt                    # The evaluation results of CPI at a given seed
│   │   ├── train_*_results.csv             # The training results at a given seed
│   │   └── valid_*_results.csv             # The validation results at a given seed
│   ├── ExternalTest/
│   │   ├── results.txt                     # The external test results of CPI
│   │   ├── external_set.csv                # The new collected WSE and CDE
│   │   ├── external_set-filters.csv        # The new collected WSE and CDE passed filters
│   │   ├── external_set-filters-features.csv # The features of filtered WSE and CDE
│   │   └── external_set-filters-features-predictions.csv # The prediction results
│   ├── chelation_propensity_index.py       # The python code for CPI
│   ├── random_test.py                      # The python code for evaluation
│   └── external_test.py                    # The python code for external test
└── environment.yml                         # The conda environment file for the project
```

# Structure of the Batt-P30K dataset
The dataset is stored in HDF5 (.h5) format. Each molecule is saved as an individual HDF5 group, where the group name corresponds to the molecule ID. Every group contains the molecular structure together with its computed electronic properties at the ωB97X-V/def2-TZVPPD/SMD(ε=18.5) level of theory.

```text
Batt-P30k.h5
├── CompMol0
│   ├── smiles          # SMILES string
│   ├── elems           # Atomic element symbols
│   ├── coord           # Cartesian coordinates
│   ├── ener            # Neutral molecule energy
│   ├── ener_anion      # Anion energy
│   ├── ener_cation     # Cation energy
│   ├── homo            # HOMO energy
│   ├── lumo            # LUMO energy
│   ├── gap             # HOMO-LUMO gap
│   ├── ea              # Electron affinity (EA)
│   ├── ip              # Ionization potential (IP)
│   ├── dipole          # Dipole moment
│   └── quadrupole      # Quadrupole moment
├── CompMol1
│   └── ...
└── ...
```

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `smiles` | `(1,)` | `string` | Canonical SMILES representation of the molecule. |
| `elems` | `(N_atoms,)` | `string` | Atomic element symbols. |
| `coord` | `(N_atoms, 3)` | `float32` | Cartesian coordinates of all atoms (Å). |
| `ener` | `(1,)` | `float32` | Total energy of the neutral molecule. |
| `ener_anion` | `(1,)` | `float32` | Total energy of the anion. |
| `ener_cation` | `(1,)` | `float32` | Total energy of the cation. |
| `homo` | `(1,)` | `float32` | Highest Occupied Molecular Orbital (HOMO) energy. |
| `lumo` | `(1,)` | `float32` | Lowest Unoccupied Molecular Orbital (LUMO) energy. |
| `gap` | `(1,)` | `float32` | HOMO–LUMO energy gap. |
| `ea` | `(1,)` | `float32` | Electron affinity (EA). |
| `ip` | `(1,)` | `float32` | Ionization potential (IP). |
| `dipole` | `(3,)` | `float32` | Dipole moment `(x, y, z)`. |
| `quadrupole` | `(6,)` | `float32` | Six independent components of the quadrupole moment. |

# Installation
+ download the project repo
```
git clone https://github.com/Teoroo-CMC/Batt-SLM.git
```
+ create the conda environment
```
cd Batt-SLM
conda env create -f environment.yml
conda activate gsds
```
+ install the PiNN package
```
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
cp -r {PROJ_DIR}/Batt-P30K/io {PiNN_DIR}/
```

# Usage
+ training PiNet2 models on GPU
```
# Note: you may need to manually download the Batt-P30K/Battery-P30K.h5 file
cd {PROJ_DIR}/Batt-P30K
# Change the target property and random seed in build_pinet2.py manually
python build_pinet2.py
```
+ predicting redox potentials for given XYZ files
```
cd {PROJ_DIR}/Redox-Pot/
mkdir XYZ
# copy your *.xyz files to XYZ/ folder
python redox_potential.py
```
+ predicting CPIs for given molecules
```
cd {PROJ_DIR}/CPI/
# using the predict_cpi function
python chelation_propensity_index.py
```
