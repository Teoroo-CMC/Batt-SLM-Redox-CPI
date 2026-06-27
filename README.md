# Batt-SLM-Redox-CPI
This repository provides the battery solvent-like molecules (Batt-SLM) dataset, redox potential prediction models and chelation propensity index (CPI) associated with the manuscript "Unlocking the Chemical Space for Rechargeable Batteries with a Generative Solvent Design System" (https://doi.org/10.26434/chemrxiv.15001594/v1).

# Project structure
In the following project structure, _*_ denotes the random seed, _<...>_ indicates an arbitrary sequence of characters.
```text
.
├── Batt-SLM/                               # Directory: the battery solvent-like molecules
│   ├── Batt-SLM.smi                        # The Batt-SLM for training the generators
│   ├── KBS-409.csv                         # The known battery solvents (KBS) dataset
│   ├── KBS-FP-174.smi                      # The molecules in KBS-409 with either F or P
│   └── filter_smiles.py                    # The python code to filter molecules from given SMILES
├── Redox-Pot/                              # Directory: the ML models for redox potential prediction
│   ├── EAIP/                               
│   │   ├── Batt-P30K.h5                    # The Batt-P30K dataset 
│   │   ├── io/                             # The dataloader for Batt-P30K.h5 in the PiNN package 
│   │   │   │                                  (https://github.com/Teoroo-CMC/PiNN/tree/master)
│   │   │   ├── __init__.py                 # The python code to initialize the dataloader
│   │   │   └── hdf5_gsds.py                # The python code to load the Batt-P30K dataset
│   │   ├── Models/                          
│   │   │   └── {IP,EA}/                    # The PiNet2 models of EA and IP  
│   │   │       └── PiNet2-<...>-B10-3E6-*/ # <...> is a key in {IP,EA}
│   │   │           ├── eval/events.<...>   # The validation event file
│   │   │           ├── checkpoint          # The file storing the paths of actual checkpoint files
│   │   │           ├── params.yml          # The hyper-parameter file
│   │   │           ├── graph.pbtxt         # The text-format file of TensorFlow computation graph
│   │   │           ├── events.<...>        # The training event files
│   │   │           └── model.ckpt-<...>    # The actual Tensorflow checkpoint files
│   │   └── build_pinet2.py                 # The python code to build PiNet2 models
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
│   ├── external_test.py                    # The python code for external test
└── environment.yml                         # The conda environment file for the project
```
# Installation
+ download the project repo
```
git clone https://github.com/zzy2014/GSDS.git
```
+ create the conda environment
```
cd Batt-SLM-Redox-CPI
conda env create -f environment.yml
conda activate gsds
```
+ install the PiNN package
```
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
cp -r {PROJ_DIR}/Redox-Pot/EAIP/io {PiNN_DIR}/
```

# Usage
+ training PiNet2 models for EA/IP on GPU
```
cd {PROJ_DIR}/Redox-Pot/EAIP
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
