# batt-slm-redox-cpi
This repository contains the battery solvent-like molecules (Batt-SLM) dataset, redox potential predictors and the chelation propensity index (CPI) associated with the manuscript "Unlocking the Chemical Space for Rechargeable Batteries with a Generative Solvent Design System" (https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001594/v1).

# Project structure
```text
Batt-SLM/
├── 1_Batt-SLM/                              # Directory: the battery solvent-like molecules
│   ├── Batt-SLM-PriorI.smi                  # The Batt-SLM for training the prior I
│   ├── KBS-409.csv                          # The placeholder for the KBS-409 dataset
│   ├── KBS-FP-174.smi                       # The placeholder for molecules in the KBS-409 with either F or P
│   └── filter_smiles.py                     # The python code to filter SMILES according to the 
│                                              selection criteria listed in Section 1 of the SI
├── 7_redox_free_ener/                       # Directory: the ML models for redox potential prediction
├── 2_Batt-P30K/                             # Directory: the Batt-P30K dataset
│   ├── Batt-P30K.h5                         # The placeholder for the Batt-P30K dataset 
│   ├── io/                                  # The dataloader for Batt-P30K.h5 in the PiNN package 
│   │   │                                      (https://github.com/Teoroo-CMC/PiNN/tree/master)
│   │   ├── __init__.py                      # The python code to initialize the dataloader
│   │   └── hdf5_gsds.py                     # The python code to load the Batt-P30K dataset
│   ├── Models/                               
│   │   └── {HOMO,LUMO,IP,EA,Dipole}/        # The PiNet2-P3 models of HOMO/LUMO/IP/EA/Dipole    
│   │       └── PiNet2-<...>-B10-3E6-*/      # <...> is a key in {HOMO,LUMO,IP,EA,Dipole}
│   │           ├── eval/events.<...>        # The validation event file
│   │           ├── checkpoint               # The file storing the paths of actual checkpoint files
│   │           ├── params.yml               # The hyper-parameter file
│   │           ├── graph.pbtxt              # The text-format file of TensorFlow computation graph
│   │           ├── events.<...>             # The training event files
│   │           └── model.ckpt-<...>         # The actual Tensorflow checkpoint files
│   └── build_pinet2.py                      # The python code to build PiNet2-P3 models
│   ├── RX-392.csv                           # The RX-392 dataset
│   ├── Input/
│   │   ├── IE-Ox.csv                        # The IP and oxidation potentials in RX-392 dataset
│   │   └── EA-Red.csv                       # The EA and reduction potentials in RX-392 dataset
│   ├── LR-EAIP-RedoxFreeEner/                
│   │   └── EAIP_Redox.jpg                   # The linear fitting results
│   └── redox_free_ener.py                   # The python code for linear fitting      
├── 8_CPI_index/                             # Diectory: the chelation propensity index (CPI) model
│   ├── SolvFunc-87.csv                      # The functions of collected solvents in battery
│   ├── Input/
│   │   ├── Features.csv                     # The features of all solvents in Fig. 5 of main text
│   │   └── Features-NoF.csv                 # The features of non-F solvents to build the CPI
│   ├── LogR-CPI/
│   │   ├── result.txt                       # The logistic regression results for CPI
│   │   └── Features-NoF-predictions.csv     # The prediction results on mols without F atoms
│   └── chelation_propensity_index.py        # The python code for CPI     
└── environment.yml                          # The conda environment file for gsds project
```
# Training PiNet2 models for EA/IP
```
git clone https://github.com/zzy2014/GSDS.git
```
+ create the conda environment
```
cd GSDS
conda env create -f environment.yml
conda activate gsds
```
+ install the PiNN package
```
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
cp -r {GSDS_DIR}/2_Batt-P30K/io {PiNN_DIR}/
```
+ install the GraphINVENT2 package
```
cd ..
git clone https://github.com/ailab-bio/GraphINVENT2.git
cp -r {GSDS_DIR}/9_molecular_generators/graphinvent GraphINVENT2
cp -r GraphINVENT2/graphinvent {GSDS_DIR}/9_molecular_generators/
```
+ export the gsds project path to PYTHONPATH
```
export PYTHONPATH={GSDS_DIR}:$PYTHONPATH
export PYTHONPATH={GSDS_DIR}/9_molecular_generators/graphinvent:$PYTHONPATH
```

# Build the CPI models
