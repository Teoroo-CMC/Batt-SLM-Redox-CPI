#  This repository implements a unified generative–predictive framework for molecular design, combining generative AI, redox property prediction, and chemical–protein interaction (CPI) modeling to enable multi-objective molecular optimization.

# Project structure
```text
├── Batt-SLM/                # generator input molecules
│   ├── molecules.smi        
│   └── ...
│
├── redox_predictor/         # 
│   ├── models/
│   ├── data/
│   ├── train.py
│   ├── inference.py
│   └── README.md
│
├── cpi_predictor/           # Chemical-Protein Interaction (CPI)
│   ├── models/
│   ├── data/
│   ├── train.py
│   ├── inference.py
│   └── README.md
├── results/                 
└── README.md
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
