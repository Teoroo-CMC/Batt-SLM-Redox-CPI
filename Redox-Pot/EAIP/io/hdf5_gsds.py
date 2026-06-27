# -*- coding: utf-8 -*-
"""
A hdf5 data loader for the EAIP dataset build by Zhan-Yun Zhang, which has the following format
    begin
    lattice float float float
    lattice float float float
    lattice float float float
    atom floatcoordx floatcoordy floatcoordz int_atom_symbol floatq 0  floatforcex floatforcey floatforcez
    atom 1           2           3           4               5      6  7           8           9
    energy float
    charge float
    comment arbitrary string
    end
Due to the flexibility of the hdf5 format, if you want to load other datasets, you need to modify this file
"""

def _hdf5_spec(label_map):
    """Returns format dict for the QM9 dataset"""
    ds_spec = {
        'elems': {'dtype':  'int32',   'shape': [None]},
        'coord': {'dtype':  'float', 'shape': [None, 3]}}
    for k, v in label_map.items():
        ds_spec[k] = {'dtype':  'float', 'shape': []}
    return ds_spec

def _gen_mol_list(fname):
    import h5py
    from tqdm.auto import tqdm
    mol_list = []
    with h5py.File(fname, 'r') as f:
        for group_name in tqdm(f):
            mol_list.append((fname, group_name))
    return mol_list

def load_hdf5(flist, label_map={'e_data': 'ea'}, splits=None, shuffle=True, seed=0, subset=None):
    """
    Loads hdf5 formatted file
    Args:
        flist (str): one or a list of hdf5 formatted files
        label_map (dict): label-dataset pairs specifying the training label
        splits (dict): key-val pairs specifying the ratio of subsets. If subset is used, then the splitting is on a subset
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
        subset (int): the number of data points in a random selected subset
    """
    import numpy as np
    import h5py
    from tqdm.auto import tqdm
    from pinn.io.base import list_loader
    from ase.data import atomic_numbers

    @list_loader(ds_spec=_hdf5_spec(label_map))
    def _hdf5_loader(fname_mol):

        fname, mol = fname_mol
        with h5py.File(fname, 'r') as f:

            group = f[mol]
            rawData = dict((k, v[()]) for k, v in group.items())

            elems = [atomic_numbers[elem.decode("utf-8")] for elem in rawData["elems"]]
            elems = np.array(elems, np.int32)
            coord = np.array(rawData["coord"], float)
            returnData = {'elems': elems, 'coord': coord}

            for k, v in label_map.items():
                if k == "d_data":
                    returnData[k] = float(np.linalg.norm(rawData[v]))
                else:
                    assert len(rawData[v]) == 1
                    returnData[k] = float(rawData[v])

            return returnData

    if isinstance(flist, str):
        flist = [flist]
    mol_list = []
    for fname in flist:
        mol_list += _gen_mol_list(fname)

    # take a subset to get the learning curve
    if subset is not None:
        np.random.seed(seed)
        sel_index = np.random.choice(range(0, len(mol_list)), subset, replace=False)
        mol_list = [mol_list[i] for i in sel_index]

    return _hdf5_loader(mol_list, splits=splits, shuffle=shuffle, seed=seed)
