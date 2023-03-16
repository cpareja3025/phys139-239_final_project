import uproot
import awkward as ak

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import h5py

### set I/O path
data_dir = Path.cwd().parent.parent.joinpath('data')
root_dir = data_dir.joinpath('root')
h5_dir = data_dir.joinpath('hdf5')
h5_dir.mkdir(parents=True, exist_ok=True)

root_train_path = root_dir.joinpath('train_50k.root')
root_test_path = root_dir.joinpath('test_40k.root')
h5_train_path = h5_dir.joinpath('train.h5')
h5_test_path = h5_dir.joinpath('test.h5')

# look-up table
pdgId2vec = {}
pdgId2vec[11] = np.array([1,0,0,0,0], dtype=float)
pdgId2vec[13] = np.array([0,1,0,0,0], dtype=float)
pdgId2vec[22] = np.array([0,0,1,0,0], dtype=float)
pdgId2vec[211] = np.array([0,0,0,1,0], dtype=float)
pdgId2vec[2212] = np.array([0,0,0,0,1], dtype=float)

def root2npy(fpath, N, start=0, interval=1000):
    intv = interval

    f = uproot.open(fpath)
    result_img, result_target = [], []

    for i in tqdm(range(int(N/intv))):
        entry_start = start+i*intv
        entry_stop = entry_start+intv
        ak_img = f["image2d_data_tree"].arrays("_image_v._img", entry_start=entry_start, entry_stop=entry_stop)["_image_v._img"]
        ak_target = f['particle_mctruth_tree'].arrays('_part_v._pdg', entry_start=entry_start, entry_stop=entry_stop)['_part_v._pdg']

        mask = ak.count(ak_img, 1)==3
        mask = ak.all(mask, -1)
        ak_img = ak_img[mask]
        ak_target = ak_target[mask]

        np_img = ak.to_numpy(ak_img)
        np_img = np_img.reshape(-1, 3, 256, 256)
        np_target = ak.to_numpy(ak_target).reshape(-1)
        np_target = np.array([pdgId2vec[pdgId] for pdgId in np_target], dtype=float)

        result_img.append(np_img)
        result_target.append(np_target)

    result_img = np.concatenate(result_img, axis=0)
    result_target = np.concatenate(result_target, axis=0)

    return result_img, result_target

def npy2h5(h5, N, x_name, x_npy, y_name, y_npy):
    with h5py.File(h5, 'a') as hf:
        if x_name not in hf.keys():
            hf.create_dataset(x_name, (N, 3, 256, 256), maxshape=(None,3,256,256), dtype='f', chunks=True)
            hf[x_name][-x_npy.shape[0]:] = x_npy
        else:
            hf[x_name].resize((hf[x_name].shape[0] + x_npy.shape[0]), axis = 0)
            hf[x_name][-x_npy.shape[0]:] = x_npy
        
        if y_name not in hf.keys():
            hf.create_dataset(y_name, (N,5), maxshape=(None,5), dtype='f', chunks=True)
            hf[y_name][-y_npy.shape[0]:] = y_npy
        else:
            hf[y_name].resize((hf[y_name].shape[0] + y_npy.shape[0]), axis = 0)
            hf[y_name][-y_npy.shape[0]:] = y_npy
    return

for i in range(5):
    interval = 10000
    start = i*interval
    X, y = root2npy(root_train_path, N=interval, start=start, interval=1000)
    npy2h5(h5_train_path, interval, 'X_train', X, 'y_train', y)

for i in range(4):
    interval = 10000
    start = i*interval
    X, y = root2npy(root_test_path, N=interval, start=start, interval=1000)
    npy2h5(h5_test_path, interval, 'X_test', X, 'y_test', y)
