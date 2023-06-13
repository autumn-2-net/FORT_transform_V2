import h5py


with h5py.File('V2_dataset_stage2.hdf5', 'r') as f:
    for i in f:
        print(i)
    aaa=f['0_1000'][()]
    pass