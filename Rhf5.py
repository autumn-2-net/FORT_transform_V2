import h5py


with h5py.File('V2_dataset_stage21.hdf5', 'r') as f:
    # for i in f:
    #     print(i)
    aaa=f['34_266'][()]
print(aaa)

pass