"""
Describe the content of a folder created by OpenEphys GUI

Should return a dataframe with useful info but be fast enough
"""

import os
import pandas as pd
import h5py

file_ext = ('.kwd', '.kwe', '.kwx', '.xml')  # file extension of oe files


def oe_info(folder):
    df = pd.DataFrame(columns=['fname', 'file_type', 'process', 'rec_type', 'num_rec'])
    for fname in os.listdir(folder):
        root, ext = os.path.splitext(fname)
        if ext in file_ext:
            d = pd.DataFrame([[fname, ext]], columns=['fname', 'file_type'])
            df = df.append(d, ignore_index=True)

    # get process and rec_type for kwd files, rec_num for kwd and kwe
    for ind, line in df.iterrows():
        if line.file_type == '.kwd':
            fname_part = line.fname.split('.')
            line['rec_type'] = fname_part[-2]
            root = '.'.join(fname_part[:-2])
            process = int(root.split('_')[-1])
            line['process'] = process
        if line.file_type in ('.kwd', '.kwe'):
            h5file = h5py.File(os.path.join(folder, line.fname), mode='r')
            recs = h5file.get('/recordings/')
            line['num_rec'] = len(recs)
            h5file.close()
        df.ix[ind] = line
    return df


if __name__ == '__main__':
    # should move to test at some point
    folder = '../tests/test_data_set'
    print(oe_info(folder))
