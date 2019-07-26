"""
Small utils usefull but not enough to make it to the cli
"""
import os
import pandas as pd


def check_if_csv_as_all_dat_file(folder2dat, csvfile, col='File name', ext='.dat',
                                 ignore='concat'):
    """Check if all the dat files in folder2dat are in the csvfile
    """

    csvdf = pd.read_csv(csvfile)
    in_csv = pd.Index(csvdf[col].values)
    fl = []
    for dirpath, dirnames, filenames in os.walk(folder2dat):
        for f in filenames:
            if f.endswith(ext) and not (ignore in f):
                fl.append(f)
    in_dat = pd.Index(fl)
    in_csv_not_in_dat = in_csv.difference(in_dat)
    print('%i file in csv but not in dat' % len(in_csv_not_in_dat))
    in_dat_not_in_csv = in_dat.difference(in_csv)
    print('%i file in dat but not in csv' % len(in_dat_not_in_csv))
    return in_csv_not_in_dat, in_dat_not_in_csv


def find_dat_in_csv(folder, fnames, col='File name'):
    """Look in all csv in folder to find if there is fnames"""
    whereto = dict([(f, []) for f in fnames])
    for fn in os.listdir(folder):
        if fn.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder, fn))
            except Exception:
                print('Cannot open %s' % fn)
                continue
            if col not in df.columns:
                print('No %s column in %s' % (col, fn))
                continue
            for f in fnames:
                if f in df[col]:
                    whereto[f].append(fn)
    return whereto


if __name__ == '__main__':
    folder2dat = '/mnt/ssd/Antonin/Morgane/processed_data'
    csvfile = '/mnt/ssd/Antonin/Morgane/processed_data/datfile_list_morgane.csv'
    in_csv_not_in_dat, in_dat_not_in_csv = check_if_csv_as_all_dat_file(folder2dat, csvfile)
    whereto = find_dat_in_csv(folder2dat, in_dat_not_in_csv)
