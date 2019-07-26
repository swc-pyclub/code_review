__author__ = 'blota'

"""
Take dat files and write command to concatenate them in a file

That will work only for my experiments since I need info for the different probe positions. But feel free to adapt to
make it more general
"""

import os
import os.path as op
import sys, shutil
import numpy as np
import re
# from setup_fran.labbook_data import read_protocol
from .profiles import profile
import time
import pandas as pd

# '151007_AF63',
# '151006_AF65',
# '151007_AF67']


def concatenate_dat_files(gp_df, col_to_group, order, root_col, filename_col, numsample, numchan, log_func=print,
                          prefix=None, subfolder=None):
    """Single iteration of concatenation. Updated Documentation in cli.concatenatedat

    col_to_group is a list of tuples

    CLI documentation: (might be outdated, see concatenatedat for updated version)
    Given a csv, concatenate dat files.

    The csv must have a single header line and then one line per file.
    A template can be found in OpenEphys/oe_clustering

    All the files that have the same values for "col_to_group" are concatenated together
    A csv file with the same name as the output dat file is also created. If numchan and
    numsamples columns exists, then this csv will contain the info about the border of
    each file in the concatenated dat file.

    root_col can either be a column or an absolute path

    Example:

        Given a csv like that

    File name, Exp,  Shank,  Home directory,Ordering key, Pos,  Mouse
    file1.dat, exp1, tet1,    /tmp,            1,         1,      M1
    file2.dat, exp1, tet2,    /tmp,            2,         1,      M1
    file3.dat, exp1, tet1,    /tmp,            4,         1,      M1
    file4.dat, exp1, tet2,    /tmp,            5,         1,      M1
    file5.dat, exp1, tet1,    /tmp,            6,         2,      M1
    file6.dat, exp2, tet1,    /tmp,            3,         1,      M1


        oe_clustering concatenatedat myfile.csv -c Shank shk -c Pos pos

        # will execute:

        cat /tmp/file1.dat /tmp/file6.dat /tmp/file3.dat > /tmp/concatenated_pos_1-shk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/concatenated_pos_1-shk_tet2.dat

        cat /tmp/file5.dat > /tmp/concatenated_pos_2-shk_tet1.dat

        # note the inversion between file6 and file3 following the ordering key

        oe_clustering concatenatedat myfile.csv -c Shank shk -c "Pos" pos -c "Exp" exp

        # will move file6 that was from another exp

        cat /tmp/file1.dat /tmp/file3.dat > /tmp/concatenated_pos_1-exp_exp1-shk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/concatenated_pos_1-exp_exp1-shk_tet2.dat

        cat /tmp/file6.dat > /tmp/concatenated_pos_1-exp_exp2-shk_tet1.dat

        cat /tmp/file5.dat > /tmp/concatenated_pos_2-exp_exp1-shk_tet1.dat

        oe_clustering concatenatedat myfile.csv -p Mouse -c Shank myshk

        # will add a prefix to output names

        cat /tmp/file1.dat /tmp/file6.dat /tmp/file3.dat /tmp/file5.dat > /tmp/M1_concatenated_myshk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/M1_concatenated_myshk_tet2.dat



    """

    if prefix is None:
        file_name = ''
    else:
        file_name = gp_df[prefix].iloc[0] + '_'
    file_name += 'concatenated_'
    file_name += '-'.join(['%s_%s' % (v, gp_df[k].iloc[0])
                           for k, v in col_to_group])
    csv_name = file_name + '_file_origin.csv'
    file_name += '.dat'

    # Check if root col is a column or an absolute path
    if not hasattr(gp_df, root_col) and root_col.startswith('/'):
        log_func('Root is an absolute path.')
        source_folder = root_col
    else:
        source_folder = gp_df[root_col].iloc[0]
        if not all(gp_df[root_col] == source_folder):
            log_func('WARNING: multiple home folders, will keep the first one')


    log_func(source_folder)
    if not op.isdir(source_folder):
        log_func('Target folder doesn\'t exists. Skip')
        return

    target_folder = source_folder
    if subfolder is not None:
        target_folder = os.path.join(source_folder, subfolder)

    target_file = os.path.join(target_folder, file_name)
    target_csv = os.path.join(target_folder, csv_name)
    ordering_series = gp_df[order].sort_values()
    source_files = [os.path.join(source_folder, gp_df.ix[i][filename_col])
                    for i in ordering_series.index]
    # Writing the csv file
    d = dict(origin_file=source_files, ordering=ordering_series.values)
    do = 0
    if numsample in gp_df.columns:
        do += 1
        ns = np.hstack([[0], gp_df[numsample].ix[ordering_series.index].values])
        d['beg_sample'] = ns.cumsum()[:-1]
        d['end_sample'] = ns.cumsum()[1:]
    if numchan in gp_df.columns:
        do += 1
        nc = gp_df[numchan].ix[ordering_series.index].values
        d['num_channels'] = nc
    if do == 2:
        nf = ns * np.hstack([[0], nc])
        d['beg_flat'] = nf.cumsum()[:-1]
        d['end_flat'] = nf.cumsum()[1:]
    csv_df = pd.DataFrame(d, index=range(len(source_files)), )
    if os.path.exists(target_csv):
        log_func('    File %s already exists' % target_csv)
        log_func('    skipping')
    else:
        csv_df.to_csv(target_csv)

    if os.path.exists(target_file):
        log_func('    File %s already exists' % target_file)
        log_func('    skipping')
    else:
        bashcommand = r'cat %s > %s' % (' '.join(source_files), target_file)
        log_func('    %s' % bashcommand)
        os.system(bashcommand)
    return

# To order the files temporally I find the year, month, day, hour, minutes info, concatenate in a integer and sort
pattern = '.*(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)-(\d\d)-(\d\d).*'


# oe_dat_path = globs.DATA_PATH['oe_data']

def concatenate_dat_files_legacy(userprofile, folder2cat=None, verbose=True):
    oe_dat_path = profile[userprofile]['target_path']
    commandfile = op.join(oe_dat_path, 'command2cat_%s.txt' % time.strftime("%y%m%d_%H%M%S"))

    # overwrite file
    F = open(commandfile, 'w')
    F.close()

    for folder in os.listdir(oe_dat_path):
        if folder2cat is not None and folder not in folder2cat:
            continue
        print('\n\n----- Doing %s -----' % folder)
        folder_fp = op.join(oe_dat_path, folder)
        if not os.path.isdir(folder_fp):
            continue

        target_folder = op.join(profile[userprofile]['target_path'], folder)
        if not os.path.isdir(target_folder):
            raise IOError('%s is not created!' % target_folder)

        mouse_name = folder.split('_')[1]
        mouse_prot = read_protocol(mouse_name)

        for pos, df in mouse_prot.groupby('electrode pos'):
            print('   -- Doing pos %s --' % pos)
            # file names need to be sorted by date
            expnames = list(df.oe_file.value_counts().index)
            datadict = {}
            for exp in expnames:
                match = re.match(pattern, exp)
                if match is None:
                    print('no match')
                    print(exp)
                    continue
                match = match.groups()
                datadict[int(''.join(match))] = exp
            sortedfile = sorted(datadict.keys())

            nmshk = profile[userprofile]['num_shanks']
            if isinstance(nmshk, dict):
                iterator = nmshk.keys()
            else:
                iterator = range(nmshk)
            for shk in iterator:
                print('       - Doing shk %s -' % shk)
                file_names_shk = [
                    op.join(folder_fp, datadict[k] + '_kwd_%s_rec0_shk%s.dat' % (profile[userprofile]['process'],
                                                                                 shk)) for k in sortedfile]
                outfile = op.join(folder_fp, '%s_concated_dat_pos%s_shk%s.dat' % (folder, pos, shk))
                if os.path.isfile(outfile):
                    if verbose:
                        print('File already exists!\n   skipping %s' % outfile)
                    continue
                bashcommand = r'cat %s > %s' % (' '.join(file_names_shk), outfile)
                with open(commandfile, 'a') as F:
                    F.write(bashcommand)
                    F.write('\n')
                if DOIT:
                    os.system(bashcommand)
