"""
Script to detect spikes

Will
"""

USER = 'Antonin_thalrec'
DOONLYCAT = True  # do only file names contenanting concated (to do on combined dat files)
DOONLYFOLDER = None  # '151009_AF75'] # do only the given folder
SKIPFOLDER = ['151002_AF64']

__author__ = 'blota'

import os, shutil
import time
from phy import session
from phy.io import create_kwik
from . import param_template as params
from .profiles import profile

doit = input('Do you want to do it (y for yes anything else cancels): ')

if doit == 'y':
    overwrite = input(
        'Do you want to do kwik file sthat already have spikes in them? (yes for yes anything else is no): ')
    if overwrite == 'yes':
        print('overwriting')
        overwrite = True
    else:
        overwrite = False

    j_ = os.path.join

    detekt_path = profile[USER]['detekt_path']
    logfile = j_(detekt_path, 'log_detekt_%s.txt' % time.strftime("%y%m%d_%H%M%S"))


    def log(txt):
        """Print and write text to log file
        """
        print(txt)
        with open(logfile, 'a') as F:
            F.write(txt + '\n')


    log('Starting detektion')

    for fdir in os.listdir(detekt_path):
        print(fdir)
        if fdir in SKIPFOLDER:
            continue
        if not os.path.isdir(j_(detekt_path, fdir)):
            continue
        if DOONLYFOLDER is not None and fdir not in DOONLYFOLDER:
            continue
        log('---- Doing %s ----' % fdir)
        daydir = j_(detekt_path, fdir)

        # copy probe file to the directory
        for expname in os.listdir(j_(detekt_path, fdir)):
            if not expname.endswith('.kwik'):
                continue  # work only with dat files
            if DOONLYCAT and 'concated' not in expname:
                continue
            print(expname)
            # if '1_partstable' not in expname: # temporary addition for 151012_AF74
            #     continue
            log('    --- Exp %s ---' % expname)
            log('        -- create param file --')

            # detekt the spikes
            log('        -- detekting --')
            sess = session.Session(j_(daydir, expname))
            if sess.model.n_spikes:
                if not overwrite:
                    print('Already spikes in %s, use overwrite if you want to detect anyway')
                    continue
                else:
                    print('Already spikes in %s, overwriting')
            sess.detect()
            log('        -- done --')
