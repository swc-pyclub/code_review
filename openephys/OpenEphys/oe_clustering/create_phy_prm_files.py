"""
Script to create prm files

Will
"""

__author__ = 'blota'

import os, shutil
import time
from . import param_template as params_module
from .profiles import profile


def move_stuff_in_folder(source_folder=None, move_prms=True):
    """Move every dat file and its associated param file from the current folder into a separate subfolder
    """

    if source_folder is None:
        source_folder = '.'
    fnames = os.listdir(source_folder)
    for fn in fnames:  # loop on all file/dir names
        if fn.endswith('.dat'):  # found a dat file
            # create destination folder if needed
            froot, ext = os.path.splitext(os.path.join(source_folder, fn))
            if os.path.isdir(froot):
                pass
            elif os.path.exists(froot):
                print('Something called %s already exists' % froot)
                print('skipping')
                continue
            else:
                os.mkdir(froot)

            # move dat file
            dest_dat = os.path.join(froot, fn)
            if os.path.exists(dest_dat):
                print('%s already exists!' % dest_dat)
                print('skipping')
                continue
            shutil.move(os.path.join(source_folder, fn), dest_dat)

            if move_prms:
                # move param file
                fparam = froot + '_params.prm'
                if not os.path.isfile(fparam):
                    print('param file not found for %s' % fn)
                    continue

                dest_dat = os.path.join(froot, fparam)
                if os.path.exists(dest_dat):
                    print('%s already exists!' % dest_dat)
                    print('skipping')
                    continue
            shutil.move(os.path.join(source_folder, fparam), dest_dat)


def write_param(expname, folder, params):
    datpath = expname if expname.endswith('.dat') else expname + '.dat'
    params['traces']['raw_data_files'] = [os.path.join(folder, datpath)]
    with open(os.path.join(folder, expname + '_params.prm'), 'w') as F:
        F.write("experiment_name = '%s'\n" % expname)
        for w in ['prb_file', 'spikedetekt', 'klustakwik2', 'traces']:
            F.write('%s=%s' % (w, repr(params[w])))
            F.write('\n')


def create_prm_for_file(expname, folderpath, shk_dict, probe_dict, origin_params, log_func=print, overwrite=False):
    """Given a path to a file (.dat), create corresponding prm file

     expname is the name of the file to use
     folderpath is the folder in which the file is
     shk_dict is a dictionary with shk names as keys and the list of channels as values
     probe_dict is a dictionary with shk names as keys and the name of the prb file as values
     origin_params is a dictionary containing the default parameters, only the following
     parameters will be updates:

     prb_file
     traces - n_channels
            - raw_data_files
    :return: 
    """

    which = [k for k in shk_dict.keys() if k in expname]
    if len(which) != 1:
        raise IOError('Cannot find which shank it is')
    shk = which[0]

    if shk not in probe_dict:
        log_func('no probe for %s. Not creating file' % shk)
        return
    params = origin_params.copy()
    params['prb_file'] = probe_dict[shk]
    params['traces']['n_channels'] = len(shk_dict[shk])

    target = os.path.join(folderpath, expname + '_params.prm')
    if os.path.isfile(target):
        if overwrite:
            log_func('%s already exists, ERASE' % target)
        else:
            log_func('%s already exists, skipping' % target)
            return
    write_param(expname, folder=folderpath, params=params)


def create_prm_for_folder(path2folder, profile_dict, origin_params, ext,
                          log_func=print, skip_warning=False, do_only=None,
                          overwrite=False, recursive=False):
    """Create a prm file for every file found in path2folder

    See cli tool create_prm_file for documentation (oe_clustering create_prm_file --help)
    profile_dict must contain num_shanks and probe_files
    """
    nfiles = 0
    for fname in os.listdir(path2folder):
        fullpath = os.path.join(path2folder, fname)
        level = len(fullpath.split(os.sep))
        if os.path.isdir(fullpath) and recursive:
            log_func('  ' * level + 'Descending into %s' % fname)
            create_prm_for_folder(fullpath, profile_dict, origin_params, ext,
                                  log_func, skip_warning, do_only,
                                  overwrite, recursive)
            continue
        if not fname.endswith(ext):
            continue
        if (do_only is not None) and (do_only not in fname):
            continue
        nfiles += 1
        log_func('  ' * level + 'Doing %s' % fname)
        create_prm_for_file(fname, path2folder, shk_dict=profile_dict['num_shanks'],
                            probe_dict=profile_dict['probe_files'], origin_params=origin_params,
                            log_func=log_func, overwrite=overwrite)
    if nfiles > 1 and (not skip_warning):
        log_func('WARNING: more than one prm file in %s' % path2folder)
        log_func('klusta works well only with a single prm file per folder')
    return


def create_prm_file_legacy(user_name, overwrite=False, do_only_folder=None, skip_folder=[], do_only_cat=False):
    os.path.join = os.path.join
    detekt_path = profile[user_name]['detekt_path']
    logfile = os.path.join(detekt_path, 'log_detekt_%s.txt' % time.strftime("%y%m%d_%H%M%S"))

    origin_params = dict([(k, getattr(params_module, k)) for k in ['prb_file', 'spikedetekt', 'klustakwik2', 'traces']])

    def log(txt):
        """Print and write text to log file
        """
        print(txt)
        with open(logfile, 'a') as F:
            F.write(txt + '\n')

    log('Starting creation')

    for fdir in os.listdir(detekt_path):
        print(fdir)
        if fdir in skip_folder:
            continue
        if not os.path.isdir(os.path.join(detekt_path, fdir)):
            continue
        if do_only_folder is not None and fdir not in do_only_folder:
            continue
        log('---- Doing %s ----' % fdir)
        daydir = os.path.join(detekt_path, fdir)

        num_shanks = profile[user_name]['num_shanks']
        probe_files = profile[user_name]['probe_files']
        if not isinstance(num_shanks, dict):
            num_chans = profile[user_name]['num_chans']
            num_shanks = dict([(shk, list(range(int(shk * num_chans / num_shanks),
                                                int((shk + 1) * num_chans / num_shanks))))
                               for shk in range(num_shanks)])

        for subfoldername in os.listdir(os.path.join(detekt_path, fdir)):
            detekt_path_sub = os.path.join(daydir, subfoldername)
            if not os.path.isdir(detekt_path_sub):
                continue
            for expname in os.listdir(detekt_path_sub):
                print(expname)
                if not expname.endswith('.dat'):
                    continue  # work only with dat files

                if do_only_cat and 'concated' not in expname:
                    continue
                expname = expname[:-4]

                log('    --- Exp %s ---' % expname)
                log('        -- create param file --')

                # create a param file with expname changed
                which = [k for k in num_shanks.keys() if k in expname]
                if len(which) != 1:
                    raise IOError('Cannot find which shank it is')
                shk = which[0]

                params = origin_params.copy()
                params['prb_file'] = probe_files[shk]
                params['traces']['n_channels'] = len(num_shanks[shk])

                target = os.path.join(detekt_path_sub, expname + '_params.prm')
                if (not overwrite) and os.path.isfile(target):
                    print('%s already exists, skipping' % target)
                    continue
                write_param(expname, folder=detekt_path_sub, params=params)
                log('        -- create kwik file --')
