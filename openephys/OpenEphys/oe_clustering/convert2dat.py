__author__ = 'blota'

"""
Write all the files to dat

Change the profile dictionnary in __init__.py to set the conversion parameters (mostly where to find the data, where
to write and should I reference)

numshanks in the __init__.py can be either an integer or a dictionary.
If it is an int the num_chans are evenly split in numshanks files, if it is a dictionary, keys are an ID for the
output file and values a list of canals.
"""

import os
import sys
from .. import oe_reader
from ..io import openephys2dat
from . import profiles
from .. import utils
import time, datetime
import re
from numpy import array

j_ = os.path.join

DEFAULT_USER = 'Francois'
FOLDER2CONVERT = '160415_CE2'


# None #['151008_AF73',
#                   '151012_AF74',
#                   '151009_AF75']  # ,
# '151007_AF63',
# '151006_AF65',
# '151007_AF67']


def convert_oefolder_to_dat(oefolder, target_folder, process, num_chans=None,
                            num_shanks=None, ref=None, prefix='', verbose=True,
                            log_func=None, level=0, csv_file=None, funcpy=None,
                            redo=False, n_cpu=None, chunk=None):
    """Convert .kwd file in a folder to dat files

    :param oefolder: OE folder to convert
    :param target_folder: str, path to save dat files
    :param process: int, processor to convert
    :param num_chans: int, used only if num_shanks is an integer. Total number
                      of channels to use (will start with channel 0)
    :param num_shanks: int or dict, if int: split num_chans data into num_shanks
                       equal parts and write to different files
                       if dict: keys are the shank names used in the output file
                       names, values are the list of channels (0 based) to use
                       for each shank
    :param ref: None, 'cmr' or 'car', if None save raw data, else subtract the
                common median or average before saving
    :param verbose: bool, print extra information on what I do
    :param prefix: str, name to add at the beginning of dat files
    :param redo: bool, re generate dat file if it already exists (default False)
    :param n_cpu: number of cpu for referencing
    :param chunk: chunk size for referencing
    :return: No output


    Examples:

    convert_oefile_to_dat(oef, 'path2target', 100, num_shanks =
                    {'kelly_shk1':range(32), 'tetrode':[32,33,34,35]})
    # Writes 'path2target/kwd_100_recX_shkkelly_shk1.dat' and
    # 'path2target/kwd_100_recX_shktetrode.dat' where X is the
    # the recording number

    convert_oefile_to_dat(oef, 'path2target', 101, num_chans = 64, num_shanks = 2, prefix = 'superdat')
    # Writes 'path2target/superdat_kwd_101_recX_shk0.dat' and 'path2target/superdat_kwd_101_recX_shk1.dat' where X is the
    # the recording number
    """
    # string used to match the date in OE file names
    match_string = '(\d{4})-([0,1]\d)-([0-3]\d)_([0-2]\d)-([0-6]\d)-([0-6]\d)'

    if log_func is None:
        log_func = print
    oefile = oe_reader.OpenEphysReader(oefolder, load_info=False, verbose=False)
    try:
        oefile.load_kwds_info(process, verbose=False)
    except AssertionError:
        log_func('    ' * level + 'Processor %i not found' % (process))
        return 1
    kwd = getattr(oefile, 'kwd_%i' % process)
    recs = kwd.file_info.recordings.values
    start_time = 0
    for rec_num in sorted(recs):
        if isinstance(num_shanks, dict):
            iterator = num_shanks.items()
        else:
            iterator = enumerate([list(range(int(shk * num_chans / num_shanks),
                                             int((shk + 1) * num_chans / num_shanks)))
                                  for shk in range(num_shanks)])
        rec = getattr(kwd, 'rec_%s' % rec_num)

        if len(prefix) and not prefix.endswith('_'):
            prefix += '_'

        for shk_id, chans in iterator:
            target_end = prefix + 'kwd_%s_rec%s_shk%s.dat' % (process, rec_num, shk_id)
            target_file = j_(target_folder, target_end)
            chans = utils.make_list(chans, int)

            if os.path.isfile(target_file) and not redo:
                if verbose:
                    log_func('    ' * level + 'File already exists!')
                    log_func('    ' * level + '   skipping %s' % target_end)
                    data = array([], ndmin=2)
                continue
            elif verbose:
                log_func('    ' * level + 'Doing %s' % target_end)

            if ref is not None:
                rec.ref_chans = chans
                rec.ref_method = ref
                try:
                    rec.load_ref()
                except IOError:
                    log_func('    ' * level + '    ref not found, compute it (will take time):\n')
                    rec.create_reference(n_cpu=n_cpu, chunk=chunk)
                    rec.save_ref()
                    log_func('    ' * level + '    done\n')
                data = rec.referenced_data
            else:
                data = rec
            try:
                openephys2dat.write_dat(target_file, data, chans, dtype='int16',
                                        chunksize=10 * 30000, funcpy=funcpy)
            except ValueError as ve:
                log_func('Could not write file. Do you have enough channels in your kwd?')
                log_func('Error: %s'%ve)

            if csv_file is not None:
                oefile_folder = os.path.split(oefile.folder.rstrip(os.sep))[-1]

                with open(csv_file, 'a') as F:
                    F.write('%s,' % target_end)
                    F.write('%s,' % prefix[:-1])  # write without the trailing _
                    F.write('%s,' % process)
                    F.write('%s,' % rec_num)
                    F.write('%s,' % shk_id)
                    F.write('%s,' % target_folder)

                    m = re.findall(match_string, oefile_folder)
                    if len(m) != 1:
                        message = 'Could not match a single date in this oe file name: %s' % oefile_folder
                        message += '\nDid you rename it?'
                        raise (IOError(message))
                    Y, M, D, h, m, s = [int(part) for part in m[0]]

                    # add time since beginning of recording
                    shift = datetime.timedelta(seconds=start_time)
                    d = datetime.datetime(Y, M, D, h, m, s) + shift

                    F.write(d.strftime('%Y%m%d%H%M%S') + ',')
                    F.write(d.strftime('%Y-%m-%d') + ',')
                    F.write(d.strftime('%H:%M:%S') + ',')
                    F.write('%s,' % oefolder)
                    F.write('%s,' % data.shape[0])
                    F.write('%s,' % len(chans))
                    F.write('%s,' % ' '.join([str(c) for c in chans]))
                    F.write('\n')
        if hasattr(kwd.file_info, 'sample_rate'):
            # it's a new kwd file
            sample_rate = kwd.file_info.sample_rate.iloc[0].mean()
        else:
            sample_rate = kwd.file_info.channel_sample_rates.iloc[0].mean()
        start_time += data.shape[0] / sample_rate

        # ['File name', 'Exp name', 'Process', 'Recording', 'Shank', 'Home directory',
        # 'Ordering key', 'Recording day', 'Recording time']
    return 0


def convertfromprofile(profile_name=None, profile_dict=None, log_file=None, log_func=None,
                       force_log=False, verbose=True, recursive=False, create_csv=True,
                       replace_folder_name=None, prefix='', funcpy=None):
    """Convert data from OpenEphys file to dat using `profile`


    :param profile_name: str, name of a profile from OpenEphys.oe_clustering.profiles
    :param profile_dict: dict, see above
    :param log_file: str, path to a file to log what I did.
    :param force_log: bool, erase log file if already exists
    :param verbose: bool, print log info
    :return:
    """

    if profile_name is not None:
        profi = profiles.profile[profile_name]
    else:
        profi = {}

    if profile_dict is not None:
        profi.update(profile_dict)

    if not len(profi):
        raise IOError('Must give a profile_name or a profile_dict')

    if replace_folder_name is not None:
        replace_folder_name = dict(replace_folder_name)
    else:
        replace_folder_name = {}

    if log_func is not None:
        log = log_func
        if log_file is not None:
            log('log_func is already provided will ignore log_file argument')
    elif log_file is None:
        def log(txt):
            if verbose: print(txt)
    else:
        if os.path.exists(log_file):
            if not force_log:
                raise IOError('Log file exists, use `force_log` to replace')
            F = open(log_file, 'w')
            F.close()

        def log(txt):
            with open(log_file, 'a') as F:
                F.write(txt + '\n')
            if verbose: print(txt)

    if create_csv:
        n = 'convert2dat_%s_output.csv' % time.strftime('%y%m%d_%H%M%S')
        if prefix:
            n = prefix + '_' + n
        csv_file = j_(profi['target_path'], n)
        if os.path.exists(csv_file):
            raise IOError('csv file %s already exists!' % csv_file)
        with open(csv_file, 'w') as F:
            F.write(','.join(['File name', 'Exp name', 'Process', 'Recording', 'Shank',
                              'Home directory', 'Ordering key', 'Recording day', 'Recording time',
                              'OE file', 'Recording length', 'Num channels', 'Channels']))
            F.write('\n')
    else:
        csv_file = None

    # find abspath and remove trailing slash
    source_path = os.path.abspath(profi['source_path'])
    source_path = source_path.rstrip(os.sep)
    target_path = os.path.abspath(profi['target_path'])
    target_path = target_path.rstrip(os.sep)

    log('Source path is: %s' % source_path)
    log('Target path is: %s' % target_path)
    log('')

    if not recursive:
        # do only local folder
        folder_fp = source_path
        target_folder = target_path
        if replace_folder_name:
            parts = target_folder.split(os.sep)
            for i in range(len(parts)):
                if parts[i] in replace_folder_name:
                    parts[i] = replace_folder_name[parts[i]]
            target_folder = os.sep.join(parts)

        if not is_oe_folder(folder_fp):
            log('%s does not contain OpenEphys data' % folder_fp)
            log('Use the --recursive flag if you want to process the subdirectories')
            return

        # remove trailing slash
        folder_fp = folder_fp.rstrip(os.sep)
        params = dict([(k, profi[k]) for k in ['num_chans', 'num_shanks', 'ref']])
        params['target_folder'] = target_folder
        params['prefix'] = os.path.split(folder_fp.rstrip(os.sep))[-1]
        if prefix:
            params['prefix'] = prefix + '_' + params['prefix']
        params['verbose'] = verbose
        params['funcpy'] = funcpy
        # if multiple processes provided, iter on them
        if hasattr(profi['process'], '__iter__'):
            for p in profi['process']:
                params['process'] = p
                convert_oefolder_to_dat(folder_fp, **params)
        else:
            params['process'] = profi['process']
            convert_oefolder_to_dat(folder_fp, **params)
        log('Done!')
        return

    # it's recursive
    base_level = len(source_path.split(os.sep))
    for folder, dirnames, filenames in os.walk(source_path):
        level = len(folder.split(os.sep)) - base_level
        if level:
            pref = '    ' * (level - 1) + '|___'
        else:
            pref = ''
        log(pref + 'Entering %s' % os.path.split(folder)[-1])

        if not is_oe_folder(folder):
            continue

        # folder contains OpenEphys data
        # Prefix all the files with the direcotry name
        subtree, exp = os.path.split(folder.rstrip(os.sep))
        log('    ' * level + 'Converting %s' % exp)

        # find the target dir and create the directory tree if needed
        subtree = subtree[len(source_path):].strip(os.sep)
        target_folder = j_(target_path, subtree)
        if replace_folder_name:
            parts = target_folder.split(os.sep)
            for i in range(len(parts)):
                if parts[i] in replace_folder_name:
                    parts[i] = replace_folder_name[parts[i]]
            target_folder = os.sep.join(parts)

        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        params = dict([(k, profi[k]) for k in ['num_chans', 'num_shanks', 'ref']])
        params['target_folder'] = target_folder
        params['prefix'] = exp
        params['funcpy'] = funcpy
        if prefix:
            params['prefix'] = prefix + '_' + params['prefix']
        params['verbose'] = verbose
        # if multiple processes provided, iter on them
        if hasattr(profi['process'], '__iter__'):
            for p in profi['process']:
                params['process'] = p
                convert_oefolder_to_dat(folder, log_func=log_func, level=level + 1, csv_file=csv_file,
                                        **params)
        else:
            params['process'] = profi['process']
            convert_oefolder_to_dat(folder, log_func=log_func, level=level + 1, csv_file=csv_file,
                                    **params)
    log('Done!')


def is_oe_folder(path):
    """Tell if a folder contains OE kwik data (i.e. it has a kwe file)

    :param path: path to test
    :return: bool
    """
    if not os.path.isdir(path):
        return False
    for fn in os.listdir(path):
        if fn.endswith('.kwe'):
            return True
    return False
