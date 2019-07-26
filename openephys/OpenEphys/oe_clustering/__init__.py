import os, shutil
from . import convert2dat, delete_temp_files, profiles


def read_python_file(path):
    """Read a profile file.

    Profile file must be a python-like file

    :param path2file:
    :return:
    """

    path = os.path.realpath(os.path.expanduser(path))
    assert os.path.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    try:
        exec(contents, {}, metadata)
    except Exception:
        raise IOError('Error in python file %s'%path)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def moveinsubfolder(root_folder, ext, do_only=None, log_func=print,
                    recursive=False, skip_lonely=False):
    """Move every files with `ext` extension in separated folder

    If do_only, is not None make a folder only for files with  do_only in their name
    If skip_lonely,  Do not move files that are already alone (with that ext) in their folder
    """
    abort = False
    if skip_lonely:
        # first look for lonely_files
        correct_ext = [os.path.splitext(fname)[1] == ext for fname in os.listdir(root_folder)]
        if len(correct_ext) < 2:
            abort = True

    for fname in os.listdir(root_folder):
        full_local_path = os.path.join(root_folder, fname)
        if os.path.isdir(full_local_path):
            if recursive:
                moveinsubfolder(full_local_path, ext, do_only, log_func, recursive)
            else:
                continue
        if abort:
            continue
        fn, fext = os.path.splitext(fname)
        if fext != ext:
            continue
        if (do_only is not None) and not (do_only in fn):
            continue
        log_func('Moving %s' % fname)
        dir_path = os.path.join(root_folder, fn)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        elif not os.path.isdir(dir_path):
            raise IOError('%s already exists and is not a directory' % dir_path)
        target = os.path.join(dir_path, fname)
        if os.path.exists(target):
            log_func('%s already exists. Skipping' % target)
            continue
        shutil.move(os.path.join(root_folder, fname), target)
