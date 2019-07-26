"""Povide delete temp files function to recursively destroy .spikdetekd, .phy and .klustakwik2 folders"""

import os, shutil
import os.path as op


def delete_temp_files(directory, delete_without_asking=False, log_func=print,
                      confirm_func=None, todelete=('.spikedetekt', '.phy', '.klustakwik2')):
    """Destroys .spikdetekd, .phy and .klustakwik2 folders in a directory"""

    if confirm_func is None:
        def confirm_func(msg):
            answer = input(msg + ' (yes for yes, anything else is no)')
            return answer == 'yes'

    if delete_without_asking:
        def confirm_func(msg):
            return True

    abs_path = op.abspath(directory)
    for fname in os.listdir(abs_path):
        if not op.isdir(op.join(abs_path, fname)):
            continue
        elif fname in todelete:
            if not delete_without_asking:
                conf = confirm_func('Do you want to delete %s' % fname)
                if not conf:
                    return
            log_func('Deleting %s' % fname)
            shutil.rmtree(op.join(abs_path, fname))
