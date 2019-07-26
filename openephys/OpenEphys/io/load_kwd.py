__author__ = 'blota'

"""
Load functions for kwd files


"""

import pandas as pd
import h5py

from ..utils import cmp_versions

def file_info(oe_kwd):
    """Read file infos

    Return a dict with all the attributes of the file: num recordings, gain ,.,

    :param oe_kwd:str
    :return:dict
    """

    with h5py.File(oe_kwd, 'r') as h5file:
        rec_info = {}
        appl_info = {}
        nrec = 0
        for rec in h5file['recordings']:
            nrec += 1
            rec_info[rec] = dict(h5file['recordings'][rec].attrs.items())
            data_shape = h5file['recordings'][rec]['data'].shape
            rec_info[rec]['num_chans'] = data_shape[1]
            rec_info[rec]['num_samples'] = data_shape[0]
            appl_data = h5file['recordings'][rec]['application_data'].attrs
            if cmp_versions(h5py.__version__, '2.4') == -1:
                print(
                    'h5py version should be at least 2.4 (got %s) to read all the recording attributes' % h5py.__version__)
                print('Will skip some info')
                rec_info[rec]['is_multiSampleRate_data'] = appl_data['is_multiSampleRate_data']
            else:
                rec_info[rec].update(appl_data.items())
                rec_info[rec].update(h5file['recordings'][rec]['data'].attrs.items())
                rec_info[rec]['recordings'] = rec
        out = pd.DataFrame([pd.Series(v, name=k) for k, v in rec_info.items()])
    return out


def load_data(oe_kwd, chan, part, recording=None):
    """Quick wrapper to open the file, load some signal and close it

    :param oe_kwd: str
    :param recording: str
    :return: np.array
    """
    with h5py.File(oe_kwd, 'r') as h5file:
        if recording is None:
            recs = h5file['recordings'].keys()[0]
            assert len(recs) == 1
            rec = list(recs)[0]
        data = h5file['recordings'][rec]['data']
        if part is None:
            part = list(range(data.shape[0]))
        if chan is None:
            chan = list(range(data.shape[1]))
        data = h5file['recordings'][rec]['data'][part, chan]
    return data

def load_application_data(oe_kwd, recording, what=None):
    """Load whatever is in application data

    Default application data includes channel_bit_volts,
    channel_sample_rates and timestamps. You can load only part by
    specifying `what`"""

    recording = str(recording)

    out = dict()
    with h5py.File(oe_kwd, 'r') as h5file:
        app_data = h5file['recordings'][recording]['application_data']
        if what is None:
            what = app_data.keys()
        for w in what:
            out[w] = app_data[w][:]
    return out
