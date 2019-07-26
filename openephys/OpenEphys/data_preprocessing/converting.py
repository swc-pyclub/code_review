__author__ = 'blota'

from ..utils import make_list
import numpy as np


def get_in_volt(oe_recording, process_number, recording_number, chans=None, part=None):
    """Read data from oe_recording, in volt

    Warning: will load all the asked data in memory and convert to float64 ... calling with chans=None
             and part=None might not always be good

    :param oe_recording: OpenEphysReader
    :param process_number: int
    :param recording_number: int
    :param part: slice or int (anything that __getitem__ of h5py would understant)
    :return:
    """

    try:
        kwd = getattr(oe_recording, 'kwd_%i' % process_number)
        rec = getattr(kwd, 'rec_%i' % recording_number)
    except AttributeError:
        raise IOError('process or recording invalid')

    if chans is None:
        chans = list(range(rec.shape[1]))
    chans = make_list(chans, int)

    # get data from file
    if part is None:
        data = rec[:, chans]
    else:
        data = rec[part, chans]

    # convert to Volts
    gain = kwd.file_info.channel_bit_volts.ix[int(recording_number)][chans]
    data = np.asarray(data, dtype=float) * gain

    if len(chans) == 1:
        data = data[:, 0]  # remove useless dimension
    return data
