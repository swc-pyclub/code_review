__author__ = 'blota'

"""
Load functions for kwx files


"""

import h5py
import pandas as pd

from ..utils import make_list


def load_spikes(oe_kwx, channel_groups=None, waveforms=False):
    """Load spikes from a kwx file

    Create a dataframe with spikes from `channel_groups`.
    It contains the recording number and the time sample plus the waveform if `waveforms` is True

    :param oe_kwx:str
    :param channel_groups: list of int
    :param waveforms: bool
    :return:
    """

    with h5py.File(oe_kwx, 'r') as h5file:
        cg = h5file['channel_groups'].keys()
        if channel_groups is None:
            channel_groups = cg
        else:
            channel_groups = make_list(channel_groups, str)  # force str in case int is given
            assert all([i in cg for i in channel_groups])

        columns = ['recordings', 'time_samples']
        if waveforms:
            columns.append('waveforms_filtered')
        out_df = pd.DataFrame(columns=columns)

        for c in channel_groups:
            node = h5file['channel_groups'][c]
            kwargs = dict(recordings=node['recordings'],
                          time_samples=node['time_samples'])
            if waveforms:
                kwargs['waveforms_filtered'] = list(node['waveforms_filtered'])
            df = pd.DataFrame(kwargs)
            df['channel_groups'] = int(c)  # force int because string of number are a pain
            out_df = pd.concat([out_df, df], ignore_index=True)
    out_df.sort_values(by='time_samples', inplace=True)  # sort by spike time
    out_df.index = range(len(out_df))  # put linear index
    return out_df
