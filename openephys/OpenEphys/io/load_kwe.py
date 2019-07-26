"""
Load data from a kwe file

kwe files contain the event and the message, so it's small enough for me and I'd rather load everything in a dataframe.
We might need to reorganise if loading everything is a problem
"""

import re
import h5py
import pandas as pd
import numpy as np


def get_events(oe_kwe, messages=True, ttl=True):
    what_to_read = []
    if messages:
        what_to_read.append('Messages')
    if ttl:
        what_to_read.append('TTL')
    with h5py.File(oe_kwe, 'r') as F:
        out = {}
        for what in what_to_read:
            ev = F['event_types'][what]['events']
            ev_rec = ev['recording']
            events = pd.DataFrame({'recording': ev_rec})
            events['time_samples'] = ev['time_samples']
            events['eventID'] = np.array(ev['user_data'].get('eventID'))
            events['nodeID'] = np.array(ev['user_data'].get('nodeID'))
            if what == 'TTL':
                events['event_channels'] = np.array(ev['user_data']['event_channels'], dtype=int)
            else:
                events['text'] = [str(t, encoding='utf-8') for t in ev['user_data']['Text']]

            out[what] = events
    return out


def get_rec_info(oe_kwe):
    """Load info about recordings from kwe file.

    Contains bit detph, name, sample rate and start time for each recording

    :param oe_kwe:
    :return:pd.DataFrame
    """
    out_df = pd.DataFrame()
    with h5py.File(oe_kwe, 'r') as F:
        for rec in F['recordings'].keys():
            s = pd.Series(name=rec, index=list(F['recordings'][rec].attrs.keys()),
                          data=list(F['recordings'][rec].attrs.values()))
            s['recordings'] = rec
            out_df = out_df.append(s)
    out_df.set_index('recordings', drop=False)
    return out_df


def find_time_start(messages, recording=None, look_for_protocols=False):
    """Find the message with the processors start times and the software start time
    parse them and return a dataframe for each

    If look_for_protocols is True, will also look for user sent messages that come in pairs with a start and an end
    """
    proc_st_ind = []
    soft_st_ind = []
    other_msg = []
    for ind, row in messages.iterrows():

        if recording is not None and row.recording != recording:
            continue
        if row.text.startswith("Processor: "):
            proc_st_ind.append(ind)
        elif row.text.startswith("Software time:"):
            soft_st_ind.append(ind)
        elif look_for_protocols:
            other_msg.append(ind)

    if len(soft_st_ind) == 0:
        m = 'Recording %s not found.' % recording
        m += ' Valid values are: %s' % ', '.join([str(i) for i in messages.recording.value_counts().keys()])
        raise ValueError(m)
    soft_start = messages.ix[soft_st_ind]
    soft_start['start_time'] = np.nan
    soft_start['sampl_freq'] = np.nan
    soft_start['units'] = ''
    for ind, row in soft_start.iterrows():
        beg, end = row.text.split('@')
        freq = re.findall('\d+', end)
        assert len(freq) == 1
        row.sampl_freq = float(freq[0])
        row.units = re.findall('[a-zA-Z]+', end)[0]
        st_time = re.findall('\d+', beg)
        assert len(st_time) == 1
        row.start_time = float(st_time[0])

        soft_start.ix[ind] = row

    proc_start = messages.ix[proc_st_ind]
    proc_start['start_time'] = np.nan
    proc_start['sampl_freq'] = np.nan
    proc_start['processor'] = np.nan
    proc_start['units'] = ''
    for ind, row in proc_start.iterrows():
        beg, end = row.text.split('@')
        freq = re.findall('\d+', end)
        assert len(freq) == 1
        row.sampl_freq = float(freq[0])
        row.units = re.findall('[a-zA-Z]+', end)[0]
        st_time = re.findall('\d+', beg)
        if len(st_time) == 2:
            # older version of the GUI have the process and the start time
            row.start_time = float(st_time[1])
            row.processor = int(st_time[0])
        elif len(st_time) == 3:
            # newer version have also a subprocess that I ignore for now
            row.start_time = float(st_time[2])
            row.processor = int(st_time[0])
        else:
            raise ValueError('Unexpected message associated to process start')

        proc_start.ix[ind] = row
    out_dict = dict(soft_start=soft_start, proc_start=proc_start)

    if look_for_protocols:
        df = messages.ix[other_msg]
        start_msg = {}
        columns = [n + '_start' for n in df.columns] + [n + '_end' for n in df.columns]
        prot_df = pd.DataFrame(columns=columns)
        n_pairs = 0
        for ind, row in messages.iterrows():
            if 'start' in row.text:
                start_msg[row.text.replace('start', '').strip()] = ind
            elif 'end' in row.text:
                end_txt = row.text.replace('end', '').strip()
                try:
                    ind_start = start_msg[end_txt]
                except KeyError:  # no start corresponding to end
                    continue
                # create and empty row
                prot_df = prot_df.append(pd.Series([np.NaN for _ in range(len(columns))],
                                                   index=columns, name=n_pairs))
                for c in row.index:
                    prot_df[c + '_end'].ix[n_pairs] = row[c]
                    prot_df[c + '_start'].ix[n_pairs] = df.ix[ind_start][c]
                n_pairs += 1
        out_dict['prot_df'] = prot_df

    return out_dict


if __name__ == '__main__':
    # should move to test at some point
    path = '../tests/test_data_set/experiment1.kwe'
    ev = get_events(path)
    print(ev.keys())
    messages = ev['Messages']
    soft, proc = find_time_start(messages, 1)
    print(proc)
