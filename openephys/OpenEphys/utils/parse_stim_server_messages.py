import pandas as pd
import numpy as np
import re

def parse_stim_server_messages(message_dataframe):
    """Parse messages produced by stim server

    The input should be a table of messages as read by read_kwe.
    The output is a table with:

    NOT FINISHED
    arguments are not parsed properly
    for now command_id should be use to group lines together but it's not fully
    tested
    """

    # make a copy to avoid touching original table
    stim_server = message_dataframe.copy()
    # get the SST and the New command messages
    stim_server['is_sst'] = stim_server.text.map(lambda x: x.startswith('SST'))
    stim_server['is_newcmd'] = stim_server.text.map(lambda x:
                                                    x.startswith('New command'))

    # remove everything else (messages that are not stim server)
    not_sst = stim_server[(~stim_server.is_sst) & (~stim_server.is_newcmd)]
    stim_server = stim_server[(stim_server.is_sst) | (stim_server.is_newcmd)]

    # Create output series
    cmd_type = pd.Series(index=stim_server.index, name='cmd_type')
    presentation_id = pd.Series(index=stim_server.index,
                                name='presentation_id')
    seq_id = pd.Series(index=stim_server.index,
                                name='seq_id')
    blank_time = pd.Series(index=stim_server.index,
                                name='blank_time')
    n_presentation_arguments = pd.Series(index=stim_server.index,
                                name='n_presentation_arguments')
    n_stimulation_arguments = pd.Series(index=stim_server.index,
                                name='n_stimulation_arguments')

    ## Parse sst
    ind_sst = stim_server[stim_server.is_sst].index
    parse_func = lambda x: x.split(' ')[1].strip()
    cmd_type.ix[ind_sst] = stim_server.ix[ind_sst].text.map(parse_func)
    # Case present
    present_ind = cmd_type[cmd_type=='PRESENT'].index
    parse_func = lambda x: int(x.split(' ')[2].strip())
    val = stim_server.ix[present_ind].text.map(parse_func)
    presentation_id.ix[present_ind] = val
    # Case seq
    seq_ind = cmd_type[cmd_type=='SEQ'].index
    parse_func = lambda x: int(x.split(' ')[2].strip())
    val = stim_server.ix[seq_ind].text.map(parse_func)
    seq_id.ix[seq_ind] = val

    # Parse new command
    ind_newcmd = stim_server[stim_server.is_newcmd].index
    parse_func = lambda x: x.strip().split(' ')
    temp = stim_server.ix[ind_newcmd].text.map(parse_func)
    # check that I have what I expect
    assert all(temp.map(lambda x: x[0]=='New'))
    assert all(temp.map(lambda x: x[1]=='command:SS'))
    cmd_type.ix[ind_newcmd] = temp.map(lambda x: x[2])
    presentation_id.ix[ind_newcmd] = temp.map(lambda x: int(x[3]))
    blank_time.ix[ind_newcmd] = temp.map(lambda x: float(x[4]))
    n_presentation_arguments.ix[ind_newcmd] = temp.map(lambda x: int(x[5]))
    n_stimulation_arguments.ix[ind_newcmd] = temp.map(lambda x: int(x[6]))

    # put new series in dataframe
    stim_server['cmd_type'] = cmd_type
    stim_server['presentation_id'] = presentation_id
    stim_server['seq_id'] = seq_id
    stim_server['blank_time'] = blank_time
    stim_server['n_presentation_arguments'] = n_presentation_arguments
    stim_server['n_stimulation_arguments'] = n_stimulation_arguments

    # group by command
    stim_server['command_id'] = np.asarray(stim_server.is_newcmd.values,
                                           dtype=int).cumsum()

    return stim_server
