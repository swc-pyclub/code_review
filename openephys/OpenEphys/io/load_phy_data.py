"""
Functions dealing with phy data. They work only in python 3 and use a phy session as input
"""

import pandas as pd
import numpy as np


def create_spk_df(session, shank=None, force=False, n_zero_pad_name=None):
    """Create a dataframe out of a clustered session

    :param session:
    :param which_spikes:
    :return:
    """
    msg = None
    clustered = bool(session.model.n_clusters)
    if not clustered:
        msg = 'Session not clustered!'
    else:
        non_clusterd = np.sum([v == 3 for v in session.model.cluster_groups.values()])
        if non_clusterd > 0:
            msg = 'Manual oe_clustering not done. Found %i "Unknown" clusters' % non_clusterd
    if msg is not None:
        if not force:
            raise IOError(msg)
        print(msg)

    # create df from spike info
    what = ['spike_ids', 'spike_clusters', 'spike_recordings', 'spike_samples', 'spike_times']
    dict_df = dict([(k, getattr(session.model, k)) for k in what])
    df = pd.DataFrame(dict_df)
    df = df.set_index('spike_ids', drop=False)
    if shank is not None:
        df['shank'] = shank
        if n_zero_pad_name is None:
            n_zero_pad_name = len(str(max(session.model.cluster_ids)))
        clabel = np.repeat(np.array(str(0).zfill(6 + len(shank) + n_zero_pad_name)), len(df))

    # add cluster group for every spike (do it by group to avoid loong loop)
    cgroup = np.array([3] * len(df))

    for cl_gp, cl_df in df.groupby('spike_clusters'):
        cgroup[cl_df.index] = session.model.cluster_groups[cl_gp]
        if shank is not None:
            clabel[cl_df.index] = 'shk%s_cl' % shank + str(cl_gp).zfill(n_zero_pad_name)
    df['cluster_group'] = cgroup
    if shank is not None:
        df['cluster_label'] = clabel
    return df


def create_cluster_df(spk_df, groupby='cluster_group'):
    '''Create a dataframe with a line per cluster

    groupby default to cluster_group: one line per cluster but can be change for instance to cluster label for
    multi shank experiements

    :param spk_df:
    :param groupby:
    :return:
    '''
    index = list(spk_df.columns)
    index += ['n_spks']
    series = []
    for gn, df_g in spk_df.groupby(groupby):
        d = [df_g[c].iloc[0] for c in index[:-1]]
        d.append(len(df_g))
        series.append(pd.Series(name=gn, index=index, data=d))
    return pd.DataFrame(series)
