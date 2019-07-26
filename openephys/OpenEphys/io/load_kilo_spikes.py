import os
import pandas as pd
import numpy as np
from ..oe_clustering import read_python_file
from ..utils.unit_quality import get_wave_spike_info

def load_kilo_sorted_spikes(folder_path):
    """Fonction to load output of kilo sort

    Kilo sort produces plenty of small files. This function load
    everything that is related to spike time and
    clustering. Waveforms are not included and not saved to disk but
    should be read directly from the dat file.

    Inputs:
    - 'folder_path': Path to a folder containing the output of kilo
    sort

    Outputs:
    - 'spk_df': a dataframe containing for each spike its cluster
    id (spike_clusters), its template id (spike_templates, different
    from cluster id in case of manual merge or split), its time
    sample (spike_times), and its amplitude.

    - 'cluster_df': a dataframe with cluster_id and cluster group

    - 'strParams': a structure with data read from params.py

    15/07/2016 - Antonin Blot"""

    if not os.path.isdir(folder_path):
        raise IOError('Folder %s not found'%folder_path)
    if not os.path.isfile(os.path.join(folder_path, 'spike_clusters.npy')):
        raise IOError('%s do not contain kilosorted data')

    npy_files = ['spike_clusters', 'spike_templates', 'spike_times',
                 'amplitudes']
    spk_df = pd.DataFrame()
    for what in npy_files:
        data = np.load(os.path.join(folder_path, what + '.npy'))
        data = data.squeeze()
        # should be 1d but can be a column
        if data.ndim != 1:
            raise IOError('I could not read %s'%what)
        spk_df[what] = data

    params = read_python_file(os.path.join(folder_path, 'params.py'))

    spk_df['spike_samples'] = spk_df.spike_times
    spk_df['spike_times'] = np.array(spk_df.spike_samples,
                                     dtype = float)/params['sample_rate']

    # Cluster group is created by phy, so it might not be here
    csv_cluster = os.path.join(folder_path, 'cluster_groups.csv')
    if not os.path.isfile(csv_cluster):
        print('%s is not clustered'%folder_path)
        cluster_id = spk_df.spike_clusters.unique()
        cluster_df = pd.DataFrame(dict(cluster_id = cluster_id,
                                       group = ['unclustered']*len(cluster_id)))
    else:
        cluster_df = pd.read_csv(csv_cluster, delimiter = '\t')
    # set the index to cluster id to facilitate indexing
    cluster_df = cluster_df.set_index('cluster_id', drop = False)
    return spk_df, cluster_df, params


def create_cluster_df(spk_df, groupby='spike_clusters'):
    '''Create a dataframe with a line per cluster

    groupby default to cluster_group: one line per cluster but can be change for
    instance to cluster label for multi shank experiements

    :param spk_df:
    :param groupby:
    :return:
    '''
    import warnings
    warnings.warn('replace by second output of load_kilo_sorted_spikes',
                  DeprecationWarning)
    index = list(spk_df.columns)
    index += ['n_spks']
    series = []
    for gn, df_g in spk_df.groupby(groupby):
        d = [df_g[c].iloc[0] for c in index[:-1]]
        d.append(len(df_g))
        series.append(pd.Series(name=gn, index=index, data=d))
    return pd.DataFrame(series)


def load_templates(folder_path, spk_df=None, which='main'):
    """Load templates per cluster

    if spk_df is None simply load the template file and return

    if spk_df is not None, then use spike_clusters and spike_templates columns
    to return templates per cluster depending on `which`

    `which` can be 'main', 'average'. If a cluster has multiple templates,
    'main' will return only the main (that accounting for most spikes)
    'average' will return a weighted average
    """
    if not os.path.isdir(folder_path):
        raise IOError('Folder %s not found'%folder_path)
    if not os.path.isfile(os.path.join(folder_path, 'templates.npy')):
        raise IOError('%s do not contain kilosorted data')
    assert which in ('main', 'average')
    npy_files = ['templates', 'templates_ind']
    data = dict()
    for what in npy_files:
        data[what] = np.load(os.path.join(folder_path, what + '.npy'))

    # Template ind are the indices of the channels used by the template
    # since now I use them all, assert that and ignore
    d = data.pop('templates_ind')
    assert(not (d-d[0]).any())

    templates = data['templates']
    del data
    if spk_df is None:
        return templates

    # List of files that are there but I don't use
    # TODO I don't know what are all the other files and I don't want
    # to look now.
    other_files = ['template_features', 'template_feature_ind', 'templates_unw',
                   'whitening_mat_inv', 'whitening_mat']

    # initialise output with same individual template shape but different
    # number of line
    cluster_list = spk_df.spike_clusters.unique()
    shape =  list(templates.shape)
    shape[0] = len(cluster_list)
    cluster_template = np.zeros(shape)

    # iterate on cluster
    for i, cl in enumerate(cluster_list):
        cl_df = spk_df[spk_df.spike_clusters == cl]
        templ = cl_df.spike_templates.value_counts()
        if which == 'main':
            templ_cl = templates[templ.argmax()]
        else: #then it's average
            templ /= templ.sum()
            templ_cl = np.sum(templates[templ.index].T * templ.values, 2).T
        cluster_template[i] = templ_cl

    return cluster_template, cluster_list

def make_unit_dataframe(spk_df, cluster_df, templates_array, templates_index,
                        sampling=30000, gain=1000):
    """Get a dataframe with unit properties:

        channel
        probe position
        mean waveform on channels
        mean ISI
        mean CV
        mean FR
        first spike
        last spike
        n_spikes
        percentage of ISI below 1ms

    spk_df and cluster_df are the outputs of load_kilo_sorted_spikes
    template_array is a 2D array with one line per template
    template_index is the cluster index corresponding to each line of
    template_array
    """
   #% make sure templates_index is a list to have index method
    templates_index = list(templates_index)
    series = []
    for gn, df in spk_df.groupby('spike_clusters'):
        line = dict()
        # Add cell identifier
        line['spike_clusters'] = gn
        line['cluster_group'] = cluster_df.group[cluster_df.cluster_id == gn].iloc[0]
        av_template = templates_array[templates_index.index(gn) ,:]
        line['average_template'] = av_template

        # Calculate ISIs and % spikes in refractory period
        spks = df.spike_times
        isi = np.diff(spks)
        line['percent_isi_below'] = np.sum(isi < 1e-3)/len(isi)

        # Add beg and end of cell recording
        line['first_spike'] = np.min(spks)
        line['last_spike'] = np.max(spks)

        # Add also basic firing properties
        line['n_spikes'] = len(spks)
        line['mean_ISI'] = isi.mean()
        line['mean_fr'] = len(spks)/np.ptp(spks)
        # cv is also sometimes used
        line['cv_isi'] = isi.mean()/isi.std()

        # get a estimate of the cell position by finding the channel with
        # the biggest signal amplitude
        line['main_channel'] = av_template.ptp(0).argmax()
        d = get_wave_spike_info(av_template[:,line['main_channel']],
                                sampling, gain).iloc[0]
        # 1000 in the previous line assumes date from templates is in V
        # 30000 is the sampling rate
        for k, v in d.iteritems():
            line[k] = v
        series.append(pd.Series(line, name = gn))
    return pd.DataFrame(series)
