"""
Global class using the subfunction per file to access all the data from a folder
"""

import os
from .io.describe_oe_folder import oe_info
from .data_preprocessing import referencing
from .io import load_kwe
from .io import load_kwd
from .io import load_kwx
from .utils import make_list
import h5py
import numpy as np
import pandas as pd


j_ = os.path.join

__author__ = 'blota'


class OpenEphysReader(object):
    def __init__(self, path, load_info=True, verbose=True):
        """Path is the path to a folder containing OE files

        :param path:str
        :param verbose: bool
        :return:
        """
        self.folder = os.path.abspath(path)
        self.content = oe_info(self.folder)
        self.num_files = self.content.file_type.value_counts()
        if load_info:
            self.load_kwe_infos(verbose=verbose)
            self.load_kwds_info(verbose=verbose)

    def load_kwe_infos(self, verbose=True, ttl=True, messages=True):
        """Load kwe data from disk

        Replace current values if already loaded

        :param bool verbose: Print more output
        :param bool ttl: Load ttl info in RAM
        :param bool messages: Load messages in RAM
        :return:
        """

        assert self.num_files['.kwe'] == 1
        # don't know if more than one can be created, maybe with multiple spike detector
        fpath = j_(self.folder, self.content[self.content.file_type == '.kwe'].iloc[0].fname)
        if verbose:
            print('Will load %s' % fpath)
        events = load_kwe.get_events(fpath, messages=messages, ttl=ttl)
        if ttl:
            self._ttl = events['TTL']
        if messages:
            self._messages = events['Messages']

        rec_info = load_kwe.get_rec_info(fpath)
        self._rec_info = rec_info
        if verbose:
            print('done')

    def load_start_times(self, look_for_protocols=False):
        """

        :param look_for_protocols:bool
        :return:
        """
        out_dict = load_kwe.find_time_start(self.messages, look_for_protocols=look_for_protocols)
        self._proc_start = out_dict['proc_start']
        self._soft_start = out_dict['soft_start']
        if look_for_protocols:
            self._prot_start = out_dict['prot_df']

    def load_kwds_info(self, which_processor=None, verbose=True):
        """Load info about kwd files

        Load info from all kwd files of the folder if which_processor is None, else load only given processor

        :param which_processor: int
        :return:
        """
        df = self.content[self.content.file_type == '.kwd']
        if which_processor is None:
            which_processor = self.process_list
        which_processor = make_list(which_processor, int)  # in case it's and integer
        for p in which_processor:
            line = df[df.process == p]
            assert len(line) == 1
            fpath = j_(self.folder, line.iloc[0].fname)
            kwd = KwdFile(fpath, verbose=verbose)
            setattr(self, 'kwd_%i' % p, kwd)


    def load_spikes(self, verbose=True, waveforms=False):
        """load spike dataframe in memory

        :param verbose:bool
        :return:None
        """
        assert self.num_files['.kwx'] == 1
        # don't know if more than one can be created, maybe with multiple spike detector
        fpath = j_(self.folder, self.content[self.content.file_type == '.kwx'].iloc[0].fname)
        if verbose:
            print('Will load %s' % fpath)
        self.spk_df = load_kwx.load_spikes(fpath, waveforms=waveforms)
        recs = self.spk_df.recordings.value_counts()
        start_sample = dict()
        for r in recs.index:
            rec_index = np.where(self.process_start.recording == r)[0]
            assert len(rec_index) == 1
            start_sample[r] = self.process_start.iloc[rec_index].start_time

        samples_in_rec = pd.Series(np.zeros_like(self.spk_df.recordings.values), index=self.spk_df.index)
        for gn, df in self.spk_df.groupby('recordings'):
            samples_in_rec.ix[df.index] = np.array(df.time_samples, dtype=int) - int(start_sample[gn])
        self.spk_df['time_samples_in_rec'] = samples_in_rec
        if verbose:
            print('done')

    @property
    def ttl(self):
        """Get the TTL info from this folder

         TTL are loaded in memory the first time this function is called

        :return:pd.DataFrame
        """
        if not hasattr(self, '_ttl'):
            self.load_kwe_infos(verbose=False, messages=False)
        return self._ttl

    @property
    def rec_info(self):
        """Get the rec_info info from this folder

         rec_info are loaded in memory the first time this function is called

        :return:pd.DataFrame
        """
        if not hasattr(self, '_rec_info'):
            self.load_kwe_infos(verbose=False)
        return self._rec_info

    @property
    def messages(self):
        """Get the Messages info from this folder

         Messages are loaded in memory the first time this function is called

        :return:pd.DataFrame
        """

        if not hasattr(self, '_messages'):
            self.load_kwe_infos(verbose=False, ttl=False)
        return self._messages

    @property
    def protocols_start(self):
        """Return the dataframe of protocol start messages (from kwe file)

        A protocol is added in this datafile if a pair of message, one with "start" one with "end", is found
        (messages must be exactly identical appart from this keyword)

        :return: pd.DataFrame
        """
        if not hasattr(self, '_prot_start'):
            self.load_start_times(look_for_protocols=True)
        return self._prot_start

    @property
    def sofware_start(self):
        """Return the dataframe of software start messages (from kwe file)

        :return: pd.DataFrame
        """
        if not hasattr(self, '_soft_start'):
            self.load_start_times()
        return self._soft_start

    @property
    def process_start(self):
        """Return the dataframe of processor start messages (from kwe file)

        :return: pd.DataFrame
        """
        if not hasattr(self, '_proc_start'):
            self.load_start_times()
        return self._proc_start

    @property
    def process_list(self):
        """Return the list of processor from kwd files

        :return: list
        """
        return list(self.content[self.content.file_type == '.kwd'].process)


class KwdFile(object):
    """Container for all the info related to a single .kwd file

    Should have a smart getter that opens and closes the kwd file and read from disc
    """

    def __init__(self, fpath, verbose=True, parent=None):
        """

        :param fpath:str
        :param verbose:bool
        :param parent:OpenEphysReader instance
        :return:KwdFile instance
        """
        self.fpath = fpath
        if parent is not None and verbose and not isinstance(parent, OpenEphysReader):
            print("Warning, parent is not an OpenEphysReader instance")
        self.parent = parent
        self.load_kwd_info(verbose=verbose)
        for rec in self.file_info.index:
            setattr(self, 'rec_%s' % rec, KwdRecording(self.fpath, rec))

    @property
    def file_info(self):
        """Read file infos

        Return a dict with all the attributes of the file: num recordings, gain ,.,

        :return:dict
        """
        if not hasattr(self, '_finfo_dict'):
            self.load_kwd_info()
        return self._finfo_dict

    def load_kwd_info(self, verbose=True):
        """Load kwd info from disk

        Replace current values if already loaded

        :param verbose: bool
        :return:
        """

        if verbose:
            print('Will load %s' % self.fpath)
        self._finfo_dict = load_kwd.file_info(self.fpath)
        if verbose:
            print('done')


class KwdRecording(object):
    """Wrapper to create a smartish getter per recordings

    TODO: Add the option to convert to true units before returning value
          Maybe add also here digital referencing options
    """
    reference_methods = dict(CMR=np.median,
                             CAR=np.mean)

    def __init__(self, fpath, rec):
        self.fpath = fpath
        self.rec = rec
        self._reference = None
        self._ref_method = 'CMR'
        self._ref_chans = None
        self.referenced_data = ReferencedData(self)

    @property
    def ref_chans(self):
        """Channel used when computing the reference
        """
        return self._ref_chans

    @ref_chans.setter
    def ref_chans(self, value):
        value = sorted(make_list(value, int))
        assert all([0 <= v < self.shape[1] for v in value])
        self._ref_chans = value
        self._reference = None # erase cached reference when changing chans

    @property
    def ref_method(self):
        """Method used to compute the reference. Currently support CMR and CAR
        (Common Median Reference and Common Average Reference)
        """
        return self._ref_method

    @ref_method.setter
    def ref_method(self, value):
        if value is not None:
            value = str(value).upper()
            assert value in self.reference_methods.keys()
        self._ref_method = value
        self._reference = None # erase cached reference when changing method

    @property
    def reference(self):
        """Return reference trace (create it if needed
        """
        if self._ref_method is None:
            return None
        if self._reference is None:  # ref not created yet
            self.create_reference()
        return self._reference

    def create_reference(self, n_cpu=None, chunk=None):
        if self.ref_chans is None:
            raise IOError('Channels to use to reference must be specified')
        self._reference = referencing.create_ref(self, self.ref_chans, chunk=chunk, n_cpu=n_cpu,
                                                 verbose=True, out_dtype=None,
                                                 method=self.reference_methods[self._ref_method])

    def save_ref(self):
        """Write reference to disk

        :return:
        """

        if self.reference is None:
            print('No reference, nothing to save')
            return
        savename = self.fpath + '_ref_%s_%s.npy' % (self.ref_method, '_'.join([str(i) for i in sorted(self.ref_chans)]))
        np.save(savename, self.reference)

    def load_ref(self):
        savedname = self.fpath + '_ref_%s_%s.npy' % (self.ref_method, '_'.join([str(i) for i in sorted(self.ref_chans)]))
        if not os.path.isfile(savedname):
            raise IOError('Reference not found (%s)' % savedname)
        self._reference = np.load(savedname)

    def load_timestamps(self):
        """For newer OpenEphys GUI only. Load the timestamps"""
        with h5py.File(self.fpath, 'r') as h5file:
            app_data = h5file['recordings'][self.rec]['application_data']
            if not app_data.has_key('timestamps'):
                print('No timestamps. Is it an old file?')
                return
            ts = app_data.timestamps[:]
        return ts

    @property
    def shape(self):
        with h5py.File(self.fpath, 'r') as h5file:
            data = h5file['recordings'][self.rec]['data']
            return data.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        with h5py.File(self.fpath, 'r') as h5file:
            data = h5file['recordings'][self.rec]['data']
            return data.size

    def __getitem__(self, item):
        with h5py.File(self.fpath, 'r') as h5file:
            data = h5file['recordings'][self.rec]['data']
            return data.__getitem__(item)

    def __len__(self):
        with h5py.File(self.fpath, 'r') as h5file:
            data = h5file['recordings'][self.rec]['data']
            return len(data)


class ReferencedData(object):
    """Subclass to create getter. Used in KwdRecording

    """

    def __init__(self, parent):
        """Reference subtracted data

        :param parent:
        :return:
        """
        assert isinstance(parent, KwdRecording)
        self.parent = parent

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) == 1:
            part_item = item
            chan_item = None
        elif len(item) == 2:
            part_item = item[0]
            chan_item = item[1]
        else:
            raise IOError("Item should have 1 or 2 dimension")

        data = self.parent.__getitem__(item)
        ref = self.parent.reference
        return (data.T - ref[part_item]).T

    @property
    def size(self):
        return self.parent.size

    @property
    def shape(self):
        return self.parent.shape

    def __len__(self):
        return len(self.parent)

    @property
    def ndim(self):
        self.parent.ndim
