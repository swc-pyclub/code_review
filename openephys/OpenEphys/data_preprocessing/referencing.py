__author__ = 'blota'

"""
Should hold helper functions to do post-hoc referencing. The idea for now is to try common average reference
"""
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\ablot_open_ephys_kilosort_branch')

import numpy as np
import gc
from functools import partial
import multiprocessing
import time
from ..utils import make_list



def create_ref(data_reader, channels, chunk=None, verbose=True,
               out_dtype=None, method=np.median, n_cpu=None):
    """Create a single data line by combining `channels` of `data_reader` using `method`

     `data_reader` is loaded by chunks of `chunk` samples, upcasted to float64, transformed by `method` and
     downcasted to `out_dtype`. If `out_dtype` is None, then same dtype as input is used. `method` can be any
     function that accepts 1D arrays and returns a scalar (np.median and np.mean for instance)

    :param data_reader: anything that points to the data (oe_reader.OpenEphysReader.kwd_#.rec_# for instance)
    :param channels: list of channels
    :param chunk: int, number of point to load at once (max memory usage will be n_cpu x chunk x len(channels))
    :param verbose: bool
    :param out_dtype: None or np.dtype
    :param method: function to call to reference. Default to np.median
    :param n_cpu: number of cpu for multiprocessing, default to all CPU but one
    :return: np.array
    """

    if chunk is None:
        chunk = 60000
    if n_cpu is None:
        n_cpu = multiprocessing.cpu_count() - 1

    channels = make_list(sorted(channels), int) # needs to be sorted to read from hdf5
    npts = data_reader.shape[0]
    calculation_dtype = np.dtype(np.int16)  # dtype used to average

    if out_dtype is None:
        out_dtype = data_reader[0, 0].dtype

    chunk_size = chunk * len(channels) * calculation_dtype.itemsize
    if verbose:
        print('%i chunks of %.2f Mb to load (with %d worker(s), so %.2f Mb)' % (np.ceil(float(npts) / chunk),
                                                                                chunk_size / 1024. / 1024.,
                                                                                n_cpu,
                                                                                chunk_size / 1024. / 1024. * n_cpu))
        print('Output will be %.2f Mb' % (float(npts) * out_dtype.itemsize / 1024. / 1024))

    # widgets = ['Creating ref: ', pgb.Percentage(), ' ', pgb.Bar(marker=pgb.RotatingMarker()),
    #            ' ', pgb.ETA()]

    # create chunck borders
    chunck_borders = []
    n_read = 0
    # pbar = pgb.ProgressBar(widgets=widgets, maxval=npts + 1).start()
    while n_read < npts:
        end = min(npts, n_read + chunk)
        chunck_borders.append([n_read, end])
        n_read = end

    map_func = partial(reference_func, data_reader=data_reader, channels=channels, calculation_dtype=calculation_dtype,
                       method=method, out_dtype=out_dtype)

    if n_cpu > 1:
        print('Starting multiprocess pool with %d workers' % n_cpu)
        pool = multiprocessing.Pool(processes=n_cpu)
        result = pool.map_async(map_func, chunck_borders)
        tic = time.time()
        while not result.ready():
            if verbose:
                elaps = '%ds' % (time.time() - tic)
                print("\rRunning... (elapsed time %s)" % elaps, end="")
            time.sleep(0.5)
        all_out = result.get()
    else:
        all_out = map(map_func, chunck_borders)
    output = np.zeros(npts, dtype=out_dtype)
    for i, ((b, e), o) in enumerate(zip(chunck_borders, all_out)):
        output[b:e] = o
    if verbose:
        print('Done')
    # pbar.update(n_read + 1)
    #
    # pbar.finish()
    return output


def reference_func(border, data_reader, channels, calculation_dtype, method, out_dtype):
    data_chunk = np.asarray(data_reader[border[0]:border[1], channels], dtype=calculation_dtype)
    out = np.asarray(np.apply_along_axis(method, 1, data_chunk), dtype=out_dtype)
    del data_chunk
    gc.collect()
    return out