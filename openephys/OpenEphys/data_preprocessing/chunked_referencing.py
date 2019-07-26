
import numpy as np
import gc
from ..utils import make_list
from .referencing import create_ref


def create_chunked_ref(data_reader, channels, data_chunks=25000000, multi_process_chunks=7000000, verbose=True,
               out_dtype=None, method=np.median):

    channels = make_list(channels, int)
    npts = data_reader.shape[0]
    calculation_dtype = np.dtype(np.int16)  # dtype used to average

    if out_dtype is None:
        out_dtype = data_reader[0, 0].dtype

    chunk_size = data_chunks * len(channels) * calculation_dtype.itemsize
    if verbose:
        print('%i chunks of %.2f Mb to load' % (np.ceil(float(npts) / data_chunks), chunk_size / 1024. / 1024.))
        print('Output will be %.2f Mb' % (float(npts) * out_dtype.itemsize / 1024. / 1024))

    output = np.zeros(npts, dtype=out_dtype)
    n_read = 0
    while n_read < npts:
        end = min(npts, n_read + data_chunks)
        data_chunk = np.asarray(data_reader[n_read:end, channels], dtype=calculation_dtype)
        output[n_read:end] = create_ref(data_chunk, channels, chunk=multi_process_chunks, verbose=True,
                                        out_dtype=out_dtype, method=method, n_cpu=None)
        del data_chunk
        gc.collect()
        n_read = end
        print(n_read)

    return output
