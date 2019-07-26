"""Functions to filter continuous data

Filter is a wrapper to simplify the use of scipy.filtfilt box
boxfilter is a simple running average.

"""

from scipy.signal import filtfilt, bessel, butter
from numpy import array, convolve, ones
import numpy as np


def custom_filter(data, dt, params=None, filt=None, axis=1, cutFreq=None,
                  filterType=None, filterShape='butter', order=4, padlen=None):
    """filter an array,

    Parameraters can be given as a single dictionary params containing:

    'cutFreq', cutting frequency in Hz
    'filterType', lowpass or highpass
    'order', order of the filter
    'filterShape', bessel, butter or box

    or as separated keyword arguments
    """
    if params is None:
        params = dict(cutFreq=cutFreq, filterShape=filterShape,
                      filterType=filterType, order=order)
    shape = params['filterShape']
    if shape == 'box':
        w_size = 1 / float(params['cutFreq'])
        if w_size < dt:
            raise ValueError('Window smaller than sampling interval\n' +
                             'reduce cutFreq below %.2f Hz' % (1 / dt))
        w = int(round(w_size / dt))
        filt_data = array(data, copy=True)
        for (i, d) in enumerate(data):
            filt_data[i] = boxfilter(d, w)
        f_type = params['filterType']
        if f_type == 'lowpass':
            return filt_data
        else:
            assert f_type == 'highpass'
            return data - filt_data
    else:
        if filt is None:
            filt = _create_filter(params, dt)
        return filtfilt(filt[0], filt[1], data, axis=axis, padlen=padlen)


def _create_filter(params, dt):
    shape = params['filterShape']
    if shape == 'box':
        return
    elif shape == 'butter':
        func = butter
    elif shape == 'bessel':
        func = bessel
    else:
        raise IOError("shape must be box, butter or bessel")
    ftype = params['filterType']
    order = params['order']
    cut_freq = params['cutFreq']
    nyq = 1 / (dt / 2.)
    if cut_freq > 1 / (dt / 2.):
        raise ValueError('Filter with cutFreq > Nyquist')
    cut = cut_freq / nyq
    # *** Potential error ***
    # If shape is neither box, butter nor bessel then func is not assigned
    # before being used, hence runtime error
    filter_func = func(order, cut, ftype)
    return filter_func


def lfp(data, sampling, low_pass=300, axis=1):
    params = dict(cutFreq=low_pass, filterType='lowpass', order=4, filterShape='bessel')
    return custom_filter(data, 1 / float(sampling), params, axis=axis)


def boxfilter(sweep, w, cumulativ=None, keep_borders=True):
    """Running average of `sweep` with a window of `w` point

    You can provide the cumulative sum of the sweep if you have it,
    the speed improvement for reasonable dataset was marginal

    The beginning and the end of the sweep that cannot be filtered
    (from 0 to w/2 and from -w/2 to the end) are kept unfiltered to
    return an array with the same length as sweep if `keep_borders` is
    True. They are cut otherwise

    """
    w = int(w)
    if w == 0:
        return sweep

    if sweep.ndim > 1:
        raise IOError('Sweep must be unidimensional')
    if cumulativ is None:
        cumulativ = sweep.cumsum()
    output = (cumulativ[w:] - cumulativ[:-w]) / float(w)
    if keep_borders:
        not_filtered_begin = sweep[:int(w / 2 + w % 2)]
        not_filtered_end = sweep[-int(w / 2):] if w != 1 else np.array([])
        output = np.hstack((not_filtered_begin, output, not_filtered_end))
    return output


def minmax_decimate(sweep, w, max_number_of_points=None):
    """Decimate by keeping min and max in window w.

    If max_number_of_points is specified, increase the window until
    the output is smaller than this number of points

    """
    N = len(sweep)
    if max_number_of_points is not None:
        max_w = int(N /max_number_of_points);
        if w < max_w:
            w = max_w

    # shorten sweep in a multiple of w and reshape
    decimated = np.zeros(int(N/w)*2)
    source = sweep[:w*int(N/w)].reshape(-1, w)
    decimated[::2] = source.min(1)
    decimated[1::2] = source.max(1)
    return decimated


def filter_convolve(sweep, window_size, keep_borders=True):
    """
    Filter a time serie with a simple running average through convolution
    The result can either be of the same size as the original data if
    ``keep_borders == True`` or of ``len(sweep) - window_size + 1 if
    keep_borders == False``

    Arguments
    ---------
    sweep: `Numpy` array
        Data to be filtered
    window_size: int
        Number of samples defining the window
    keep_borders: boolean
        Should the returned data be the same size as the input ones even though
        borders can not be filtered? See :func:`numpy.convolve` for details.

    Return:
    -------
    f_data: `Numpy` array
        Filtered data
    """
    if len(sweep) == 0 or window_size == 0:
        return sweep
    filt = ones(window_size, dtype=np.float64)
    filt = filt / filt.sum()
    if keep_borders:
        mode = 'same'
    else:
        mode = 'valid'
    return convolve(sweep, filt, mode)


def _butter_bandpass(lowcut, highcut, fs, order=5):
    # from http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
