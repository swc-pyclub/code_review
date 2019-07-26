"""Utility functions relatively common

Contains mostly small helper functions"""

import numpy as np
from scipy import stats
from itertools import islice


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def embed(x, span):
    """
    From one vector and a window size make a 2D array with that vector
    shifted from 1 to span samples. Makes finding extremas over a specific
    window easier. Quite fast thanks to the strides tricks
    """
    x_e = np.lib.index_tricks.as_strided(x, [x.shape[0] - span] + [span],
                                         x.strides + (x.strides[0],))
    return x_e


def peaks(x, span):
    """
    Find peaks (maxima) in a certain window, to guard against noise
    """
    z = embed(x, span)
    s = int(span) / 2
    v = np.argmax(z, 1) == s
    result = np.hstack(([False] * s, v, [False] * s))
    return result


from . import gaussian_fit_2d, oe_stats


def check_input(value, possible_values, all_if_none=True):
    """check if value is a list of element in possibleValues
    if value is None return possibleValues if allIfNone, None else"""
    if value is None:
        return possible_values if all_if_none else None
    if not isinstance(value, list):
        value = [value]
    if any([i not in possible_values for i in value]):
        raise ValueError('%s not in possible values' % value)
    return value


def half_gauss_density(data, sd, start=None, end=None, dstep=None, verbose=True):
    """ Takes a sequence of spike times and produces a non-normalised density
    estimate by summing Normals defined by sd at each spike time. The range of
    the output is guessed from the extent of the data (which need not be
    ordered), the resolution is automagically determined from sd; we currently
    used sd*0.05 A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows. """

    # Note: once I've understood convolutions and Fourier transforms, they
    # probably represent the quick way of doing this.
    # note: try to fft this

    # Resolution as fraction of sd

    data = np.array(data)
    dmax = np.max(data) + sd * 4. if end is None else float(end)
    dmin = np.min(data) - sd * 4. if start is None else float(start)

    res = 0.05
    if dstep is None:
        dstep = sd * res
    else:
        if dstep > sd * res:
            if verbose: print('Warning dstep big relative to sd')
    time = np.arange(dmin, dmax, dstep)

    # dens = np.zeros_like(time)
    # def t_to_i(t):
    #    return int(round((t-dmin)/dstep))

    hal = int(time.size / 2)
    gauss = np.zeros(time.size, dtype='float')
    gauss[hal:] = 2 / np.sqrt(2 * np.pi * sd ** 2) * np.exp(-(time[hal:]) ** 2 /
                                                            (2 * sd ** 2))
    kernel = np.vstack((time, gauss))

    time, dens = kernel_density(data, kernel, dmin=dmin, dmax=dmax, dstep=dstep,
                                verbose=verbose)
    return np.vstack((time, dens))


def half_exp_density(data, sd):
    """ Takes a sequence of spike times and produces a non-normalised density
    estimate by summing Half-exponential (asymetric) defined by sd at each spike
    time. The range of the output is guessed from the extent of the data (which
    need not be ordered), the resolution is automagically determined from sd; we
    currently used sd/10. A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows. """

    # Resolution as fraction of sd
    res = 0.1
    data = np.array(data)
    dmax = float(np.max(data) + sd * 4.)
    dmin = float(np.min(data) - sd * 4.)
    dstep = sd * res
    time = np.arange(start=dmin, stop=dmax, step=dstep)
    if time.size % 2 != 0:
        time = time[:-1]
    r = np.arange(0, len(time), dtype=int)
    hal = r.size / 2
    exp = np.zeros(r.size, dtype='float')
    exp[hal:] = 2 / np.sqrt(2 * sd ** 2) * np.exp(
        -np.sqrt(2) * np.abs(time[hal:]) / sd)

    time, dens = kernel_density(data, np.vstack((time, exp)))
    return np.vstack((time, dens))


def exponential_density(data, sd, start=None, end=None, dstep=None):
    """ Takes a sequence of spike times and produces a non-normalised density
    estimate by summing Exponential defined by sd at each spike time. The range of
    the output is guessed from the extent of the data (which need not be
    ordered), the resolution is automagically determined from sd; we currently
    used sd/10. A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows. """

    data = np.array(data)
    dmax = np.max(data) + sd * 4. if end is None else float(end)
    dmin = np.min(data) - sd * 4. if start is None else float(start)

    res = 0.05
    if dstep is None:
        dstep = sd * res
    else:
        if dstep > sd * res:
            print('Warning dstep big relative to sd')

    # Resolution as fraction of sd

    time = np.arange(start=dmin, stop=dmax, step=dstep)
    exp = 1 / np.sqrt(2 * sd ** 2) * np.exp(-np.sqrt(2) * np.abs(-time) / sd)
    time, dens = kernel_density(data, np.vstack((time, exp)))
    return np.vstack((time, dens))


def gaussian_density(data, sd, start=None, end=None, dstep=None, verbose=True):
    """ Takes a sequence of spike times and produces a non-normalised density
    estimate by summing Normals defined by sd at each spike time. The range of
    the output is guessed from the extent of the data (which need not be
    ordered), the resolution is automagically determined from sd; we currently
    used sd*0.05 A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows. """

    # Note: once I've understood convolutions and Fourier transforms, they
    # probably represent the quick way of doing this.
    # note: try to fft this

    # Resolution as fraction of sd

    data = np.array(data)
    dmax = np.max(data) + sd * 4. if end is None else float(end)
    dmin = np.min(data) - sd * 4. if start is None else float(start)

    res = 0.05
    if dstep is None:
        dstep = sd * res
    else:
        if dstep > sd * res:
            if verbose: print('Warning dstep big relative to sd')
    time = np.arange(dmin, dmax, dstep)

    # dens = np.zeros_like(time)
    # def t_to_i(t):
    #    return int(round((t-dmin)/dstep))

    norm = 1 / np.sqrt(2 * np.pi * sd ** 2) * np.exp(
        -(time - time[int(time.size / 2)]) ** 2 /
        (2 * sd ** 2))
    kernel = np.vstack((time, norm))

    time, dens = kernel_density(data, kernel, dmin=dmin, dmax=dmax, dstep=dstep,
                                verbose=verbose)
    # for t in data:
    #      dens[t_to_i(t-sd*3.)+r] += norm
    return np.vstack((time, dens))


def kernel_density(data, kernel, dmin=None, dmax=None, dstep=None, verbose=True):
    """Kernel density estimation

    Given a 2-D kernel (one line for the time, one for the values) and a
    list of time (data), compute the kde (just do the convolution basically)

    if dmin and/or dmax not None, use it as the minimum/maximum time value for
       the output
    else the output has the minimum size to fit all data points plus a kernel
       half-width


    return time, kde two 1-D arrays
    """

    if dstep is None:
        dstep = kernel[0][1] - kernel[0][0]
    length = kernel[0][-1] - kernel[0][0]
    if dmin is None:
        dmin = data.min() - length / 2
    if dmax is None:
        dmax = data.max() + length / 2

    time = np.arange(dmin, dmax, dstep)

    # add one dstep to have the smallest time bigger than dmin to dmax
    out = np.zeros(time.size, dtype=int)
    ignored = 0
    for d in data:
        if dmin < d < dmax:
            out[int((d - dmin) / dstep)] += 1
        else:
            ignored += 1
    bigkernel = np.zeros(out.size, dtype='float')
    beg = int(bigkernel.size / 2 - kernel.shape[1] / 2)
    end = int(bigkernel.size / 2 + int(kernel.shape[1] / 2. + 0.5))
    bigkernel[beg:end] = kernel[1][:]
    conv = np.convolve(bigkernel, out, mode='same')
    if ignored > 0 and verbose:
        print('%i data points out of [dmin, dmax] interval were ignored' % ignored)
    return time, conv


def findextrema(sweep, maximum=1, threshold=None, pointinterval=0):
    """Find maxima (if maximum=1/True) or minima (if maximum=0/False) that fall
    beyond threshold

    Requires crossing of threshold between extrema of given sign. If
    there are several (e.g.) maxima interspersed with local minima
    beyond threshold, a single largest maximum is retained. This
    essentially guards against noise.

    TODO: make "rearming" an
    option.

    """
    deriv = np.diff(sweep)
    index = np.arange(len(sweep))

    # Should find both maxima and minima.
    # But what about inflections??
    if threshold is None:
        if maximum:
            extindex = index[:-2][np.logical_and((deriv[:-1] > 0), (deriv[1:]) <= 0)] + 1
        else:
            extindex = index[:-2][np.logical_and((deriv[:-1] <= 0), (deriv[1:] > 0))] + 1
    else:
        if maximum:
            extindex = index[:-2][np.logical_and(np.logical_and((deriv[:-1] > 0), (deriv[1:]) <= 0),
                                                 (sweep[1:-1] >
                                                  threshold))] + 1
        else:
            extindex = index[:-2][np.logical_and(np.logical_and((deriv[:-1] <= 0), (deriv[1:]) > 0),
                                                 (sweep[1:-1] < threshold))] + 1
    extindex = np.asarray(extindex)
    if extindex.size > 1:
        # retain only the maximum in a windows of pointint points.
        cleansample = np.concatenate([[0], np.where((extindex[1:] - extindex[:-1])
                                         > pointinterval)[0] + 1,
                                      [len(extindex+1)]])

        func = np.argmax if maximum else np.argmin

        cleanextindex = np.zeros(len(cleansample)-1, dtype=int)
        for i in range(len(cleansample) - 1):
            indices = extindex[cleansample[i]:cleansample[i + 1]]
            values = sweep[indices]
            cleanextindex[i] = indices[func(values)]
    else:
        cleanextindex = np.array(extindex)
    return cleanextindex


def two_dim_findextrema(sweep, axis=1, maximum=1, threshold=None,
                        pointinterval=0):
    """Find maxima (if maximum=1/True) or minima (if maximum=0/False) that fall
    beyond threshold; requires crossing of threshold between extrema of given
    sign; if there are several (e.g.) maxima interspersed with local minima
    beyond threshold, a single largest maximum is retained. This essentially
    guards against noise.

    Return a 2D array([[line_index, line_index ...]
                      [col_index,  col_index ...]])
    TODO: make "rearming" an option."""
    if sweep.ndim != 2:
        raise IOError('I can deal only with 2D arrays')
    if axis == 0:
        sweep = sweep.T
    elif axis != 1:
        raise IOError('I can deal only with 2D arrays, axis must be 0 or 1')
    deriv = np.diff(sweep, 1)

    if threshold is None:
        if maximum:
            extindex = np.where(
                np.logical_and((deriv[:, :-1] > 0), (deriv[:, 1:]) <= 0))
        else:
            extindex = np.where(
                np.logical_and((deriv[:, :-1] <= 0), (deriv[:, 1:] > 0)))
    else:
        if maximum:
            extindex = np.where(np.logical_and(
                np.logical_and((deriv[:, :-1] > 0), (deriv[:, 1:]) <= 0),
                (sweep[:, 1:-1] > threshold)))
        else:
            extindex = np.where(np.logical_and(
                np.logical_and((deriv[:, :-1] <= 0), (deriv[:, 1:]) > 0),
                (sweep[:, 1:-1] < threshold)))

    if extindex[0].size > 1:
        # retain only the maximum in a windows of pointint points.
        diff_pt = np.diff(extindex[1])
        chline = np.asarray(np.diff(extindex[0]), dtype='bool')

        # find groups to sort: need to be on the same line and
        # < pointinterval (if they are negativ they should be on the same line)
        cleansample = np.asarray(
            np.logical_and(np.logical_not(chline), diff_pt <= pointinterval),
            dtype='int')
        # notOKPoints = np.hstack((cleansample[0],np.logical_or(cleansample[:-1], cleansample[1:]),
        #                          cleansample[-1]))
        # add cleansample[0] in case first is in pointint (first diff will be -1)
        # add -cleansample[-1] to add a last -1 in case it ends with a 1
        groupchg = np.hstack((cleansample[0],
                              np.diff(np.asarray(cleansample, dtype='int')),
                              -cleansample[-1]))
        begs = np.where(groupchg == 1)[0]
        ends = np.where(groupchg == -1)[0]
        out = (list(extindex[0]), list(extindex[1]))
        func = np.argmax if maximum else np.argmin
        shift = 0
        for b, e in zip(begs, ends):
            line = extindex[0][b]
            totest = extindex[1][b:e + 1]
            # check to delete at some point:
            assert extindex[0][e] == line
            assert all(np.diff(extindex[1][b:e])) < pointinterval
            if e + 1 < len(extindex[1]):
                assert (
                           extindex[1][e + 1] - extindex[1][e]) > pointinterval or \
                       (extindex[0][e + 1] - extindex[0][e]) != 0
            val = totest[func(sweep[line, totest])]
            b -= shift
            e -= shift
            out[1][e] = val
            del out[0][b:e]
            del out[1][b:e]
            shift += e - b

        out = np.array(out)
    else:
        out = np.array(extindex)
    out[1] += 1  # add one because of deriv (I think)
    return out


def cross_threshold(sweep, rise=True, maximum=True, threshold=0,
                    pointinterval=0):
    """Find indices of sweep where a threhsold is crossed

    Requires crossing of `threshold` from a lower to a bigger value if
    `rise`, from above to below if not `rise`. If there are several
    crossings interspersed in `pointinterval`, the last crossing is
    kept if if `maximum`, the first otherwise

    TODO: make "rearming" an option.

    """

    index = np.arange(len(sweep))
    threshold = float(threshold)

    if rise:
        extindex = index[:-1][np.logical_and((sweep[1:] > threshold),
                                             (sweep[:-1] <= threshold))]
    else:
        extindex = index[:-1][np.logical_and((sweep[1:] < threshold),
                                             (sweep[:-1] >= threshold))]

    if len(extindex) > 1:
        # retain only the maximum in a windows of pointint points.
        # I will look at the window cleansample[i]:cleansample[i+1], so need +1 to have the end sample
        cleansample = np.hstack([0,
                                 np.where(np.diff(extindex) > pointinterval)[0] + 1,
                                 len(extindex)])
        func = np.max if maximum else np.min
        cleanextindex = np.zeros(len(cleansample) - 1)
        for i in range(len(cleanextindex)):
            cleanextindex[i] = func(extindex[cleansample[i]:cleansample[i + 1]])
    else:
        cleanextindex = extindex
    # raise IOError
    if any(np.diff(cleanextindex) < 0):
        raise ValueError('Error')
    return cleanextindex


def flaten_list(list_of_arrays):
    """Return a single array from list of arrays

    Usefull mostly if you have lot of arrays with different length
    (typically a crosscorrelogram)"""
    if not list_of_arrays:
        return np.array([])
    ntot = sum(map(np.size, list_of_arrays))
    out = np.array(np.zeros(ntot), dtype=list_of_arrays[0].dtype)
    n = 0
    for i, j in enumerate(map(np.size, list_of_arrays)):
        out[n:n + j] = list_of_arrays[i]
        n += j
    return out


def flaten_series(series):
    """Same as flatenList but works with a panda series
    returns a series with a single line"""
    import pandas as pd
    if not isinstance(series, pd.Series):
        print('I work only with Series, don\'t complain if I crash')
    if not len(series):
        return pd.Series(np.array([]))
    ntot = sum(map(np.size, series))
    out = np.array(np.zeros(ntot), dtype=series.iloc[0].dtype)
    n = 0
    for i, j in enumerate(map(np.size, series)):
        out[n:n + j] = series.iloc[i]
        n += j
    return pd.Series(out)


def flaten_df_list(dataframe_of_array):
    """Same as flatenList but works with a panda dataframe
    returns a dataframe with a single line"""
    import pandas as pd

    # generate empty dataframe
    col = dataframe_of_array.columns
    data = [pd.Series(np.array([])) for _ in col]
    out_df = pd.DataFrame(dict(zip(col, data)))
    if not len(dataframe_of_array):
        return out_df
    for c in col:
        out_df[c] = [np.array(flaten_series(dataframe_of_array[c]))]
    return out_df


def cumulativ(arr, win2linearise=None, no_time=False):
    """do a cumulativ sum over one axis and subtract linear regression
    of win2linearise

    if arr.ndim is 1, win2linearise is the index of the first and last
    points to use
    if arr.ndim is 2, if """
    if arr.ndim == 1 or no_time:
        time = np.arange(arr.size)
        cumsum = np.array(arr, ndmin=2).cumsum(1)
    elif arr.ndim == 2:
        time = arr[0]
        cumsum = arr[1:].cumsum(1)
    else:
        raise ValueError('need a 2-d array')

    if win2linearise is None:
        win2linearise = [0, time.size]
    else:
        win2linearise = time.searchsorted(win2linearise)
    out = np.array(np.zeros_like(arr), ndmin=2)
    if arr.ndim == 2 and not no_time:
        out[0] = time
    for ind, i in enumerate(cumsum):
        gradient, intercept, r_value, p_value, std_err = \
            stats.linregress(time, i[win2linearise[0]: win2linearise[1]])
        if r_value < 0.9:
            print('Bad fit, gradient:%s' % gradient)
            print('intercept: %.2f' % intercept)
            print('r_value: %.2f' % r_value)
            print('p_value: %.2f' % p_value)
            print('std_err: %.2f' % std_err)
        out[ind + 1] = i - time * gradient + intercept
    if arr.ndim == 1:
        return out[0]
    return out


def filter_values(array, comp, value):
    """Test if values of an array agree with comp

    comp can be 0 for inferior to, 1 for superior to, 'in', or 'out' to select a
         window of value and 'among' to keep only specific values
    value is a float if comp is 0 or 1 and a doublet of float if comp is 'in' or
         'out', a list of values if comp is 'among'
    """
    if comp in [0, 1]:
        testarr = array - value
        if not comp:
            testarr *= -1
        out = testarr >= 0
    elif comp in ['in', 'out']:
        vals = [float(v) for v in value]
        assert len(vals) == 2
        if comp == 'in':
            out = np.logical_and(array > vals[0], array < vals[1])
        else:
            out = np.logical_xor(array < vals[0], array > vals[1])
    elif comp == 'among':
        out = reduce(np.logical_or, [array == i for i in value])
    else:
        raise IOError('unknown comp: %s' % comp)
    return out


def r_wilcox(array, array2=None, paired=False):
    """Just a small function to remind me how to do a paired wilcoxon test using
    rpy2.

    Get the p value like that: wilcox_result[2][0]

    Note that most of other modules (scipy.stats, pandas) use a gaussian
    approximation when performing wilcoxon tests and are not suitable for
    low (<30) N"""

    import rpy2
    import rpy2.robjects
    if array2 is None:
        wilcox_result = rpy2.robjects.r['wilcox.test'](
            rpy2.robjects.FloatVector(array))
    else:
        wilcox_result = rpy2.robjects.r['wilcox.test'](
            rpy2.robjects.FloatVector(array),
            rpy2.robjects.FloatVector(array2),
            paired=paired)
    return wilcox_result
