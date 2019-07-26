# -*- coding: utf-8 -*-
"""Functions related to analysis of time series

use cc_func to make a cross correlogram. The rest of the functions
should allow to do some recurence test analysis (inspired by Johnson,
D. and Kiang, N. (1976). Analysis of discharges recorded
simultaneously from pairs of auditory nerve fibers. Biophysical
Journal, 16:719–734.)

"""

import numpy as np
from numpy import logical_not as lnot
from numpy import logical_and as land
from OpenEphys.common import flaten_list


def cc_func(ls0, ls1, trange, last_bef_cond=None, keep_zero=True, check = False):
    """Compute crosscorrelogram between two time series

    If force is not True, will assert if ls1 is sorted. ls0 does not need to be sorted
    keep_zero keeps or reject event synchronous in both time series (useful for autocorrs)
    last_beg_cond is a tuple (cmp, value). If the last spike before the event does not
    respect the condition, the crosscorr for that ls0 event will be discarded. (IIRC it was
    used to get cc with no other ls1 event before ls0 in the pinceau analysis)

    :param ls0:
    :param ls1:
    :param trange:
    :param last_bef_cond:
    :param keep_zero:
    :param force:
    :return:
    """
    dts = []
    if check:
        assert all(np.sort(ls1)==ls1)

    for spike in ls0:
        trangei = ls1.searchsorted(spike + trange)
        # find where the trange around this spike would fit in other.spikes
        if hasattr(trangei, 'count'):
            if not trangei.count():
                dts.append(np.array([]))
                continue
        dt = ls1[trangei[0]:trangei[1]] - spike
        if last_bef_cond is not None:
            lastbef = dt[:dt.searchsorted(0)]
            if lastbef.size:
                lastbef = lastbef[-1]
                if cmp(lastbef, last_bef_cond[1]) != last_bef_cond[0]:
                    dt = np.array([])
            elif last_bef_cond[0] in [0, 1]:
                dt = np.array([])
                # find dt between this spike and only those other.spikes that are in trange of this spike
        dts.append(dt)
    # ipshell()
    if not keep_zero:
        dts = [line[line != 0] for line in dts]
    return dts


def recurtime(array0, array1, border=None, exclude_zero=False,
              relative_time=False):
    """Return a list of time bef and time aft array0

    if border is None, put nan for missing values (first and last time)
    if border is "exclude" keep only time with a tbef and a taft
    if exclude_zero is True, do not count synchronous times
    if relative_time,return the shift, otherwise the absolute time
    """
    if not array0.size or not array1.size:
        print("One empty array in recurtime")
        if border is None:
            return [np.ones(array0.size) * np.inf] * 2
        else:
            return [np.array([])] * 2
    # initalise output
    exact_match = np.zeros(array0.shape, dtype=bool)
    afts = np.zeros(array0.shape) + np.nan
    tbefs = np.zeros(array0.shape) + np.nan
    tafts = np.zeros(array0.shape) + np.nan

    # search array0 in array1
    ind = array1.searchsorted(array0)

    # first deal with the middle:
    tbefs[ind > 0] = array1[ind[ind > 0]-1]
    tafts[ind < len(array1)] = array1[ind[ind < len(array1)]]

    # find exact match:
    exact_match = tafts == array0
    if exclude_zero:
        # pb if the last match is exact
        lastmatch = ind >= len(array1)-1
        to_push = land(exact_match, lnot(lastmatch))
        to_nan =  land(exact_match, lastmatch)
        tafts[to_push] = array1[ind[to_push]+1]
        tafts[to_nan] = np.nan
    else:
        tbefs[exact_match] = array0[exact_match]
        tafts[exact_match] = array0[exact_match]

    if relative_time:
        tbefs = array0 - tbefs
        tafts = tafts - array0

    if border =='exclude':
        to_ret = land(lnot(np.isnan(tbefs)), lnot(np.isnan(tbafts)))
    else:
        to_ret = ones(tbefs.shape, dtype=bool)
    return tbefs[to_ret], tafts[to_ret]

def epdf(array, step=1, start=None, end=None, x=None):
    """ Compute an histogram of probability density fonction of an array
    x is an array of len = len(array)+1, it's the borders of the histo
    if x is None, an x is computed from start, end and step,


    return x, the time corresponding and the edpf at that values"""
    if x is not None:
        if start is not None or end is not None:
            raise ValueError('cannot use start/end AND x')
    else:
        if start is not None:
            array = array[array > start]
        else:
            start = array.min()
        if end is not None:
            array = array[array < end]
        else:
            end = array.max()
        x = np.arange(start, end, step)
    deltas = np.diff(x)
    array.sort()
    pdf = np.diff(array.searchsorted(x)) / (deltas * array.size)
    return x[:-1], pdf


def recur_test(train0, train1, step, start=None, end=None):
    """trains must be two list of arrays"""
    isi = [np.diff(t) for t in train0]
    flat_isi = flaten_list(isi)
    n_b = flaten_list(train1).size

    x, pdf_isi = epdf(flat_isi, step, start=start, end=end)
    ecdf_isi = np.cumsum(pdf_isi) * step
    e_isi = flat_isi.mean()
    pv_0 = 1 / e_isi * (ecdf_isi[-1] - ecdf_isi)

    befs, afts = [flaten_list(i) for i in recurtime(train0, train1)]
    npv_bef, xb = np.histogram(befs, np.hstack((x, x[-1] + step)))
    npv_aft, xt = np.histogram(afts, np.hstack((x, x[-1] + step)))

    sigma = np.sqrt(n_b * pv_0 * step * (1 - pv_0 * step))

    res_b = npv_bef - pv_0 * n_b * step
    res_a = npv_aft - pv_0 * n_b * step
    return x, flat_isi, pv_0, npv_bef, npv_aft, res_b, res_a, sigma


def stack_bef_aft(x, res_b, res_a, sigma):
    res = np.hstack((res_b[::-1], res_a))
    x = np.hstack((-x[::-1], x))
    s = np.hstack((sigma[::-1], sigma))
    return x, res, s


def plot_recur(x, residual, sigma, nsigm=4, ax=None):
    if ax is None:
        from matplotlib.pyplot import figure
        fig = figure()
        ax = fig.add_subplot(111)
    ax.fill_between(x, nsigm * sigma, -nsigm * sigma, color='grey', alpha=.3)
    ax.plot(x, residual, color='k', drawstyle='steps-post')
    pos_sign = residual > nsigm * sigma
    neg_sign = residual < -nsigm * sigma
    ax.bar(x[pos_sign], residual[pos_sign] - nsigm * sigma[pos_sign], np.diff(x[:2])[0],
           bottom=nsigm * sigma[pos_sign], color='r', lw=0, alpha=.5)
    ax.bar(x[neg_sign], residual[neg_sign] + nsigm * sigma[neg_sign], np.diff(x[:2])[0],
           bottom=-nsigm * sigma[neg_sign], color='b', lw=0, alpha=.5)
    ax.set_xlim(-25, 25)
    return ax, pos_sign, neg_sign


def full_test(train0, train1, start=0, end=50, step=.5, nsigm=4):
    small_x, isi, pv_0, npv_bef, npv_aft, res_b, res_a, sigma = recur_test(train0, train1, step, start, end)
    x, r, s = stack_bef_aft(small_x, res_b, res_a, sigma)
    ax, psig, nsig = plot_recur(x, r, s, nsigm)
    return x, r, s, psig, nsig, ax
