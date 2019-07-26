import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import plot_helper as ph
from ..time_series_analysis import cc_func
from ..common import flaten_list
from ..continuous_analysis import extract_sts
from scipy import stats

def autocorr_hist_plot(ax, trange, spks=None, mean_f=None,
                       ref_per=1.5e-3, cor=None, bin_width=0.5e-3,
                       **kwargs):
    if cor is None:
        if spks is None:
            raise IOError('Need to give cor or spks')
        cor = cc_func(np.array(spks), np.array(spks),
                      trange=trange, keep_zero=False)
        mean_f = 1 / float(np.diff(spks).mean())
    elif mean_f is None:
        if spks is not None:
            mean_f = 1 / float(np.diff(spks).mean())
    h, b = np.histogram(flaten_list(cor), np.arange(trange[0], trange[1], bin_width))
    bad = b.searchsorted([0, ref_per])
    bad_h = h[bad].max()
    bad_m = h[bad].mean()
    ax.axhline(bad_h, color='r')
    if mean_f is not None:
        h_mean = mean_f * bin_width * len(cor)
        ax.axhline(h_mean, color='k')

    d = dict(drawstyle='steps-post')
    d.update(**kwargs)
    ax.plot(b[:-1] * 1000, h, **d)
    ax.set_xlim(*np.array(trange) * 1000)
    measure = bad_h / h_mean * 100.
    msg = '%i events < %.1fms\n(%.2f%% of random)' % (bad_h, ref_per * 1000,
                                                      measure)
    ax.text(0.9, 0.9, msg, color='darkred' if measure > 1 / 100. else 'g',
            transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right')
    ax.set_xlabel('Time (ms)')
    return cor, h_mean, bad_h, bad_m, measure


def autocorr_raster_plot(ax, trange, spks=None, cor=None, **kwargs):
    """

    kwargs arre given to plt.scatter
    :param ax:
    :param trange:
    :param spks:
    :param cor:
    :param kwargs:
    :return:
    """
    if cor is None:
        if spks is None:
            raise IOError('Need to give cor or spks')
        cor = cc_func(np.array(spks), np.array(spks),
                      trange=trange, keep_zero=False)

    ys = [np.zeros_like(v) + i for i, v in enumerate(cor)]
    d = dict(color='k', alpha=.1, s=1)
    d.update(kwargs)
    ax.scatter(flaten_list(cor) * 1000, flaten_list(ys), **d)
    ax.set_ylim(0, len(cor))
    ax.set_xlim(*np.array(trange) * 1000)
    return cor, ys


def firing_rate_plot(ax, spks, win2av=10, part2check=None):
    if len(spks) == 1:
        print('a single spike has no rate')
        return None
    if part2check is None:
        min_end = 0
    else:
        min_end = part2check[1] + win2av / 2
    end = max(spks.max() + win2av / 2, min_end)
    borders = np.arange(0, end + win2av / 2, win2av)
    spk_cnt = np.array(np.diff(np.asarray(spks).searchsorted(borders)),
                       dtype=float)
    rate = spk_cnt / win2av  # avg fr on win2av windows
    isi = np.diff(spks)
    ifr = 1 /isi  # instantaneous firing rate
    # I can have simultaneous spikes, especially in noise cluster
    # that would produce infinite ifr, put 0 instead (because hist
    # does't like nan and 0 ifr is also a non-valid value)
    ifr[isi==0] = 0

    # create output dict
    if part2check is None:
        b, e = [0, len(borders) - 1]
    else:
        b, e = borders.searchsorted(part2check)
    gradient, intercept, r_value, p_value, std_err = \
        stats.linregress(np.array(borders[b:e]), np.array(rate[b:e]))
    out = dict(fr_win=win2av, av_fr=rate[b:e].mean(),
               std_fr=rate[b:e].std(), av_ifr=ifr[b:e].mean(),
               fr_fit_slope=gradient, fr_fit_inter=intercept,
               fr_fit_r=r_value)

    # create axes
    bbox = ax.get_position()
    l, b, w, h = bbox.bounds
    ax.clear()
    fig = ax.get_figure()
    fig.axes.remove(ax)
    ax.set_axis_off()

    ax = fig.add_axes((l, b, w * 0.9, h))
    ax_side = fig.add_axes((l + w * 0.9, b, w * 0.1, h))

    ax.plot(spks[:-1], ifr, '.', alpha=.5)
    ax.plot(borders[:-1], rate, drawstyle='steps', lw=2)

    h, b = np.histogram(ifr, bins=np.arange(0, min(max(ifr), 500), min(.25, max(ifr) / 10.)))
    ax_side.bar(left=np.zeros_like(h),
                height=np.diff(b),
                bottom=b[:-1],
                width=np.array(h, dtype=float) / h.max(),
                color=ph.set2[0])
    h, b = np.histogram(rate, bins=np.arange(0, max(rate), min(.25, max(rate) / 10.)))
    out['cv_fr'] = h.std() / h.mean()
    ax_side.bar(left=np.zeros_like(h),
                height=np.diff(b),
                bottom=b[:-1],
                width=np.array(h, dtype=float) / h.max(),
                color=ph.set2[1],
                edgecolor=ph.set2[1], alpha=.7)
    # ax_side.plot(np.array(h,dtype=float)/h.max(), b[:-1],
    #             drawstyle = 'steps', lw =2)
    ax.set_ylim(0, max(np.median(ifr), 10))
    ax_side.set_ylim(0, max(np.median(ifr), 10))
    ax.set_ylabel('Firing freq (Hz)')
    xt = ax_side.get_xticks()
    ax_side.set_xticks(xt[1::len(xt) / 3])
    ax.set_xticks(ax.get_xticks()[:-1])
    ax.set_xlabel('Time (s)')
    ph.clean(ax)
    ph.clean(ax_side, spines=['top', 'left'])

    return out, borders, rate


def isi_plot(ax, spks, ref_per=2e-3, limit=None):
    if len(spks)==1:
        print('single spike has not isi')
        return
    isi = np.diff(spks)
    if limit is None:
        limit = 50e-3  # np.median(isi)*1.1

    isi_1ms = isi < ref_per
    n1 = np.sum(isi_1ms)
    bbox = ax.get_position()
    l, b, w, h = bbox.bounds
    inset_p = (l + w / 2, b + h / 2, w / 2, h / 2)
    inset = ax.get_figure().add_axes(inset_p)

    h, b = np.histogram(isi, bins=np.arange(0, max(isi), 10e-3))
    inset.plot(b[:-1], h, drawstyle='steps')
    h, b = np.histogram(isi, bins=np.arange(0, limit, 1e-3))
    ax.plot(b[:-1] * 1000, h, drawstyle='steps')
    ax.axhline(n1, color='r')
    ax.set_xlabel('Time (ms)')
    inset.set_xlabel('Time (s)')
    ph.clean(inset, spines=['top', 'left', 'right'])
    color = 'g' if n1 / float(len(isi)) < 1 else 'r'
    msg = '%i ISI <%.1fms\n(%.2f%%)' % (n1, ref_per * 1000, n1 / float(len(isi)))
    inset.text(0.9, 0.9, msg, transform=inset.transAxes,
               verticalalignment='top', horizontalalignment='right', color=color)
    for i, x in enumerate((inset, ax)):
        f = 1000 if i else 1  # put in ms for ax
        inset2 = x.twinx()
        bp = inset2.boxplot(np.array(isi) * f, vert=False, showfliers=False)
        plt.setp(bp['boxes'], color=ph.almost_black)
        plt.setp(bp['whiskers'], color=ph.almost_black,
                 linestyle='-')
        plt.setp(bp['caps'], ls='')
        plt.setp(bp['medians'], color=ph.set2[0])
        inset2.set_yticks([])
        inset2.set_ylim(0.75, 2)
        ph.clean(inset2, spines=['top', 'left', 'right', 'bottom'])
    inset.set_xlim(0, min(np.percentile(isi, 80), .5))
    ax.set_xlim(0, limit * 1000)
    out = dict(isi_in_ref=n1, n_isi=len(isi),
               isi_ref_per=ref_per,
               pisi_in_ref=n1 / float(len(isi)))
    return out


def quality_figure(fig, spks, trange=[-150e-3, 150e-3], part2check=None,
                   ref_per=1.8e-3):
    cor = cc_func(np.array(spks), np.array(spks),
                  trange=trange, keep_zero=False)
    ax = fig.add_subplot(4, 3, 1)
    autocorr_raster_plot(ax, trange, spks=spks, cor=cor)
    ph.clean(ax)
    ax = fig.add_subplot(4, 3, 4)
    cor, h_mean, bad_h, bad_m, measure = autocorr_hist_plot(ax, trange, spks=spks,
                                                     cor=cor, ref_per=ref_per)
    ph.clean(ax)

    ax = fig.add_subplot(4, 3, 2)
    autocorr_raster_plot(ax, [-20e-3, 20e-3], spks=spks, cor=cor)
    ph.clean(ax)
    ax = fig.add_subplot(4, 3, 5)
    cor, h_mean, bad_h, bad_m, measure = autocorr_hist_plot(ax, [-20e-3, 20e-3], spks=spks,
                                                            cor=cor, ref_per=ref_per)
    df = pd.DataFrame(dict(autocorr_measure=[measure],
                           autocorr_h_mean=[h_mean],
                           autocorr_h_bad=[bad_h],
                           autocorr_mean_bad=[bad_m],
                           autocorr_ref_per=[ref_per]))

    ph.clean(ax)
    ax = fig.add_subplot(2, 3, 3)
    dictout = isi_plot(ax, spks, ref_per=ref_per, limit=.4)
    if dictout is not None:
        for k, v in dictout.items():
            df[k] = [v]
    ph.clean(ax)
    ax = fig.add_subplot(2, 1, 2)
    output = firing_rate_plot(ax, spks, win2av=10,
                                              part2check=part2check)
    if output is not None:
        dictout, borders, rate = output
        for k, v in dictout.items():
            df[k] = [v]
    return df

def plot_spike_waveforms(ax, mean_waveforms, gain, sampling, std_waveforms=None,
                         channels=None, spacing=25,
                         line_kwargs={}, area_kwargs={}):
    if channels is None:
        channels = np.arange(mean_waveforms.shape[1])
    channels = np.asarray(channels)
    shift = -channels * spacing
    time = np.arange(mean_waveforms.shape[0]) / sampling * 1e3
    m_w = mean_waveforms * gain
    m_w += shift
    kw = {'color': 'k'}
    kw.update(line_kwargs)
    lines = ax.plot(time, m_w, **kw)
    for l, c in zip(lines, channels):
        l.set_label(str(c))
    if std_waveforms is not None:
        kw = {'color': 'k', 'alpha': .5}
        kw.update(area_kwargs)
        std_w = std_waveforms * gain
        for m, s in zip(m_w.T, std_w.T):
            shades = ax.fill_between(time, m - s,
                                     m + s, **kw)

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.plot([xl[0] + np.diff(xl) * 10 / 100.] * 2, [yl[1], yl[1] - spacing], color='k', lw=10)
    ax.text(xl[0] + np.diff(xl) * 10 / 100., yl[1] - spacing / 2, str(spacing) + ' microV',
            horizontalalignment='left', verticalalignment='center', rotation=90)
    return time

def make_waveform_plot(fig, mean_waveforms, std_waveforms, channels, main_spike, gain, sampling,
                       spacing=100):
    ax = fig.add_subplot(121)
    time = plot_spike_waveforms(ax, mean_waveforms, sampling=sampling, spacing=25, gain=gain)

    ax = fig.add_subplot(222)
    time = plot_spike_waveforms(ax, mean_waveforms=mean_waveforms[:, channels],
                                std_waveforms=std_waveforms[:, channels], channels=channels,
                                sampling=sampling, spacing=spacing,
                                gain=gain)

    m_spk_data = mean_waveforms[:, main_spike] * gain
    ax.plot(time, m_spk_data - main_spike * spacing, lw=2)

    ax = fig.add_subplot(224)
    ax.plot(time, m_spk_data, 'o', label='data', ms=5)

    df_wave = get_wave_spike_info(mean_waveforms[:, main_spike], sampling=sampling, gain=gain)
    line = df_wave.iloc[0]
    # x=df_wave.x.values
    # m_spk=line.m_spk.values
    ax.plot(line.x, line.m_spk, label='interpolated')
    ax.axhline(0, color=ph.almost_black, zorder=-10)
    extremum = [line.arg_max_bef, line.arg_p, line.arg_max_aft]
    ax.plot([line.x[e] for e in extremum],
            [line.m_spk[e] for e in extremum], 'o', ms=10)
    ax.plot([line.x[f] for f in line.fwhm_ind],
            [line.m_spk[f] for f in line.fwhm_ind], '-o', ms=7)

    ax.legend()
    return df_wave


def get_wave_spike_info(spk_waveform, sampling, gain):
    m_spk_data = np.array(spk_waveform, dtype=float) * gain
    time = np.arange(m_spk_data.shape[0]) / sampling * 1e3  # create interpolation function in ms
    from scipy.interpolate import interp1d
    m_spk_func = interp1d(time, m_spk_data, kind='cubic')
    x = np.arange(time[0], time[-1], 1e-3)  # interpolate at the microsecond
    m_spk = m_spk_func(x)
    arg_p = m_spk.argmin()
    ptp = m_spk.ptp()
    half_maximum = ptp / 2 + m_spk[arg_p]

    # some spikes are positive and min can be the last value
    if (arg_p == len(m_spk)-1) or (arg_p == 0):
        # in that case there is not much point in doing anything else
        return pd.DataFrame(dict(m_spk_func=[m_spk_func],
                                 arg_max_bef=[np.nan],
                                 arg_max_aft=[np.nan],
                                 arg_p=[arg_p],
                                 fwhm_ind=[np.nan],
                                 ptp=[ptp],
                                 x=[x],
                                 m_spk=[m_spk],
                                 max_aft=[np.nan],
                                 max_bef=[np.nan],
                                 peak=[m_spk[arg_p]],
                                 fwhm=[np.nan],
                                 fwhm_p=[np.nan],
                                 fwhm_p_ind=[np.nan]),
                            index=[0])

    if arg_p == len(m_spk):
        arg_max_aft = len(m_spk)
        fwhm_1 = [len(m_spk)]
        fwhm_p1 = [len(m_spk)]
    else:
        arg_max_aft = arg_p + m_spk[arg_p:].argmax()
        # Find first point after peak superior to half maximum
        # sup cause negative spikes
        fwhm_1 = arg_p + np.where(m_spk[arg_p:arg_max_aft] > half_maximum)[0]
        # do the same but using peak/2 not ptp
        fwhm_p1 = arg_p + np.where(m_spk[arg_p:arg_max_aft] >
                                   m_spk[arg_p] / 2)[0]  # sup cause negative spikes
    if arg_p == 0:
        arg_max_bef = 0
        fwhm_0 = 0
        fwhm_p0 = 0
    else:
        arg_max_bef = m_spk[:arg_p].argmax()
        # Find last point before peak superior to half maximum
        fwhm_0 = arg_max_bef + np.where(m_spk[arg_max_bef:arg_p] >
                                        half_maximum)[0]  # sup cause negative spikes
        if len(fwhm_0):
            fwhm_0 = fwhm_0[-1]
        else:  # it doesn't cross baseline before peak. Take 0
            fwhm_0 = 0
        # do the same but using peak/2 not ptp
        fwhm_p0 = arg_max_bef + np.where(m_spk[arg_max_bef:arg_p] >
                                         m_spk[arg_p] / 2)[0]
        if len(fwhm_p0):
            fwhm_p0 = fwhm_p0[-1]  # sup cause negative spikes
        else:
            fwhm_p0 =0


    if len(fwhm_1) != 0:
        fwhm_1 = fwhm_1[0]
    else:
         # it doesn't cross the baseline again, the highest point
        fwhm_1 = arg_p + m_spk[arg_p:arg_max_aft].argmax()
    if len(fwhm_p1) != 0:
        fwhm_p1 = fwhm_p1[0]
    else:
        # it doesn't cross the baseline again, the highest point
        fwhm_p1 = arg_p + m_spk[arg_p:arg_max_aft].argmax()

    fwhm = [fwhm_0, fwhm_1]
    fwhm_p = [fwhm_p0, fwhm_p1]
    return pd.DataFrame(dict(m_spk_func=[m_spk_func], arg_max_bef=[arg_max_bef],
                             arg_max_aft=[arg_max_aft], arg_p=[arg_p], fwhm_ind=[fwhm],
                             ptp=[ptp], x=[x], m_spk=[m_spk], max_aft=[m_spk[arg_max_aft]],
                             max_bef=[m_spk[arg_max_bef]], peak=[m_spk[arg_p]],
                             fwhm=[np.diff(x[fwhm])[0]], fwhm_p=[np.diff(x[fwhm_p])[0]],
                             fwhm_p_ind=[fwhm_p]),
                        index=[0])
