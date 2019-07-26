__author__ = 'blota'

from . import cross_threshold
import numpy as np
import pandas as pd


def find_pulses(signal, sampling, pointinterval=None, sample_zero=0,
                threshold=None, force=False, baseline_high=False):
    """Use to find pulses in a analog signal

    Return a dataframe with pulse beg, end and amplitude (use trace
    median as baseline if threshold is None)

    If sample_zero is not none, shift detection by that amount

    """
    if pointinterval is None:
        pointinterval = int(sampling / 100.)  # 10 ms
    if baseline_high:
        signal = -1*np.array(signal) # starting from with, so revert the signal.
    if threshold is None:
        threshold = np.median(signal) + signal.std()
    cross_up = np.asarray(cross_threshold(signal, threshold=threshold,
                               maximum=False, rise=True,
                                          pointinterval=pointinterval),
                          dtype=int)
    cross_down = np.asarray(cross_threshold(signal, threshold=threshold,
                                 maximum=True, rise=False,
                                 pointinterval=pointinterval),
                            dtype=int)
    if not force:
        assert len(cross_up) == len(cross_down)
    Np = min(len(cross_down), len(cross_up))
    if not force:
        assert all((cross_down - cross_up) > 0)
    ledbase = np.median(signal)
    pulse_amp = [signal[b:e].mean() - ledbase for b, e in
                 zip(cross_up[:Np], cross_down[:Np])]

    assert isinstance(sample_zero, int)
    cross_down += sample_zero
    cross_up +=sample_zero
    df = pd.DataFrame(index=np.arange(Np))
    df['beg_pulse_sample'] = np.array(cross_up[:Np], dtype=int)
    df['end_pulse_sample'] = np.array(cross_down[:Np], dtype=int)
    df['beg_pulse_time'] = cross_up[:Np] / sampling
    df['end_pulse_time'] = cross_down[:Np] / sampling
    df['pulse_amp'] = pulse_amp
    df['amp'] = np.round(df['pulse_amp'], 1)
    return df


def make_pulse_from_ttl(ttl_df, t0=None, sampling=None):
    """Take a ttl df as produced by oe_reader and make a dataframe shaped
    like find_pulses (with beg and end)

    :param ttl_df:
    :return:

    """

    on_times = ttl_df[ttl_df.eventID == 1].time_samples.values
    off_times = ttl_df[ttl_df.eventID == 0].time_samples.values
    off_times_ind = off_times.searchsorted(on_times + 1)
    bad_times = off_times_ind == len(off_times)
    if any(bad_times):
        print('Found %i/%i bad times, removes pulses' %
              (np.sum(bad_times), len(bad_times)))
    on_times = on_times[np.logical_not(bad_times)]
    off_times_ind = off_times_ind[np.logical_not(bad_times)]
    off_times = off_times[off_times_ind]

    if t0 is None:
        t0 = 0
    out_df = pd.DataFrame(dict(beg_pulse_sample=on_times - t0,
                               end_pulse_sample=off_times - t0
                               ))
    if sampling is not None:
        out_df['beg_pulse_time'] = out_df.beg_pulse_sample.values / float(sampling)
        out_df['end_pulse_time'] = out_df.end_pulse_sample.values / float(sampling)
    return out_df
