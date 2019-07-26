import numpy as np
from scipy.signal import filtfilt
import pyopenephys
import peakutils
from utils import change_date_format, read_analog_binary_signals, butter_bandpass
from scipy import interpolate
import gc


def filter_all_channels(b, a, all_channels_trace):
    num_chans = np.shape(all_channels_trace)[1]
    num_samples = np.shape(all_channels_trace)[0]
    filtered_traces = np.zeros((num_samples, num_chans))
    for channel in range(num_chans):
        channel_trace = all_channels_trace[:, channel]
        filtered_traces[:, channel] = filtfilt(b, a, channel_trace)
    return filtered_traces


def fill_nan(A):
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B


def arti_remove_and_save(mouse, experiment, num_rec_chans=32):
    """Filters the data, detects large amplitude artifacts and interpolates between values either side of artifacts"""
    OEpath = 'W:\\OpenEphys_data\\' + mouse + '\\'
    output_path = OEpath + experiment + '\\' + change_date_format(experiment) + '_no_artifacts.dat'

    # opens the continuous.dat file for that experiment
    cont_file_path = OEpath + experiment + '\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\'
    cont_filename = 'continuous.dat'
    OEfile = pyopenephys.core.File(OEpath + experiment)
    recording = OEfile.experiments[0].recordings[0]
    num_chan = recording.nchan

    with open(cont_file_path + cont_filename, 'rb') as fh:
        [raw_data, num_samples] = read_analog_binary_signals(fh, num_chan)

    b, a = butter_bandpass(300, 6000, 30000)

    no_artifact_all_chans = np.memmap(output_path, np.int16, mode='w+', shape=(num_samples, 32))

    for channel in range(num_rec_chans):
        chan_raw = filtfilt(b, a, raw_data[:, channel])

        peak_indexes = peakutils.indexes(chan_raw, thres=5000, min_dist=10, thres_abs=True)
        chan_artifacts_removed = np.copy(chan_raw)
        arti_window_start = peak_indexes - 80
        arti_window_end = peak_indexes + 80

        for peak in range(peak_indexes.shape[0]):
            chan_artifacts_removed[arti_window_start[peak]:arti_window_end[peak]] = np.nan

        interpolated_trace = fill_nan(chan_artifacts_removed)
        no_artifact_all_chans[:, channel] = interpolated_trace
        gc.collect()
        print(channel, 'has been processed')

    print('done')

