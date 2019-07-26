from OpenEphys.data_preprocessing import chunked_referencing as ch_ref
from OpenEphys.io import openephys2dat as OE2dat
import pyopenephys
import numpy as np
import pickle
import os.path as op
import os
import gc
from utils import change_date_format, read_analog_binary_signals, butter_bandpass
from scipy.signal import filtfilt


def cav_and_save(mouse, experiment, num_rec_chans=32):
    """Performs common average referencing (using median) on data after removing large amplitude artifacts"""
    OEpath = 'W:\\OpenEphys_data\\' + mouse + '\\'
    arti_sub_path = OEpath + experiment + '\\' + change_date_format(experiment) + '_no_artifacts.dat'

    # where the final binary will be saved out ready for kilosort to be run on it
    kilosort_folder = 'W:\\Tetrode_kilosort\\'
    kilosort_mouse_path = kilosort_folder + '\\' + mouse + '\\' + mouse + '_' + change_date_format(
        experiment) + '_arti_sub' + '\\'
    kilosort_filename = mouse + '_' + change_date_format(experiment) + '_arti_sub_mmap.dat'

    if not op.exists(kilosort_mouse_path):
        os.makedirs(kilosort_mouse_path)

    OEpath = 'W:\\OpenEphys_data\\' + mouse + '\\'
    ref_memmap = kilosort_mouse_path + kilosort_filename
    ref_output_file = OEpath + experiment + '\\reference_that_was_subtracted.pkl'

    # opens the continuous.dat file for that experiment
    with open(arti_sub_path, 'rb') as fh:
        [raw_data, num_samples] = read_analog_binary_signals(fh, num_rec_chans)

    channels = range(num_rec_chans)

    # calculates the reference signal and saves it out
    reference_to_sub = ch_ref.create_chunked_ref(raw_data, channels, data_chunks=25000000,
                                                 multi_process_chunks=7000000, verbose=True)

    with open(ref_output_file, 'wb') as f:
        pickle.dump([reference_to_sub], f)
    print(reference_to_sub.shape)
    gc.collect()

    # performs the subtraction of the reference from the raw data
    referenced_chans = np.memmap(ref_memmap, np.int16, mode='w+',
                                 shape=(reference_to_sub.shape[0], num_rec_chans))
    for chan in range(num_rec_chans):
        referenced_chans[:, chan] = raw_data[:, chan] - reference_to_sub
        print(chan, 'has been cav-ed')

    gc.collect()

    OE2dat.write_dat(kilosort_mouse_path + kilosort_filename, referenced_chans, range(num_rec_chans),
                      chunksize=20000000)

def cav_and_save_no_arti_sub(mouse, experiment, num_rec_chans=32):
    """Performs common average referencing (using median) on data without removing large amplitude artifacts"""
    OEpath = 'W:\\OpenEphys_data\\' + mouse + '\\'

    # opens the continuous.dat file for that experiment
    cont_file_path = OEpath + experiment + '\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\'
    cont_filename = 'continuous.dat'
    OEfile = pyopenephys.core.File(OEpath + experiment)
    recording = OEfile.experiments[0].recordings[0]
    num_chan = recording.nchan

    with open(cont_file_path + cont_filename, 'rb') as fh:
        [raw_data, num_samples] = read_analog_binary_signals(fh, num_chan)

    # where the final binary will be saved out
    kilosort_folder = 'W:\\Tetrode_kilosort\\'
    kilosort_mouse_path = kilosort_folder + '\\' + mouse + '\\' + mouse + '_' + change_date_format(
        experiment) + '_no_arti_sub' + '\\'
    kilosort_filename = mouse + '_' + change_date_format(experiment) + '_mmap.dat'

    if not op.exists(kilosort_mouse_path):
        os.makedirs(kilosort_mouse_path)

    OEpath = 'W:\\OpenEphys_data\\' + mouse + '\\'
    ref_memmap = kilosort_mouse_path + kilosort_filename
    filtered_memmap =  'W:\\temp_memmap\\' + 'temp_ref2.dat'
    ref_output_file = OEpath + experiment + '\\reference_that_was_subtracted.pkl'
    channels = range(num_rec_chans)

    # bandpass filter
    b, a = butter_bandpass(300, 6000, 30000)

    filt_chans = np.memmap(filtered_memmap, np.int16, mode='w+',
                          shape=(np.shape(raw_data)[0], num_rec_chans))
    for channel in range(num_rec_chans):
        filt_chans[:, channel] = filtfilt(b, a, raw_data[:, channel])
        gc.collect()
        print(channel, 'has been filtered')

    # calculates the reference signal and saves it out
    reference_to_sub = ch_ref.create_chunked_ref(filt_chans, channels, data_chunks=25000000,
                                                 multi_process_chunks=7000000, verbose=True)

    with open(ref_output_file, 'wb') as f:
        pickle.dump([reference_to_sub], f)
    print(reference_to_sub.shape)
    gc.collect()

    # performs the subtraction of the reference from the raw data
    referenced_chans = np.memmap(ref_memmap, np.int16, mode='w+',
                                 shape=(reference_to_sub.shape[0], num_rec_chans))
    for chan in range(num_rec_chans):
        referenced_chans[:, chan] = filt_chans[:, chan] - reference_to_sub
        print(chan, 'has been cav-ed')

    gc.collect()

    # I only need to save the file back in the original format using OE2dat if I'm using kilosort for spike sorting.
    # I only remove artifacts if I'm using kilosort, so if I don't remove artifacts I don't need to call OE2dat



