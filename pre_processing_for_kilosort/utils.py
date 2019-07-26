from datetime import datetime
from scipy.signal import butter
import numpy as np
import os

def read_analog_binary_signals(filehandle, numchan):
    numchan=int(numchan)
    nsamples = os.fstat(filehandle.fileno()).st_size // (numchan*2)
    print('Estimated samples: ', int(nsamples), ' Numchan: ', numchan)

    samples = np.memmap(filehandle, np.int16, mode='r',
                        shape=(nsamples, numchan),  order='C')
    return samples, nsamples

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def change_date_format(experiment_name):
    date_of_exp = experiment_name.split('__')[1]
    old_format = date_of_exp.split('_')[0]
    stripped_format = datetime.strptime(old_format, '%Y-%m-%d')
    new_format = stripped_format.strftime('%Y%m%d')
    return new_format