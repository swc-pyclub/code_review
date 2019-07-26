"""Used to generate a dat file out of the open ephys data"""

import os
import gc
import h5py
import numpy as np
from ..data_preprocessing.referencing import create_ref
from ..utils import make_list


def oe2dat(path2kwd, chan2save=range(64), force=False, dtype='int16',
           outname=None, path2save=None, chunksize=20 * 30000):
    """Save a dat file out of a oe file.

    Quick and dirty. Concatenate the recordings

    """
    chan2save = make_list(chan2save, int)
    # if ref_chans is None:
    #     ref_chans = chan2save

    p, fn = os.path.split(path2kwd)
    if outname is None:
        outname = 'data'
    if path2save is None:
        path2save = p
    # if concat:
    #     outfile = os.path.join(path2save, '%s_%s_concat.dat'%(outname, analogsignal_name))
    #     if os.path.isfile(outfile):
    #         if not force:
    #             print 'File %s already exists'%outfile
    #             return
    #         else:
    #             print "Replacing %s"%outfile
    #         f=open(outfile, 'w')
    #         f.close() # erase file
    outfile = os.path.join(path2save, '%s_%s.dat' % (outname, fn.split('.')[0]))
    if os.path.isfile(outfile):
        if not force:
            print('File %s already exists' % outfile)
            print('skiping')
            return
        else:
            print("Replacing %s" % outfile)
            f = open(outfile, 'w')
            f.close()  # erase file

    with h5py.File(path2kwd, mode='r') as h5file:
        recs = h5file.get('/recordings/')
        recs = sorted(recs)
        for rec in recs:
            print('Saving rec %s' % rec)
            data = h5file.get('/recordings/%s/data' % rec)
            write_dat(outfile, data, chan2save, dtype, chunksize)


def write_dat(target_file, data, chan2save, first_sample=None,
              last_sample=None, dtype='int16', chunksize=20 * 30000,
              funcpy=None):
    """Write data to a dat file

    funcpy can be either a python function or a dictionary of function. If
    it is a function, funcpy(data) is saved in target_file, if it is a
    dictionary, keys are prefix and values are functions, funcpy[key](data) is
    saved in key_target_file
    """

    chan2save = make_list(chan2save, int)

    if first_sample is None:
        n = 0
    else:
        assert isinstance(first_sample, int)
        n = first_sample

    npts = data.shape[0]
    if last_sample is not None:
        assert (isinstance(last_sample, int) and
                (last_sample <= npts) and
                (last_sample > n))
        npts = last_sample

    # widgets = ['converting: ', pgb.Percentage(), ' ', pgb.Bar(marker=pgb.RotatingMarker()),
    #            ' ', pgb.ETA()]
    # pbar = pgb.ProgressBar(widgets=widgets, maxval=npts + 1).start()
    while n < npts:
        end = min(n + chunksize, npts)

        if funcpy is None:
            d = np.array([data[n:end, i] for i in chan2save], dtype=dtype).T
            with open(target_file, "ab") as f:
                d.tofile(f)
        elif isinstance(funcpy, dict):
            data_chunck = np.array([data[n:end, i] for i in chan2save], dtype=dtype)
            for k, v in funcpy.items():
                d = v(data_chunck).T
                if k:
                    dir, name = os.path.split(target_file)
                    name = os.path.join(dir, k + '_' + name)
                else:
                    name = target_file
                with open(name, "ab") as f:
                    d.tofile(f)
        else:
            d = np.array([data[n:end, i] for i in chan2save], dtype=dtype)
            d = funcpy(d).T
            with open(target_file, "ab") as f:
                d.tofile(f)

        n = end
        # pbar.update(n+1)
    del data
    # pbar.finish()
    gc.collect()
