# OE process to use for convert2dat
process = 101

# where to write dat files
target_path = '/mnt/ssd/Antonin/temp_clustering/datfiles/'

# where are the OE files
source_path = '/mnt/ssd/Antonin/temp_clustering/OpenEphysData/'

# Total number of channels used only if num_shanks is an integer
num_chans = None

# Reference the data before covnerting? change to 'CMR' or 'CAR'
# if you want to use a common (M)edian or (A)verage reference before converting
ref = None

# path for dat files used
# to detect spikes. Is usually target_path if you don't move files around. Will be
# hard coded in the prm file (should I change that??)
detekt_path = '/mnt/ssd/Antonin/temp_clustering/datfiles/'

# Path to a template param file to use for klusta/phy. If None will use that in OpenEphys.oe_clustering
param_template = None

# nums_shanks is an int or a dict. If it is an int, the num_chans channels are split
# in num_shanks equal parts and writen in different files (useful to split shanks or
# tetrodes). If it is a dict, then keys are a identifier, that will be append to the
# file name and values are the ORDERED list of channels to write in this file

num_shanks = dict(v1_probe=range(32),
                  tet1=range(32, 32 + 4),
                  tet2=range(32 + 4, 32 + 8)),

# a dictionary with  keys equal to shank IDs and values to probe files
probe_files = dict(v1_probe='crisNiell1Shank',
                   tet1='tetrode',
                   tet2='tetrode')
