profile = dict(Antonin_thalrec=dict(process=101,  # OE process to use for convert2dat
                                    target_path='/mnt/ssd/Antonin/temp_clustering/datfiles/',
                                    # where to write dat files
                                    source_path='/mnt/ssd/Antonin/temp_clustering/OpenEphysData/',
                                    # where are the OE files
                                    num_chans=None,  # used only if num_shanks is an integer
                                    ref=None,
                                    # change to 'CMR' or 'CAR' if you want to use a reference before converting
                                    detekt_path='/mnt/ssd/Antonin/temp_clustering/datfiles/',
                                    # path for dat files used
                                    # to detect spikes. Is usually target_path if you don't move files around. Will be
                                    # hard coded in the prm file (should I change that??)
                                    num_shanks=dict(v1_probe=range(32),
                                                    tet1=range(32, 32 + 4),
                                                    tet2=range(32 + 4, 32 + 8)),
                                    # nums_shanks is an int or a dict. If it is an int, the num_chans channels are split
                                    # in num_shanks equal parts and writen in different files (useful to split shanks or
                                    # tetrodes). If it is a dict, then keys are a identifier, that will be append to the
                                    # file name and values are the ORDERED list of channels to write in this file
                                    probe_files=dict(v1_probe='crisNiell1Shank',
                                                     tet1='tetrode',
                                                     tet2='tetrode')),

               # probe_file to use for each shank. Should be a dict with dat file identifier as
               # keys and probe name as value. Probe file should be copied to the relevant
               # klusta/phy folder (klusta/probes/ or phy/electrode/probes/)

               Francois=dict(process=108,  # for convert2dat
                             target_path='/home/blota/Basel/Cerbellum/datfiles/',
                             source_path='/home/blota/Basel/Cerbellum/OE_file/',
                             num_chans=64,
                             ref=None,
                             # for detekt spikes
                             #  detekt_path='/mnt/microscopy/Data/Antonin/LP_project/thal_rec/datfiles/'),
                             detekt_path='/home/blota/Basel/Cerbellum/datfiles/',
                             num_shanks=dict(tet16=range(64 - 4, 64),
                                             tet15=range(64 - 8, 64 - 4),
                                             ),
                             probe_files=dict(tet16='tetrode',
                                              tet15='tetrode', ), ),
               Morgane=dict(process=103,
                            target_path='/home/blota/microscopy/Data/Antonin/Morgane/datfiles',
                            source_path='/home/blota/microscopy/Data/Antonin/Morgane/OpenEphysData',
                            num_chans=64,
                            ref=None,
                            # for detekt spikes
                            #  detekt_path='/mnt/microscopy/Data/Antonin/LP_project/thal_rec/datfiles/'),
                            detekt_path='/home/blota/microscopy/Data/Antonin/Morgane/datfiles',
                            num_shanks=dict(shk0=range(0, 32),
                                            shk1=range(32, 64),
                                            ),
                            probe_files=dict(shk0='kellyClancy1Shank',
                                             shk1='kellyClancy1Shank', ), )

               )
