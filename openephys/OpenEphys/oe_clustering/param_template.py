experiment_name = None,  # will be updated by script
prb_file = None,  # will be updated by script

traces = dict(
    raw_data_files=None,  # will be updated by script
    voltage_gain=1.94999993e-01,
    sample_rate=30000.,
    n_channels=None,  # will be updated by script
    dtype='int16',
)

spikedetekt = {
    'filter_low': 600.,
    'filter_high_factor': 6000. / 30000.,  # will be multiplied by the sample rate
    'filter_butter_order': 4,

    # Data chunks.
    'chunk_size_seconds': 1.,
    'chunk_overlap_seconds': .015,

    # Threshold.
    'n_excerpts': 50,
    'excerpt_size_seconds': 1.,
    'use_single_threshold': True,
    'threshold_strong_std_factor': 6.5,
    'threshold_weak_std_factor': 3.5,
    'detect_spikes': 'negative',

    # Connected components.
    'connected_component_join_size': 1,

    # Spike extractions.
    'extract_s_before': 15,
    'extract_s_after': 15,
    'weight_power': 4,

    # Features.
    'n_features_per_channel': 3,
    'pca_n_waveforms_max': 10000,

}
klustakwik2 = {'max_quick_step_candidates': 100000000,
               'always_split_bimodal': False, 'num_starting_clusters': 500,
               'prior_point': 1, 'max_possible_clusters': 1000, 'fast_split': False,
               'max_quick_step_candidates_fraction': 0.4, 'max_iterations': 1000,
               'break_fraction': 0.0, 'split_first': 20, 'noise_point': 1,
               'num_changed_threshold': 0.05, 'penalty_k_log_n': 1.0,
               'full_step_every': 1, 'mua_point': 2, 'use_noise_cluster': True,
               'split_every': 40, 'subset_break_fraction': 0.01, 'dist_thresh':
                   9.210340371976184, 'use_mua_cluster': True,
               'consider_cluster_deletion': True, 'points_for_cluster_mask': 100,
               'penalty_k': 0.0, 'num_cpus': 8}
