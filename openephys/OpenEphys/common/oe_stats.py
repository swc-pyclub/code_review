import numpy as np


def significant_bins(bin_heights, n_bins, first_bin_tested, baseline,
                     alpha=.05):
    """Find bins significantly different from baseline

    bin_heights: a list of values (typicaly output of np.histogram)
    n_bins: the number of bins to test (size of the response area)
    first_bin_tested: index of the first bin of the response area
    baseline: if integer, will take n_bins around that value,
              else: must be a tuple delimiting the baseline
    alpha: alpha for significance (see code for details)"""
    d = {}  # output dictionary

    # Find the baseline
    if isinstance(baseline, int):
        b_start = int(baseline - n_bins / 2)
        b_stop = int(baseline + n_bins / 2)
    else:
        assert [0 < b < len(bin_heights) for b in baseline[:2]]
        b_start = baseline[0]
        b_stop = baseline[1]

    # Compute how many sds we need to take. We assume a normal distribution
    # First get how many sd corresponds to that alpha:
    # Find the adapted alpha for testing multiple bins:
    a_alpha = 1 - (alpha / n_bins / 2)  # divide by 2 for two tailed
    from scipy.stats import norm
    d['n_sd'] = norm.ppf(a_alpha)

    baseline = np.array(bin_heights[b_start:b_stop])
    d['baseline_sd'] = baseline.std()
    d['baseline_m'] = baseline.mean()
    d['baseline_n_pts'] = baseline.sum()
    resp = np.array(bin_heights[first_bin_tested:first_bin_tested + n_bins])
    residuals = resp - d['baseline_m']
    d['sig_pos'] = residuals > d['baseline_sd'] * d['n_sd']
    d['sig_neg'] = residuals < -(d['baseline_sd'] * d['n_sd'])
    d['resp_m'] = resp.mean()
    d['resp_sd'] = resp.std()
    d['resp_n_pts'] = resp.sum()
    d['resp_peak'] = resp.max()
    d['pos_mod'] = any(d['sig_pos'])
    d['neg_mod'] = any(d['sig_neg'])
    return d


def r_wilcox(array, array2=None, paired=False):
    """Just a small function to remind me how to do a paired wilcoxon test using
    rpy2.

    Get the p value like that: wilcox_result[2][0]"""
    import rpy2
    import rpy2.robjects
    if array2 is None:
        wilcox_result = rpy2.robjects.r['wilcox.test'](rpy2.robjects.FloatVector(array))
    else:
        wilcox_result = rpy2.robjects.r['wilcox.test'](rpy2.robjects.FloatVector(array),
                                                       rpy2.robjects.FloatVector(array2),
                                                       paired=paired)
    return wilcox_result
