import numpy as np
import scipy.stats as st


def confidence_interval(data, confidence=0.95):
    """
    Calculates the mean and confidence interval for a given dataset.

    Args:
        data (list or numpy.ndarray): The dataset.
        confidence (float, optional): The confidence level (e.g., 0.95 for 95%). Defaults to 0.95.

    Returns:
        tuple: A tuple containing the mean, lower bound of the confidence interval, and upper bound.
    """
    a = 1.0 * data.numpy()
    n = len(a)
    m = np.mean(a).item()
    se = st.sem(a)
    h = (se * st.t.ppf((1 + confidence) / 2., n-1)).item()
    return m, h, m-h, m+h

def bootstrap_ci_2d(data, num_resamples=10000, confidence=0.95, axis=0, seed=None):
    """
    Calculates the mean and confidence interval for a given dataset.
    """
    if seed is not None:
        np.random.seed(seed)

    data = data.numpy()
    n = data.shape[axis]
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    # Move resampling axis to 0 for simplicity
    data = np.swapaxes(data, axis, 0)
    resampled_means = []

    for _ in range(num_resamples):
        resample_idx = np.random.choice(n, size=n, replace=True)
        sample = data[resample_idx]
        resampled_means.append(sample.mean(axis=0))  # mean across resamples

    resampled_means = np.stack(resampled_means)
    lower = np.percentile(resampled_means, lower_percentile, axis=0)
    upper = np.percentile(resampled_means, upper_percentile, axis=0)
    original_mean = data.mean(axis=0)
    margin = (upper - lower) / 2

    return original_mean, margin, lower, upper


def diff_from_margins(mean1, moe1, mean2, moe2, scale=1, print_output=False):
    diff = mean1 - mean2
    moe_diff = np.sqrt(moe1**2 + moe2**2)
    lower = diff - moe_diff
    upper = diff + moe_diff
    if print_output:
        print(f'\t diff_from_margins')
        print(f'\t mean: {mean1:.4f}-{mean2:.4f} = {diff:.4f}')
        print(f'\t dev: {moe1:.4f}+{moe2:.4f} = {moe_diff:.4f}')
        print(f'\t min: {scale*(lower):.4f}')
        print(f'\t max: {scale*(upper):.4f}')
        print()
    return diff, moe_diff, lower, upper