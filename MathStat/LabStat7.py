import numpy as np
import math
from scipy.stats import ks_2samp, kstwobign, kstest, shapiro, norm, uniform, binom


def kolmogorov_smirnov_one_sample(data, cdf, args=()):
    n = len(data)
    sorted_data = np.sort(data)

    ecdf = np.arange(1, n + 1) / n

    cdf_values = np.array([cdf(x, *args) for x in sorted_data])

    d_statistic = np.max(np.abs(ecdf - cdf_values))

    def p_value_ks(D_n, n):
        ks_sum = 0.0
        for k in range(1, 101):
            term = (-1) ** (k - 1) * math.exp(-2 * (k * D_n * np.sqrt(n)) ** 2)
            ks_sum += term
        p_value = 1 - 2 * ks_sum
        return p_value

    p_value1 = p_value_ks(d_statistic, n)
    p_value2 = kstwobign.sf(d_statistic * np.sqrt(n))
    return d_statistic, p_value1, p_value2


def kolmogorov_smirnov_two_samples(sample1, sample2):
    data1 = np.sort(sample1)
    data2 = np.sort(sample2)
    n = len(data1)
    m = len(data2)

    all_values = np.sort(np.concatenate([data1, data2]))

    ecdf1 = np.searchsorted(data1, all_values, side='right') / len(data1)
    ecdf2 = np.searchsorted(data2, all_values, side='right') / len(data2)

    d_statistic = np.max(np.abs(ecdf1 - ecdf2))
    u = np.sqrt(n * m / (n + m)) * d_statistic
    p_value = 1 - kstwobign.cdf(u)

    return d_statistic, p_value


def normality_tests(samples):
    results = {}
    for name, sample in samples.items():
        ks_stat, ks_p_value = kstest(sample, 'norm', args=(sample.mean(), sample.std(ddof=1)))
        shapiro_stat, shapiro_p_value = shapiro(sample)
        results[f'{name}'] = {
            'Kolmogorov-Smirnov': (ks_stat, ks_p_value),
            'Shapiro-Wilk': (shapiro_stat, shapiro_p_value)
        }
    return results


sample_norm1 = norm.rvs(loc=0, scale=1, size=100)
sample_norm2 = norm.rvs(loc=2, scale=4, size=100)

sample_unif1 = uniform.rvs(loc=0, scale=1, size=100)
sample_unif2 = uniform.rvs(loc=0, scale=1, size=100)

sample_binom = binom.rvs(n=20, p=0.4, size=100)

d_stat_manual, p_value_manual = kolmogorov_smirnov_two_samples(sample_unif1, sample_unif2)
d_stat_builtin, p_value_builtin = ks_2samp(sample_unif1, sample_unif2)

# print(f"Manual KS statistic: {d_stat_manual, p_value_manual}")
# print(f"Builtin KS statistic: {d_stat_builtin, p_value_builtin}")
print(kolmogorov_smirnov_one_sample(sample_norm1, norm.cdf, args=(sample_norm1.mean(), sample_norm1.std(ddof=1))))
# print(kolmogorov_smirnov_one_sample(sample_norm2, norm.cdf, args=(2, 4)))
print()

samples = {
    'norm1': sample_norm1,
    'norm2': sample_norm2,
    'unif1': sample_unif1,
    'unif2': sample_unif2,
    'binom': sample_binom
}

normality_results = normality_tests(samples)

for sample_name, test_results in normality_results.items():
    print(f"\nResults for {sample_name}:")
    for test_name, (stat, p_value) in test_results.items():
        print(f"{test_name} test: statistic = {stat:.4f}, p-value = {p_value:.4f}")
