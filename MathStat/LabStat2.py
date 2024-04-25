import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF


def ecdf_custom(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


def conf_interval(data, alpha=0.5):
    n = len(data)
    epsilon = np.sqrt(np.log(2. / alpha) / (2 * n))
    x, y = ecdf_custom(data)
    lower = np.clip(y - epsilon, 0, 1)
    upper = np.clip(y + epsilon, 0, 1)
    return x, lower, upper


def plot_distribution(name, data):
    cdf = None
    x = np.linspace(min(data), max(data), len(data))
    res = stats.ecdf(data)
    low, high = res.cdf.confidence_interval(0.95)
    if 'uniform' in name:
        cdf = stats.uniform.cdf(x, 0, 1)
    elif 'bernoulli' in name:
        cdf = stats.bernoulli.cdf(x, 0.5)
    elif 'binomial' in name:
        cdf = stats.binom.cdf(x, 10, 0.5)
    elif 'normal' in name:
        cdf = stats.norm.cdf(x, 0, 1)
    ecdf = ECDF(data)
    x_ecdf_custom, y_ecdf_custom = ecdf_custom(data)
    x_ci, lower, upper = conf_interval(data)

    plt.plot(x, cdf, label='Theoretical CDF')
    plt.plot(x_ecdf_custom, y_ecdf_custom, label='Empirical CDF (Custom)')
    plt.step(ecdf.x, ecdf.y, label='Empirical CDF (statsmodels)')
    plt.fill_between(x_ci, lower, upper, color='purple', alpha=0.5, label='Границы доверительного интервала')
    plt.plot(low.quantiles, low.probabilities, label='Lower bound', color='blue')
    plt.plot(high.quantiles, high.probabilities, label='Upper bound', color='blue')
    # plt.plot(x_ci, lower, ls='--', color='purple', label='Lower CI')
    # plt.plot(x_ci, upper, ls='--', color='purple', label='Upper CI')
    plt.title(f"{name.capitalize()} Distribution")
    plt.legend()
    plt.show()


distributions = {
    'uniform100': stats.uniform.rvs(0, 1, size=100),
    'uniform1000': stats.uniform.rvs(0, 1, size=1000),
    'bernoulli100': stats.bernoulli.rvs(0.5, size=100),
    'bernoulli1000': stats.bernoulli.rvs(0.5, size=1000),
    'binomial100': stats.binom.rvs(10, 0.5, size=100),
    'binomial1000': stats.binom.rvs(10, 0.5, size=1000),
    'normal100': stats.norm.rvs(0, 1, size=100),
    'normal1000': stats.norm.rvs(0, 1, size=1000),
}

for name, data in distributions.items():
    plot_distribution(name, data)
