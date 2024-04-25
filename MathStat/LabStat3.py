import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import bootstrap


def binom_estimate(data: np.ndarray, name: str):
    n = len(data)
    dataB = (data,)
    ci_mean = bootstrap(dataB, np.mean, confidence_level=0.95, method='percentile').confidence_interval
    ci_std = bootstrap(dataB, np.std, confidence_level=0.95, method='percentile').confidence_interval
    ci_var = bootstrap(dataB, np.var, confidence_level=0.95, method='percentile').confidence_interval

    # https://kpfu.ru/portal/docs/F1799623674/158_2_phys_mat_6.pdf Метод моментов
    xn = np.sum(data) / n
    sample_variance = np.sum((data - xn) ** 2) / (n - 1)  # несмещенная оценка дисперсии
    pn = (xn - sample_variance) / xn  # Оценка p^
    nn = max(xn ** 2 / (xn - sample_variance), data.max())  # Оценка n^

    if pn < 0:
        pn = xn / nn

    ci_p = (ci_mean.low / nn, ci_mean.high / nn)
    ci_n = (ci_mean.low / pn, ci_mean.high / pn)

    print(name)
    print(f'mean: {ci_mean}')
    print(f'std: {ci_std}')
    print(f'var: {ci_var}')
    print(f'p: {ci_p}')
    print(f'n: {ci_n}')
    print(f'mle p: {mle_binomial(data)}')
    print()


def norm_estimate(data: np.ndarray, name: str, true_mean: float, true_std: float):
    # n = len(data)
    dataB = (data,)
    ci_mean = bootstrap(dataB, np.mean, confidence_level=0.95, method='percentile').confidence_interval
    ci_std = bootstrap(dataB, np.std, confidence_level=0.95, method='percentile').confidence_interval
    ci_var = bootstrap(dataB, np.var, confidence_level=0.95, method='percentile').confidence_interval

    sample_mean = np.mean(data)

    data_min = max(min(data), true_mean - true_std * 3)
    data_max = min(max(data), true_mean + true_std * 3)

    x = np.linspace(data_min, data_max, 1000)
    y = norm.pdf(x, true_mean, true_std)

    mle_mean, mle_std = mle_normal(data)

    print(name)
    print(f'mean: {ci_mean}')
    print(f'std: {ci_std}')
    print(f'var: {ci_var}')
    print(f'mle mean: {mle_mean}')
    print(f'mle std: {mle_std}')
    print()

    plt.plot(x, y, label='Нормальное распределение (плотность вероятности)')
    plt.axvline(x=true_mean, color='red', label='Истинное матожидание')
    plt.axvline(x=sample_mean, color='green', linestyle='--', label='Выборочное среднее')
    plt.axvspan(ci_mean.low, ci_mean.high, color='purple', alpha=0.5, label='Границы доверительного интервала')
    plt.xlim(data_min, data_max)
    plt.title(f'{name}')
    plt.legend(loc='upper right')
    plt.show()


def mle_binomial(data: np.array, true_n: int) -> float:
    total_successes = sum(data)
    total_trials = len(data) * true_n
    p_estimate = total_successes / total_trials
    return p_estimate


def mle_normal(data):
    mean_estimate = sum(data) / len(data)

    variance_estimate = sum((x - mean_estimate) ** 2 for x in data) / len(data)
    std_dev_estimate = variance_estimate ** 0.5

    return mean_estimate, std_dev_estimate


if __name__ == '__main__':
    n = 65
    p = 0.43

    true_mean = 3.5
    true_std = 2.7

    binom_distributions = {
        'binom100': binom(p=p, n=n).rvs(size=100),
        'binom1000': binom(p=p, n=n).rvs(size=1000)
    }

    norm_distributions = {
        'norm100': norm.rvs(true_mean, true_std, size=100),
        'norm1000': norm.rvs(true_mean, true_std, size=1000)
    }

    for name, data in binom_distributions.items():
        binom_estimate(data, name)

    for name, data in norm_distributions.items():
        norm_estimate(data, name, true_mean, true_std)
