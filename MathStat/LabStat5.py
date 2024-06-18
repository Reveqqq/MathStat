from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def one_sided_test(alpha, mu0, xbar, std, n):
    """
        Односторонний ассимптотический тест
        --------------------------------------
        alpha - вероятность ошибки первого рода (уровень значимости)
        mu0 - ожидаемое среднее
        xbar - выборочное среднее Sn
        std - среднеквадратичное отклонение
        n - размер выборки
    """
    z = np.sqrt(n) * (xbar - mu0) / std
    p_value = 1 - norm.cdf(z)
    z_crit = norm.ppf(1 - alpha)

    fig, ax = plt.subplots(figsize=(10, 1))
    x = np.linspace(z - n / 4, z + n / 4, 1000)
    y = norm.pdf(x)
    ax.plot(x, y)
    ax.axvline(z, color='red', label=f'Z={z:.2f}')
    ax.axvline(z_crit, color='blue', label=f'Z_crit={z_crit:.2f}')
    ax.fill_between(x, 0, y, where=(x >= z_crit), alpha=0.5, color='red')
    ax.legend()
    plt.show()

    print(f"p-value: {p_value:.4f}")
    if p_value < alpha:
        print("Нулевая гипотеза отвергается")
    else:
        print("Нулевая гипотеза не отвергается")


def two_sided_test(alpha, mu0, xbar, std, n):
    """
            Двусторонний ассимптотический тест
            --------------------------------------
            alpha - вероятность ошибки первого рода (уровень значимости)
            mu0 - ожидаемое среднее
            xbar - выборочное среднее Sn
            std - среднеквадратичное отклонение
            n - размер выборки
        """
    z = np.sqrt(n) * (xbar - mu0) / std
    p_value = 2 * (1 - norm.cdf(abs(z)))
    z_crit = norm.ppf(1 - alpha / 2)

    fig, ax = plt.subplots(figsize=(10, 1))
    x = np.linspace(z - n / 4, z + n / 4, 1000)
    y = norm.pdf(x)
    ax.plot(x, y)
    ax.axvline(z, color='red', label=f'Z={z:.2f}')
    ax.axvline(z_crit, color='blue', label=f'Z_crit={z_crit:.2f}')
    ax.axvline(-z_crit, color='blue')
    ax.fill_between(x, 0, y, where=(x <= -z_crit) | (x >= z_crit), alpha=0.5, color='red')
    ax.legend()
    plt.show()

    print(f"p-value: {p_value:.4f}")
    if p_value < alpha:
        print("Нулевая гипотеза отвергается")
    else:
        print("Нулевая гипотеза не отвергается")


# one_sided_test(n=64, alpha=0.01, mu0=130, xbar=136.5, std=40)
# two_sided_test(n=100, alpha=0.05, mu0=21, xbar=25.66, std=4.6)
# two_sided_test(n=64, alpha=0.01, mu0=130, xbar=136.5, std=40)
one_sided_test(n=121, alpha=0.01, mu0=0.5, xbar=0.53, std=0.11)

