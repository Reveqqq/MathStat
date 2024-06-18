import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def z_test(n1, n2, mean1, mean2, var1, var2, alpha, type):
    pooled_se = np.sqrt(var1 / n1 + var2 / n2)
    z = (mean1 - mean2) / pooled_se

    if type == 0:  # двусторонний тест
        critical_z1 = stats.norm.ppf(alpha / 2)
        critical_z2 = stats.norm.ppf(1 - alpha / 2)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif type == 1:  # правосторонний  тест H0: mean1 >= mean 2 H1: mean1 < mean2
        critical_z1 = stats.norm.ppf(1 - alpha)
        critical_z2 = float('inf')
        p_value = 1 - stats.norm.cdf(z)
    else:  # левосторонний тест
        critical_z1 = float('-inf')
        critical_z2 = stats.norm.ppf(alpha)
        p_value = stats.norm.cdf(z)

    if z < critical_z1 or z > critical_z2:
        verdict = "отвергается"
    else:
        verdict = "принимается"

    z_values = np.linspace(z - n1 / 8, z + n1 / 8, 300)
    plt.figure(figsize=(10, 5))
    plt.plot(z_values, stats.norm.pdf(z_values), label="Нормальное распределение")
    plt.fill_between(z_values, 0, stats.norm.pdf(z_values), where=(z_values <= critical_z1) | (z_values >= critical_z2),
                     color='red', alpha=0.5, label='Критическая область')
    plt.axvline(z, color='blue', label=f'Z={z:.2f}')
    plt.legend()
    plt.title(f'Z-тест: p-value={p_value:.4f}, вердикт={verdict}')
    plt.xlabel('Z')
    plt.ylabel('Плотность вероятности')
    plt.grid(True)
    plt.show()

    return z, p_value, verdict


def fisher_test(n1, n2, var1, var2, alpha, type):
    """
                Тест Фишера
                --------------------------------------
                H0: var1 == var2
                type == 0: H1: var1 != var2
                type == 1: H1: var1 >= var2
                type == -1: H1: var1 <= var2
    """
    F = var1 / var2 if var1 > var2 else var2 / var1
    df1 = n1 - 1
    df2 = n2 - 1

    if type == 0:  # двусторонний тест
        critical_F1 = stats.f.ppf(alpha / 2, df1, df2)
        critical_F2 = stats.f.ppf(1 - alpha / 2, df1, df2)
        p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
    elif type == 1:  # правосторонний тест
        critical_F1 = stats.f.ppf(1 - alpha, df1, df2)
        critical_F2 = float('inf')
        p_value = 1 - stats.f.cdf(F, df1, df2)
    else:  # левосторонний тест
        critical_F1 = float('-inf')
        critical_F2 = stats.f.ppf(alpha, df1, df2)
        p_value = stats.f.cdf(F, df1, df2)

    if F < critical_F1 or F > critical_F2:
        verdict = "отвергается"
    else:
        verdict = "принимается"

    f_values = np.linspace(0.1, 5, 300)
    plt.figure(figsize=(10, 5))
    plt.plot(f_values, stats.f.pdf(f_values, df1, df2), label="F-распределение")
    plt.fill_between(f_values, 0, stats.f.pdf(f_values, df1, df2),
                     where=(f_values <= critical_F1) | (f_values >= critical_F2), color='red', alpha=0.5,
                     label='Критическая область')
    plt.axvline(F, color='blue', label=f'Z={F:.2f}')
    plt.legend()
    plt.title(f'Тест Фишера: p-value={p_value:.4f}, вердикт={verdict}')
    plt.xlabel('F')
    plt.ylabel('Плотность вероятности')
    plt.grid(True)
    plt.show()

    return F, p_value, verdict


def test_variances(n1, n2, var1, var2, alpha):
    """
    :return: False if var1 != var2 True if var1 == var2
    """
    F = var1 / var2 if var1 > var2 else var2 / var1
    df1 = n1 - 1
    df2 = n2 - 1
    critical_F1 = stats.f.ppf(1 - alpha, df1, df2)
    critical_F2 = float('inf')
    print(F, critical_F1)

    if F < critical_F1:
        return True
    else:
        return False


def t_test(n1, n2, mean1, mean2, var1, var2, alpha, type):
    if not test_variances(n1, n2, var1, var2, alpha):
        print("Дисперсии неравны")
        return

    k = n1 + n2 - 2
    # k = min(n1 - 1, n2 - 1)  # используем приближенные степени свободы
    t = ((mean1 - mean2) / np.sqrt((n1 - 1) * var1 + (n2 - 1) * var2)) * np.sqrt((n1 * n2 * k) / (n1 + n2))

    if type == 0:  # двусторонний тест
        critical_t1 = stats.t.ppf(alpha / 2, k)
        critical_t2 = stats.t.ppf(1 - alpha / 2, k)
        p_value = 2 * (1 - stats.t.cdf(np.abs(t), k))
    elif type == 1:  # правосторонний тест
        critical_t1 = stats.t.ppf(1 - alpha, k)
        critical_t2 = float('inf')
        p_value = 1 - stats.t.cdf(t, k)
    else:  # левосторонний тест
        critical_t1 = float('-inf')
        critical_t2 = stats.t.ppf(alpha, k)
        p_value = stats.t.cdf(t, k)

    if t < critical_t1 or t > critical_t2:
        verdict = "отвергается"
    else:
        verdict = "принимается"

    t_values = np.linspace(t - n1 / 3, t + n1 / 3, 300)
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, stats.t.pdf(t_values, k), label="t-распределение")
    plt.fill_between(t_values, 0, stats.t.pdf(t_values, k),
                     where=(t_values <= critical_t1) | (t_values >= critical_t2), color='red', alpha=0.5,
                     label='Критическая область')
    plt.axvline(t, color='blue', label=f'Z={t:.2f}')
    plt.legend()
    plt.title(f'T-тест Стьюдента: p-value={p_value:.4f}, вердикт={verdict}')
    plt.xlabel('t')
    plt.ylabel('Плотность вероятности')
    plt.grid(True)
    plt.show()

    return t, p_value, critical_t1, verdict


# z_test(100, 100, 105, 100, 15 ** 2, 20 ** 2, 0.05, 0)
# fisher_test(30, 30, 15 ** 2, 20 ** 2, 0.05, 0)
# t_test(20, 20, 100, 105, 15 ** 2, 20 ** 2, 0.05, 0)
print(t_test(12, 18, 31.2, 29.2, 0.84, 0.40, 0.05, 0))
