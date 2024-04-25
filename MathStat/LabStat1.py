import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

un_100 = sp.uniform(1, 100).rvs(size=100)
un_1000 = sp.uniform(1, 100).rvs(size=1000)

bern_100 = sp.bernoulli(p=0.5).rvs(size=100)
bern_1000 = sp.bernoulli(p=0.5).rvs(size=1000)

bin_100 = sp.binom(n=100, p=0.5).rvs(size=100)
bin_1000 = sp.binom(n=100, p=0.5).rvs(size=1000)

norm_100 = sp.norm(loc=0, scale=1).rvs(size=100)
norm_1000 = sp.norm(loc=0, scale=1).rvs(size=1000)

distributions = [un_100, un_1000, bern_100, bern_1000, bin_100, bin_1000, norm_100, norm_1000]

# мат ожидание
print("\nmeans:")
for dist in distributions:
    print(dist.mean())

# дисперсия
print("\nvars:")
for dist in distributions:
    print(dist.var())

# medians
print("\nmedians:")
for dist in distributions:
    print(np.median(dist))

# среднеквадратичное отклонение
print("\nsquares:")
for dist in distributions:
    print(np.std(dist))

for dist in distributions:
    plt.hist(dist, bins=20, density=True)
    plt.show()