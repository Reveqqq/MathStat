import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import norm

n = 100
X = np.linspace(0, 10, n)
Y = 2 * X + np.sin(X) + np.random.normal(scale=1, size=n)

# Линейная регрессия с срезом
X_lin_with_const = sm.add_constant(X)
model_lin_with_const = sm.OLS(Y, X_lin_with_const).fit()

print(model_lin_with_const.summary())

# Линейная регрессия без среза
model_lin_no_intercept = sm.OLS(Y, X).fit()
print(model_lin_no_intercept.summary())

print()

def nonlinear_func(X):
    return X ** 2

# # Нелинейная регрессия
# X_nonlin = nonlinear_func(X)
# X_nonlin = sm.add_constant(X_nonlin)
# model_nonlin = sm.OLS(Y, X_nonlin).fit()
# print(model_nonlin.summary())

# Полиномиальная регрессия второй степени
X_poly = np.column_stack((X**2, X))
X_poly_with_const = sm.add_constant(X_poly)
model_nonlin = sm.OLS(Y, X_poly_with_const).fit()
print(model_nonlin.summary())

print()


# Проверка условий теоремы Гаусса-Маркова
def check_gauss_markov_conditions(model):
    residuals = model.resid

    # Мат ожидание остатков
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals, ddof=1)
    z_stat = mean_residuals / (std_residuals / np.sqrt(len(residuals)))
    p_value_ztest = 2 * (1 - norm.cdf(np.abs(z_stat)))

    print(f"Mean of residuals: {mean_residuals}")
    print(f"z-statistic: {z_stat}, p-value: {p_value_ztest}")
    if p_value_ztest > 0.05:
        print("Вердикт: Матожидание остатков не отличается от нуля. Условие выполнено.")
    else:
        print("Вердикт: Матожидание остатков отличается от нуля. Условие не выполнено.")

    # Дисперсия остатков
    _,pval, _ , _ = het_breuschpagan(residuals, model.model.exog)
    print(f"p-значение теста Бреуша-Пагана: {pval}")
    if pval > 0.05:
        print("Дисперсия остатков одинакова для линейной модели (гомоскедастичность).")
    else:
        print("Дисперсия остатков неодинакова для линейной модели (гетероскедастичность).")

    # Корреляция остатков (тест Дарбина-Уотсона)
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat}")

    if dw_stat < 1.5 or dw_stat > 2.5:
        print("Остатки линейной модели коррелированы.")
    else:
        print("Остатки линейной модели некоррелированы.")

    return mean_residuals, p_value_ztest, pval, dw_stat


# Проверка для линейной регрессии
print("Линейная регрессия:")
check_gauss_markov_conditions(model_lin_with_const)

print()

# Проверка для нелинейной регрессии
print("Нелинейная регрессия:")
check_gauss_markov_conditions(model_nonlin)

# Построение графиков
plt.figure(figsize=(14, 7))

# Исходные данные
plt.scatter(X, Y, label='Data')

# Линейная регрессия
plt.plot(X, model_lin_no_intercept.predict(X), label='Linear Fit', color='red')

# Полиномиальная регрессия
plt.plot(X, model_nonlin.predict(X_poly_with_const), label='Polynomial Fit', color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Time Series Analysis with Linear and Polynomial Regression')
plt.show()
