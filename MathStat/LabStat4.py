import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def corr_coff_plot(corr_features, text=''):
    sns.regplot(x=corr_features[0], y=corr_features[1], data=wine_quality)
    plt.title(f'Диаграмма рассеяния с линией регрессии {text}')
    plt.xlabel(corr_features[0])
    plt.ylabel(corr_features[1])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    wine_quality = pd.read_csv('./../Wine_quality/winequality-red.csv', sep=';')

    # Описание
    print(wine_quality.describe())

    # Гистограммы
    for i, column in enumerate(wine_quality.columns):
        wine_quality[column].hist()
        # plt.hist(wine_quality[column])
        # sns.histplot(wine_quality[column], kde=True)
        plt.xlabel(column)
        plt.ylabel('frequency')
        plt.title(column)
        plt.tight_layout()
        plt.show()

    # Матрица корреляций
    corr_matrix = wine_quality.corr()
    print(corr_matrix)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    np.fill_diagonal(corr_matrix.values, np.nan)

    # Max, min коэф корреляции по модулю
    max_corr_features = corr_matrix.abs().stack().idxmax()
    min_corr_features = corr_matrix.abs().stack().idxmin()
    corr_coff_plot(max_corr_features, 'Max_corr')
    corr_coff_plot(min_corr_features, 'Min_corr')

    # Независимые признаки
    X = wine_quality.iloc[:, :-1]
    # Зависимый признак качество
    y = wine_quality.iloc[:, -1]
    # Добавили 1 чтобы получить модель линейной регрессии
    # X = sm.add_constant(X)

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    predictions = results.predict(X.iloc[:3])
    print("Predictions: \n", predictions)
    print("True values:", y.iloc[:3].values)
