import pandas as pd
import scipy.stats as sts
import numpy as np
import re


def convert_google_sheet_url(url):
    """
    Конвертирует url google sheets в csv format чтобы распарсить пандасом
    ---------------------------------------------------------------------
    Вернет законверченный url
    """
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (
        f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'

    new_url = re.sub(pattern, replacement, url)

    return new_url


def parse_google_sheets(urls):
    '''
    Парсит google sheets табличку и вносит в датафрейм пандас
    '''
    df = pd.DataFrame(columns=['anxiety', 'social phobia', 'spiders', 'superiors', 'future', 'responsibility'])
    for num, url in enumerate(open('urls'), 1):
        new_url = convert_google_sheet_url(url)
        temp = pd.read_csv(new_url)
        if num == 1:
            df['anxiety'] = temp['Общая']
        elif num == 2:
            df['social phobia'] = temp['Общий']
        else:
            df['spiders'] = temp['Пауки']
            df['superiors'] = temp['Начальство']
            df['future'] = temp['Будущее']
            df['responsibility'] = temp['Ответственность']
    return df


def find_correlation(x, y, mode='pearson'):
    """
    :param x: нампаевский массив
    :param y: нампаевский массив
    :param mode: как рассчитывать кореляцию (spearman, pearson)
    :return: выборочный коэффицент корреляции
    """
    if mode not in ('spearman', 'pearson'):
        raise ValueError('Неизвестная формула для вычисления!')

    if mode == 'spearman':
        built_in_corr = sts.spearmanr(x, y).statistic
        x = sts.rankdata(x)
        y = sts.rankdata(y)
    else:
        built_in_corr = sts.pearsonr(x, y).statistic
    mu_x = x.mean()
    mu_y = y.mean()
    n = x.size
    numerator = ((x - mu_x) * (y - mu_y)).sum()
    denominator = np.sqrt(((x - mu_x) ** 2).sum() * ((y - mu_y) ** 2).sum())
    print(f'Вычисление встроенным методом {built_in_corr}')
    print(f'Вычисление по формуле {numerator / denominator}')
    assert round((built_in_corr), 15) == round((numerator / denominator), 15)
    return numerator / denominator


def correlation_test(x, y):
    """
    Проверяет гипотезу о значимости выборочного коэффицента корреляции
    H0: r(x,y) = 0
    H1: r(x,y) != 0
    :param x: выборка 1
    :param y: выборка 2

    Выводит pvalue провеки на нормальность с названием критерия, вердикт о распределении,
    используемый коэффициент корреляции, его значение, pvalue, вердикт значимости.
    Вернет corr, pvalue
    """
    _, p1_x = sts.kstest(x, 'norm', args=(x.mean(), x.std(ddof=1)))
    _, p1_y = sts.kstest(y, 'norm', args=(y.mean(), y.std(ddof=1)))
    _, p2_x = sts.shapiro(x)
    _, p2_y = sts.shapiro(y)
    if all((p1_x >= 0.05, p1_y >= 0.05, p2_x >= 0.05, p2_y >= 0.05)):
        corr = find_correlation(x, y)
        print('(x,y) распределены нормально на уровне значимости 0.05')
        print('Используем корреляцию Пирсона')
        print(sts.pearsonr(x, y))
    else:
        corr = find_correlation(x, y, mode='spearman')
        print('(x,y) не распределены нормально на уровне значимости 0.05')
        print('Используем корреляцию Спирмена')
        print(sts.spearmanr(x, y))
    print(f'Тест Колмогорова для x с p_value = {p1_x}')
    print(f'Тест Колмогорова для y с p_value = {p1_y}')
    print(f'Тест Шапиро-Уилка для x с p_value = {p2_x}')
    print(f'Тест Шапиро-Уилка для y с p_value = {p2_y}')
    statistic = (corr * np.sqrt(x.size - 2)) / (np.sqrt(1 - corr ** 2))
    p_value = 2 * (sts.t(x.size - 2).sf(np.abs(statistic)))
    print(f'{corr = }, {p_value = }')
    if p_value < 0.05:
        print('На уровне значимости 0.05 H0 отвергается')
    else:
        print('На уровне значимости 0.05 H0 принимается')
    print('-' * 45)
    return corr, p_value


def blt_in_cmp(res, x, y, mode='pearson'):
    '''
    Сравнивает результаты ручного теста со встроенным тестом
    '''
    stat, pval = res
    if mode == 'pearson':
        blt_in_stat, blt_in_pval = sts.pearsonr(x, y)
    else:
        blt_in_stat, blt_in_pval = sts.spearmanr(x, y)
    assert (round(stat, 15), round(pval, 15)) == (round(blt_in_stat, 15), round(blt_in_pval, 15))


if __name__ == '__main__':
    df = parse_google_sheets(open('urls'))
    res1 = correlation_test(df['anxiety'].values, df['social phobia'].values)
    res2 = correlation_test(df['spiders'].values, df['superiors'].values)
    res3 = correlation_test(df['future'].values, df['responsibility'].values)
    blt_in_cmp(res1, df['anxiety'].values, df['social phobia'].values)
    blt_in_cmp(res2, df['spiders'].values, df['superiors'].values, mode='spearman')
    blt_in_cmp(res3, df['future'].values, df['responsibility'].values, mode='spearman')
