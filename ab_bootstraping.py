import pandas as pd
import numpy as np
from scipy import stats
from random import randint
from bootstrapped import bootstrap as bs
from bootstrapped import compare_functions as bs_compare
from bootstrapped import stats_functions as bs_stats
from datetime import datetime as dt

import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('data/test.csv', sep = '\t')


# Проверка на нормальность - критерий Шапиро:
def shapiro_norm_test(df, colname = 'target', pvalue = 0.05):
    st = stats.shapiro(df)
    print(st)
    if st[1] < pvalue:
        print('{} {} is NOT normal\n'.format('None' if df.name is None else df.name.upper(), colname.upper()))
    else:
        print('{} {} is normal\n'.format('None' if df.name is None else df.name.upper(), colname.upper()))


# Проверка на нормальность - критерий Колмагорова-Смирнова:
def kstest_norm_test(df, colname = 'target', pvalue = 0.05):
    st = stats.kstest(df, 'norm')
    print(st)
    if st[1] < pvalue:
        print('{} {} is NOT normal\n'.format('None' if df.name is None else df.name.upper(), colname.upper()))
    else:
        print('{} {} is normal\n'.format('None' if df.name is None else df.name.upper(), colname.upper()))


# Проверка на равенство дисперсий:
def barlett_test(df1, df2, pvalue = 0.05):
    st = stats.bartlett(df1, df2)
    if st[1] < pvalue:
        print('Dispersias (Variances) of {} {} is NOT equals\n'.format('None' if df1.name is None else df1.name.upper(), 'CRITEO and NOT CRITEO'))
    else:
        print('Dispersias (Variances) of {} {} is equals\n'.format('None' if df1.name is None else df1.name.upper(), 'CRITEO and NOT CRITEO'))


# Рассчёт стат.значимости:
def stat_test(df1, df2, test='ttest'):
    if test=='ttest':
        st, pval = stats.ttest_ind(df1, df2)
    elif test=='mannwhitney':
        st, pval = stats.mannwhitneyu(df1, df2)
    else:
        return 0

    print('%s Statistic: %s\tAvg: %s' % (test, st, pval))
    if ((pval >= 0.05) and (st is not None)):
        print('%s Same average'  % (test))
    elif ((pval < 0.05) and (st is not None)):
        print('%s Different average' % (test))
    else:
        print('%s is not applicapable' % (test))
    return 0

# 1. Sub-bucket Method
data['subbucket'] = data['user_id'].apply(lambda x: randint(0,1000)) # Variant 1
data['subbucket'] = data['user_id'].apply(lambda x: hash(x)%1000) # Variant 2

# 2. Bootstrap Method
data_a = data[data['group'] == 'experiment_buckets']
data_b = data[data['group'] == 'control_buckets']
bs_ab_estims = bs.bootstrap_ab(data_a.groupby(data['user_id']).target.sum().values,
                               data_b.groupby(data['user_id']).target.sum().values,
                                   bs_stats.mean,
                                   bs_compare.percent_change, num_iterations=5000, alpha=0.10,
                                   iteration_batch_size=100, scale_test_by=1, num_threads=4)

bs_data_a = bs.bootstrap(data_a.groupby(data['user_id']).target.sum().values,
                         stat_func=bs_stats.mean, num_iterations=10000, iteration_batch_size=300,
                         return_distribution=True)
bs_data_b = bs.bootstrap(data_b.groupby(data['user_id']).target.sum().values,
                         stat_func=bs_stats.mean, num_iterations=10000, iteration_batch_size=300,
                         return_distribution=True)