
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
import pandas as pd #collection of functions for data processing and 
# analysis modeled after R dataframes with SQL like features
import matplotlib #collection of functions for scientific and publication-ready visualization
import numpy as np #foundational package for scientific computing
import scipy as sp #collection of functions for scientific computing and advance mathematics
# numpyより高度なことができる。フーリエ変換とかね。
import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
# https://qiita.com/5t111111/items/7852e13ace6de288042f
# ipythonは便利なシェル？
import sklearn #collection of machine learning algorithms
# misc libraries
# mist = その他
import random
import time
# ignore warning
import warnings
warnings.filterwarnings('ignore')
# よくわからん原因不明の警告が表示されることがあるので、それを防止できる。

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Any results you write to the current directory are saved as output.


# In[3]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().magic('matplotlib inline')
mpl.style.use('ggplot')  # なんかよくわからんが、プロットのスタイル
sns.set_style('white')  # 背景白、グリッドなし
pylab.rcParams['figure.figsize'] = 12,8  # matplotの調整


# In[6]:


#import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_raw = pd.read_csv('../input/train.csv')
print('--------train data---------')
print(data_raw)
#a dataset should be broken into 3 splits: train, test, and (final) validation
#the test file provided is the validation file for competition submission
#we will split the train set into train and test data in future sections
data_val = pd.read_csv('../input/test.csv')
print('--------test data---------')
print(data_val)
#to play with our data we'll create a copy
#remember python assignment or equal passes by reference vs values,
# so we use the copy function: 
# https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep = True)
# http://kurochan-note.hatenablog.jp/entry/20110316/1300267023
print('--------data1---------')
print(data1)

#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]
# 一度にデータがそうさできて楽なんで、まとめる。
print('--------data_cleaner---------')
print(data_cleaner)

#preview data
print('---------raw info----------')
print (data_raw.info()) 
print('---------samples----------')
data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html


# In[12]:


# nullを見に行く
# print('-'*12, 'Train:\n', data1.isnull())  # 中身がnullか調べて、nullならtrue 
print('Train column has with null values:\n', data1.isnull().sum())
# .sum()で、各columsごとに、Trueの数を数えてる
print('-'*10)
print('Test column has with null values:\n', data_val.isnull().sum())
# .sum()で、各columsごとに、Trueの数を数えてる
print('-'*10)


# In[15]:


data_raw.describe(include = 'all')
# uniqueはデータの種類(bool)
# top：最頻値
# uniqueとかtopはboolにのみ作用できる。数値データとかは無理ぽい
# freque：最も一般的な値のでる回数


# In[22]:


# age, cabin, embarkedの欠損を埋める
# ageは数値データ、cabinはcategorical
# age, embarkedを埋める。cabinは消す.cabinは欠損が多すぎる。
# print(data1['Cabin'])
# print(data1['Embarked'])
for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace =  True)
    # inpulaceで元のオブジェクト自体が変更される。
    dataset['Embarked'].fillna(dataset['Embarked'].mode(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
data_column = ['Passengers', 'Cabin', 'Ticket']
data1.drop(data_column, axis=1,  inplace = True)
#dropはindexを指定して削除できる。axis=1だと列で削除



# In[24]:


data_raw.describe(include = 'all')


# In[25]:


print(data1)


# In[29]:


###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    dataset['Familysize'] = dataset['SibSp'] + dataset['SibSp'] + 1
    dataset['IsAlone'] = 1 # initialize by 1
    dataset['IsAlone'].loc[dataset['Familysize'] > 1] = 0
    # loc[]なのに注意参照元.loc[指定したい条件]
    # loc[]で、条件に合うリストをとってこれる
    dataset['Title'] = dataset['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
    # pandasはstr型のデータに当たるとき、.str＝strアクセを使うと各データに対して処理を適用できる
    # expand=Trueで複数の列に分割できる
    # qcut, cutは連続値を任意の境界で区分けして離散値にする。=ビニング処理をする
    # qcutはビンに含まれる要素数を等しくビニング処理
    # cutは最大値と最小値の間で値の間隔が等しくなるよう分割
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4)
    # astypeでpandasの中身をキャストできる
    dataset['Agebin'] = pd.cut(dataset['Age'].astype(int), 5)


# In[32]:


#cleanup rare title names
stat_min = 10
# print(data1['Title'].value_counts())
title_name = (data1['Title'].value_counts() < stat_min)
# 各タイトルに対して、出現数をカウントしたリストが生成され、その各要素に対して、判定
# print(title_name)


# In[34]:


# rareなタイトルなら、その他=Miscとする。
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_name.loc[x] == True else x)
# apply()各列に関数を適用


# In[36]:


print(data1['Title'].value_counts())


# In[37]:


data1.info()
# 「有効データ数」「データ型」「メモリ使用量」などの総合的な情報を表示
# なんかよくわからんがobjectはstrでcategoryは値の範囲という理解にしとこう。無理。


# In[42]:


print(data1['Title'],data1['Cabin'],  data1['Farebin'])


# In[43]:


data_val.info()


# In[44]:


data1.sample(10)


# In[54]:


drop_column = ['PassengerId', 'Cabin', 'Ticket']
# cagin=客室
data1.drop(drop_column, axis=1, inplace=True)


# In[51]:


# sklearnとpandasを使って、カテゴリカルをダミーに
print(data_cleaner)


# In[55]:


data1.info()


# In[53]:


# sex、embarked、title(object)と、farebin、Agebin(category)を変換
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = 

